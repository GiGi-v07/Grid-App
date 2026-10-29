import os
import sys
import subprocess as sp
import xmlrpc.client as Client
import threading as thread
import socket
from xmlrpc.server import SimpleXMLRPCServer as Server, SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn
import queue
import uuid
import time
import AITrainer

# Ensure console output is line-buffered so worker logs appear immediately
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

# Thread-local storage to track per-request client IP and outbound transfer stats
_thread_local = thread.local()

# Shared state for tracking file uploads and worker queue depth across threads
_upload_tracker = {}
_reported_uploads = set()
_upload_lock = thread.Lock()

_queue_lock = thread.Lock()
_queued_count = 0
_enqueued_jobs = {}


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def format_bytes(num_bytes):
    """Formats a raw byte count into a human-readable string (B, KB, or MB)."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    return f"{num_bytes / (1024 * 1024):.2f} MB"


def get_local_ip():
    """Detects and returns the active local LAN IP address of this machine."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except Exception:
        try:
            return str(socket.gethostbyname(socket.gethostname()))
        except Exception:
            return "127.0.0.1"


# ==============================================================================
# XML-RPC SERVER & REQUEST HANDLER CLASSES
# ==============================================================================

class CleanRequestHandler(SimpleXMLRPCRequestHandler):
    """
    Custom XML-RPC request handler that records the caller's IP address,
    suppresses noisy default HTTP logs, and prints outbound transfer stats
    when a job finishes sending its response back to the Master.
    """
    def do_POST(self):
        client_ip = self.client_address[0]
        _thread_local.client_ip = client_ip
        _thread_local.is_job_request = False
        _thread_local.response_len = 0
        super().do_POST()
        if getattr(_thread_local, "is_job_request", False):
            job_name = getattr(_thread_local, "job_name", "Job")
            send_time = max(0.001, time.time() - getattr(_thread_local, "job_done_time", time.time()))
            resp_size = getattr(_thread_local, "response_len", 0)
            print(f"[TRANSFER OUT] Sent {job_name} response to {client_ip} | Size: {format_bytes(resp_size)} | Outbound Transfer: {send_time:.2f}s", flush=True)

    def log_message(self, format, *args):
        # Suppress raw HTTP POST logs to keep the worker console clean
        pass


class ThreadedXMLRPCServer(ThreadingMixIn, Server):
    """Multi-threaded XML-RPC server that handles concurrent uploads and requests."""
    daemon_threads = True


# ==============================================================================
# GLOBAL MASTER & WORKER CONFIGURATION VARIABLES
# ==============================================================================

wkr_ips = set()                       # Set of active Worker IP addresses registered on Master
test = "test.py"                      # Handshake verification script
job_queue = queue.Queue()             # FIFO queue holding jobs waiting to execute on Worker
master_ip = '.'.join(get_local_ip().split('.')[:-1]) + '.'  # Local subnet prefix (e.g. '192.168.1.')

# Configurable paths for Single Job and AI Training (passed to AITrainer)
single_script_path = "Script.py"
ai_script_path = "ai_script.py"
ai_data_path = "ai_data.txt"


# ==============================================================================
# FILE TRANSFER & WORKER EXECUTION FUNCTIONS
# ==============================================================================

def upload_chunk(filename, chunk, mode, is_last=False, job_id=None, job_name=None):
    """
    RPC function called on the Worker to receive a 1MB file chunk from the Master.
    - On the first chunk ('wb'), registers the job in the queue and logs '[QUEUE +1]'
      and '[TRANSFER IN] Receiving...'.
    - Writes the chunk bytes to disk.
    - On the final chunk (is_last=True), immediately logs the completed transfer stats.
    """
    global _queued_count
    client_ip = getattr(_thread_local, "client_ip", "Unknown")
    filepath = os.path.join(os.path.dirname(__file__), filename)
    data = chunk.data if hasattr(chunk, "data") else chunk

    with _upload_lock:
        now = time.time()
        is_first_chunk = (mode == "wb" or filename not in _upload_tracker)
        if is_first_chunk:
            _reported_uploads.discard(filename)
            _upload_tracker[filename] = {"start": now, "bytes": len(data), "last": now}
        else:
            _upload_tracker[filename]["bytes"] += len(data)
            _upload_tracker[filename]["last"] = now

    # When the first chunk of a job arrives, enqueue it immediately and log file reception
    if is_first_chunk:
        key = job_id if job_id else filename
        with _queue_lock:
            if key not in _enqueued_jobs and (job_id or filename.startswith("temp_job")):
                resolved_name = job_name or ("Virtual Handshake" if filename == "temp_job.py" else f"Job ({filename})")
                _enqueued_jobs[key] = resolved_name
                _queued_count += 1
                print(f"[QUEUE +1    ] Enqueued {resolved_name} from {client_ip} | Queue Size: {_queued_count}", flush=True)
        print(f"[TRANSFER IN ] Receiving {filename} from {client_ip}...", flush=True)

    with open(filepath, mode) as f:
        f.write(data)

    if is_last or len(data) < (1024 * 1024):
        report_inbound_transfer(filename, client_ip)

    return True


def report_inbound_transfer(filename, client_ip):
    """Logs the total size and transfer duration once a file finishes uploading to the Worker."""
    with _upload_lock:
        if filename in _reported_uploads:
            return
        info = _upload_tracker.pop(filename, None)
        if info is not None:
            _reported_uploads.add(filename)

    if info:
        elapsed = max(0.001, info["last"] - info["start"])
        print(f"[TRANSFER IN ] Received {filename} from {client_ip} | Size: {format_bytes(info['bytes'])} | Inbound Transfer: {elapsed:.2f}s", flush=True)


def upload_file_in_chunks(worker, local_path, remote_filename, job_id=None, job_name=None):
    """
    Master-side function that reads a local file in 1MB binary chunks and
    streams it to the Worker via the 'upload_chunk' XML-RPC method.
    """
    t0 = time.time()
    total_bytes = 0
    file_size = os.path.getsize(local_path)
    with open(local_path, "rb") as f:
        mode = "wb"
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            total_bytes += len(chunk)
            is_last = total_bytes >= file_size
            worker.upload_chunk(remote_filename, Client.Binary(chunk), mode, is_last, job_id, job_name)
            mode = "ab"
    elapsed = max(0.001, time.time() - t0)
    print(f"[TRANSFER OUT] Uploaded {remote_filename} | Size: {format_bytes(total_bytes)} | Upload Time: {elapsed:.2f}s", flush=True)


def _run_job_file(filepath):
    """
    Executes a received Python script in a subprocess on the Worker, deletes
    the temporary script file afterward, and returns its console output.
    """
    result = sp.run(["python", filepath], capture_output=True, text=True)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass
    if result.returncode != 0 and result.stderr:
        return (result.stdout + result.stderr) if result.stdout else result.stderr
    return result.stdout


def execute_job(job_target="temp_job.py", job_id=None):
    """
    RPC function called on the Worker to place an uploaded script into the
    execution queue, wait for the background worker thread to run it, and
    return the resulting output to the Master.
    """
    global _queued_count
    client_ip = getattr(_thread_local, "client_ip", "Unknown")
    scriptdir = os.path.dirname(__file__)
    target_path = os.path.join(scriptdir, job_target)
    base_name = os.path.basename(target_path)

    report_inbound_transfer(base_name, client_ip)

    # Retrieve the job name registered during chunk upload (or enqueue now if called directly)
    key = job_id if (job_id and job_id in _enqueued_jobs) else base_name
    with _queue_lock:
        job_name = _enqueued_jobs.pop(key, None)
        if not job_name:
            job_name = "Virtual Handshake" if base_name == "temp_job.py" else f"Job ({base_name})"
            _queued_count += 1
            print(f"[QUEUE +1    ] Enqueued {job_name} from {client_ip} | Queue Size: {_queued_count}", flush=True)

    res_q = queue.Queue()
    job_queue.put((_run_job_file, (target_path,), res_q, client_ip, job_name))

    success, result = res_q.get()

    # Store metadata on thread-local storage for CleanRequestHandler outbound logging
    _thread_local.is_job_request = True
    _thread_local.job_name = job_name
    _thread_local.job_done_time = time.time()
    _thread_local.response_len = len(result.encode("utf-8")) if isinstance(result, str) else 0

    if not success:
        raise Exception(result)
    return result


def get_queue_size():
    """RPC function returning the current number of queued jobs on the Worker."""
    with _queue_lock:
        return _queued_count


# ==============================================================================
# MASTER DISPATCH & MANAGEMENT MENU FUNCTIONS
# ==============================================================================

def helper(i, ip, scriptpath):
    """
    Master-side dispatcher that connects to a specific Worker IP:
    - i == 0: Sends 'test.py' as a handshake check to verify the Worker is online.
    - i == 1: Uploads and executes a standard single Python job on the Worker.
    """
    if not os.path.isabs(scriptpath):
        scriptpath = os.path.join(os.path.dirname(__file__), scriptpath)

    try:
        worker = Client.ServerProxy(f"http://{ip}:8000", allow_none=True)
        uid = uuid.uuid4().hex[:8]
        job_id = f"job_{uid}"

        if i == 0:
            upload_file_in_chunks(worker, scriptpath, "temp_job.py", job_id=job_id, job_name="Virtual Handshake")
            t_call = time.time()
            response = worker.execute_job("temp_job.py", job_id)
            rt = time.time() - t_call
            print(f"[TRANSFER IN ] Received response from {ip} | Size: {format_bytes(len(response))} | Round-Trip: {rt:.2f}s", flush=True)
            if response == "Test\n":
                print(f"Worker {ip} is online and ready.", flush=True)
            else:
                print(f"Worker {ip} responded with unexpected output: {response}", flush=True)

        elif i == 1:
            print(f"Uploading script to {ip}...", flush=True)
            job_file = f"temp_job_{uid}.py"
            job_name = f"Job ({job_file})"
            upload_file_in_chunks(worker, scriptpath, job_file, job_id=job_id, job_name=job_name)
            t_call = time.time()
            response = worker.execute_job(job_file, job_id)
            rt = time.time() - t_call
            print(f"[TRANSFER IN ] Received response from {ip} | Size: {format_bytes(len(response))} | Round-Trip: {rt:.2f}s", flush=True)
            print(f"Worker output from {ip}:\n{response}", flush=True)

    except Exception as e:
        print(f"Error with worker {ip}: {e}", flush=True)


def single_job(name=None):
    """
    Sends a single Python script in parallel threads to all registered Workers
    and waits for all Workers to finish executing it.
    """
    script_to_run = name if name else single_script_path
    print(f"Starting distributed single job using script '{script_to_run}'...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = script_to_run if os.path.isabs(script_to_run) else os.path.join(scriptdir, script_to_run)
    if not os.path.exists(scriptpath):
        print(f"Error: Script file '{scriptpath}' not found.")
        return

    print("Sending job...")
    threads = [thread.Thread(target=helper, args=(1, ip, scriptpath)) for ip in list(wkr_ips)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def Menu():
    """
    Interactive console menu for Master node management:
    Allows adding/removing Worker IPs, configuring script/dataset paths,
    and launching Single Jobs or Distributed AI Training (via AITrainer.py).
    """
    global single_script_path, ai_script_path, ai_data_path
    while True:
        print("\n--- Master Node Management ---")
        print("(A)dd an IP")
        print("(R)emove an IP")
        print("(V)iew current IPs")
        print("(C)hange Script / Path")
        print("(S)tart Job")
        print("(Q)uit to Main Menu")
        menu_choice = input("Select an option: ").strip().upper()

        if menu_choice == 'A':
            ip = input("Enter the Worker IP to add: ").strip()
            new_ip = ip if (ip == 'localhost' or '.' in ip) else str(master_ip) + ip
            wkr_ips.add(new_ip)
            helper(0, new_ip, test)

        elif menu_choice == 'R':
            del_ip = input("Enter the Worker IP to remove: ").strip()
            if del_ip in wkr_ips:
                wkr_ips.remove(del_ip)
                print(f"Removed {del_ip}.")
            else:
                print("Error: IP not found in the list.")

        elif menu_choice == 'V':
            print(f"Current Worker IPs: {wkr_ips}")

        elif menu_choice == 'C':
            print("\n--- Configure Scripts & Paths ---")
            print(f"1. Single Job Script   : {single_script_path}")
            print(f"2. AI Training Script  : {ai_script_path}")
            print(f"3. AI Training Dataset : {ai_data_path}")
            print("4. Switch AI to ai_script.py / ai_data.txt")
            print("5. Reset all to defaults (Script.py / Data.txt)")
            print("6. Back to Master Menu")
            cfg = input("Select option (1-6): ").strip()
            if cfg == '1':
                val = input("Enter path/filename for Single Job script: ").strip()
                if val:
                    single_script_path = val
            elif cfg == '2':
                val = input("Enter path/filename for AI Training script: ").strip()
                if val:
                    ai_script_path = val
            elif cfg == '3':
                val = input("Enter path/filename for AI Training dataset: ").strip()
                if val:
                    ai_data_path = val
            elif cfg == '4':
                ai_script_path, ai_data_path = "ai_script.py", "ai_data.txt"
                print("AI job configured to use: ai_script.py and ai_data.txt")
            elif cfg == '5':
                single_script_path, ai_script_path, ai_data_path = "Script.py", "Script.py", "Data.txt"
                print("All scripts reset to defaults: Script.py and Data.txt")

        elif menu_choice == 'S':
            print("Single task, type (S)ingle")
            print("AI train, type (A)I")
            c = input("Choice: ").strip().upper()
            if c == 'S':
                single_job()
            elif c == 'A':
                # Delegate all AI training execution to AITrainer.py
                AITrainer.start_ai_training(wkr_ips, ai_script_path, ai_data_path, upload_file_in_chunks, format_bytes)
            else:
                print("Invalid choice")

        elif menu_choice == 'Q':
            print("Exiting Master mode...")
            break
        else:
            print("Invalid choice, please try again.")


# ==============================================================================
# WORKER SERVER MODE
# ==============================================================================

def worker():
    """
    Starts the Worker node:
    1. Displays the Worker's local IP address and port in the top banner.
    2. Runs a background consumer thread ('process_queue') to execute queued jobs sequentially.
    3. Starts the multi-threaded XML-RPC server on port 8000 until [ENTER] is pressed.
    """
    global _queued_count
    stop_event = thread.Event()
    worker_ip = get_local_ip()

    print("\n" + "=" * 62)
    print("                 WORKER NODE ONLINE")
    print(f"   Worker IP : {worker_ip} (Host Octet: {worker_ip.split('.')[-1]})")
    print("   Listening on Port 8000 | Managing Job Queue")
    print("   Press [ENTER] at any time to stop worker mode.")
    print("=" * 62 + "\n")

    def process_queue():
        """Background worker loop that pulls jobs from job_queue and runs them one by one."""
        global _queued_count
        while not stop_event.is_set():
            try:
                func, args, res_q, client_ip, job_name = job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            with _queue_lock:
                _queued_count = max(0, _queued_count - 1)
                q_remaining = _queued_count
            print(f"[EXECUTE     ] Starting {job_name} from {client_ip} | Queue Remaining: {q_remaining}", flush=True)

            start_time = time.time()
            try:
                result = func(*args)
                elapsed = time.time() - start_time
                with _queue_lock:
                    q_size = _queued_count
                print(f"[SUCCESS     ] Completed {job_name} from {client_ip} | Execution Time: {elapsed:.2f}s | Queue Size: {q_size}", flush=True)
                res_q.put((True, result))
            except Exception as e:
                elapsed = time.time() - start_time
                with _queue_lock:
                    q_size = _queued_count
                print(f"[ERROR       ] Failed {job_name} from {client_ip} | Execution Time: {elapsed:.2f}s | Error: {e} | Queue Size: {q_size}", flush=True)
                res_q.put((False, str(e)))
            finally:
                job_queue.task_done()

    queue_worker_thread = thread.Thread(target=process_queue, daemon=True)
    queue_worker_thread.start()

    server = ThreadedXMLRPCServer(("0.0.0.0", 8000), requestHandler=CleanRequestHandler, allow_none=True, logRequests=False)
    server.register_function(upload_chunk, "upload_chunk")
    server.register_function(execute_job, "execute_job")
    server.register_function(get_queue_size, "get_queue_size")

    server_thread = thread.Thread(target=server.serve_forever)
    server_thread.start()

    input()

    print("\nShutting down worker node...")
    server.shutdown()
    server_thread.join()
    try:
        server.server_close()
    except Exception:
        pass
    stop_event.set()
    queue_worker_thread.join(timeout=2)
    print("Worker stopped successfully.\n")


if __name__ == "__main__":
    while True:
        print("Run this laptop as (M)aster, (W)orker or (Q)uit Program?")
        ch = input("Select an option: ").strip().upper()
        if ch == 'Q':
            break
        elif ch == 'W':
            worker()
        elif ch == 'M':
            Menu()
        else:
            print("Invalid choice.")

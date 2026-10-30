import os
import sys
import json
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
# UTILITY & NETWORKING FUNCTIONS
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


def get_subnet_prefix():
    """
    Dynamically fetches the current Master IP's first 3 octets (e.g. '192.168.1.')
    each time an IP is added or removed so network changes are handled automatically.
    """
    local_ip = get_local_ip()
    return '.'.join(local_ip.split('.')[:-1]) + '.'


def resolve_worker_ip(user_input):
    """
    Converts a user-entered host octet (e.g. '42') into a full IP address using
    the current Master subnet prefix, while also supporting 'localhost' or full IPs.
    """
    val = user_input.strip()
    if val == "localhost" or "." in val:
        return val
    return get_subnet_prefix() + val


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

wkr_ips = set()                       # Set of verified Worker IP addresses
wkr_specs = {}                        # Dict mapping verified Worker IP -> hardware specs dict
handshake_script = "Handshake.py"     # Node Spec Handshake & hardware profiler script
job_queue = queue.Queue()             # FIFO queue holding jobs waiting to execute on Worker

# Configurable paths for Single Job and AI Training (passed to AITrainer)
single_script_path = "Script.py"
ai_script_path = "ai_script.py"
ai_data_path = "ai_data.txt"
TEMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Temp_Jobs")
os.makedirs(TEMP_DIR, exist_ok=True)


# ==============================================================================
# FILE TRANSFER & WORKER EXECUTION FUNCTIONS
# ==============================================================================

def upload_chunk(filename, chunk, mode, is_last=False, job_id=None, job_name=None):
    """
    RPC function called on the Worker to receive a 1MB file chunk from the Master.
    - On the first chunk ('wb'), registers the job in the queue and logs '[QUEUE +1]'
      and '[TRANSFER IN] Receiving...'.
    - Writes the chunk bytes into the 'Temp_Jobs/' directory.
    - On the final chunk (is_last=True), immediately logs the completed transfer stats.
    """
    global _queued_count
    client_ip = getattr(_thread_local, "client_ip", "Unknown")
    os.makedirs(TEMP_DIR, exist_ok=True)
    filepath = os.path.join(TEMP_DIR, os.path.basename(filename))
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
            if key not in _enqueued_jobs and (job_id or filename.startswith("temp_")):
                resolved_name = job_name or ("Node Spec Handshake" if filename == "temp_handshake.py" else f"Job ({filename})")
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
    os.makedirs(TEMP_DIR, exist_ok=True)
    base_name = os.path.basename(job_target)
    target_path = os.path.join(TEMP_DIR, base_name)

    report_inbound_transfer(base_name, client_ip)

    # Retrieve the job name registered during chunk upload (or enqueue now if called directly)
    key = job_id if (job_id and job_id in _enqueued_jobs) else base_name
    with _queue_lock:
        job_name = _enqueued_jobs.pop(key, None)
        if not job_name:
            job_name = "Node Spec Handshake" if base_name == "temp_handshake.py" else f"Job ({base_name})"
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
    - i == 0: Sends 'Handshake.py' (Node Spec Handshake) to verify the Worker is
              online and fetch its hardware specs. Only adds the IP to wkr_ips
              if the handshake succeeds.
    - i == 1: Uploads and executes a standard single Python job on the Worker.
    """
    if not os.path.isabs(scriptpath):
        scriptpath = os.path.join(os.path.dirname(__file__), scriptpath)

    try:
        worker = Client.ServerProxy(f"http://{ip}:8000", allow_none=True)
        uid = uuid.uuid4().hex[:8]
        job_id = f"job_{uid}"

        if i == 0:
            upload_file_in_chunks(worker, scriptpath, "temp_handshake.py", job_id=job_id, job_name="Node Spec Handshake")
            t_call = time.time()
            response = worker.execute_job("temp_handshake.py", job_id)
            rt = time.time() - t_call
            print(f"[TRANSFER IN ] Received response from {ip} | Size: {format_bytes(len(response))} | Round-Trip: {rt:.2f}s", flush=True)

            marker = "HANDSHAKE_SPECS:"
            specs = None
            for line in response.splitlines():
                if line.strip().startswith(marker):
                    specs = json.loads(line.strip()[len(marker):])
                    break

            if specs and specs.get("status") == "ONLINE":
                wkr_ips.add(ip)
                wkr_specs[ip] = specs
                print(
                    f"Worker {ip} is online and verified! | "
                    f"CPU: {specs['cores']} Cores ({specs['cpu_mops']} MOPS) | "
                    f"RAM: {specs['avail_ram_gb']}/{specs['total_ram_gb']} GB Free | "
                    f"Spec Score: {specs['spec_score']}",
                    flush=True,
                )
            else:
                print(f"Handshake failed: Worker {ip} returned unexpected output ({response.strip()}). IP was NOT added.", flush=True)

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
        print(f"Error connecting to worker {ip}: {e}", flush=True)
        if i == 0:
            print(f"Handshake failed for {ip}. Worker was NOT added to the active list.", flush=True)


def single_job(name=None):
    """
    Sends a single Python script in parallel threads to all registered Workers
    and waits for all Workers to finish executing it.
    """
    if not wkr_ips:
        print("Error: No worker IPs added. Please add at least one worker IP first.")
        return

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
    Allows adding/removing Worker IPs (dynamically resolving the local subnet
    prefix each time), viewing Worker hardware specs, configuring script paths,
    and launching Single Jobs or Distributed AI Training (via AITrainer.py).
    """
    global single_script_path, ai_script_path, ai_data_path
    while True:
        print("\n--- Master Node Management ---")
        print("(A)dd an IP")
        print("(R)emove an IP")
        print("(V)iew current IPs & Specs")
        print("(C)hange Script / Path")
        print("(S)tart Job")
        print("(Q)uit to Main Menu")
        menu_choice = input("Select an option: ").strip().upper()

        if menu_choice == 'A':
            ip_input = input("Enter the Worker IP to add (last octet, e.g. 42): ").strip()
            if not ip_input:
                continue
            new_ip = resolve_worker_ip(ip_input)
            # Only adds new_ip inside helper(0, ...) if Node Spec Handshake succeeds
            helper(0, new_ip, handshake_script)

        elif menu_choice == 'R':
            del_input = input("Enter the Worker IP to remove (last octet, e.g. 42): ").strip()
            if not del_input:
                continue
            candidate_ip = resolve_worker_ip(del_input)
            target_ip = None
            if candidate_ip in wkr_ips:
                target_ip = candidate_ip
            else:
                # Also match by final octet in case the local subnet changed
                for existing_ip in wkr_ips:
                    if existing_ip == del_input or existing_ip.split('.')[-1] == del_input:
                        target_ip = existing_ip
                        break

            if target_ip:
                wkr_ips.remove(target_ip)
                wkr_specs.pop(target_ip, None)
                print(f"Removed {target_ip}.")
            else:
                print(f"Error: Worker '{del_input}' not found in the active list.")

        elif menu_choice == 'V':
            if not wkr_ips:
                print("No active Workers currently registered.")
            else:
                print("\n--- Connected Workers & Hardware Specs ---")
                for ip in sorted(wkr_ips):
                    sp_info = wkr_specs.get(ip)
                    if sp_info:
                        print(
                            f"  * {ip} (Octet: {ip.split('.')[-1]}) -> "
                            f"CPU: {sp_info['cores']} Cores ({sp_info['cpu_mops']} MOPS) | "
                            f"RAM: {sp_info['avail_ram_gb']}/{sp_info['total_ram_gb']} GB | "
                            f"Spec Score: {sp_info['spec_score']}"
                        )
                    else:
                        print(f"  * {ip}")

        elif menu_choice == 'C':
            print("\n--- Configure Scripts & Paths ---")
            print(f"1. Single Job Script   : {single_script_path}")
            print(f"2. AI Training Script  : {ai_script_path}")
            print(f"3. AI Training Dataset : {ai_data_path}")
            print("4. Switch AI to ai_script.py / ai_data.txt")
            print("5. Reset Single Job to Script.py")
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
                single_script_path = "Script.py"
                print("Single Job script reset to: Script.py")

        elif menu_choice == 'S':
            print("Single task, type (S)ingle")
            print("AI train, type (A)I")
            c = input("Choice: ").strip().upper()
            if c == 'S':
                single_job()
            elif c == 'A':
                # Delegate all AI training execution and proportional data splitting to AITrainer.py
                AITrainer.start_ai_training(wkr_ips, ai_script_path, ai_data_path, upload_file_in_chunks, format_bytes, wkr_specs)
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

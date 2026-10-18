import os as os
import sys as sys
import subprocess as sp
import xmlrpc.client as Client
import threading as thread
import socket as socket
from xmlrpc.server import SimpleXMLRPCServer as Server, SimpleXMLRPCRequestHandler
from socketserver import ThreadingMixIn
import queue as queue
import uuid as uuid
import time as time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass

_thread_local = thread.local()


def format_bytes(num_bytes):
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.2f} MB"


_upload_tracker = {}
_upload_lock = thread.Lock()


class CleanRequestHandler(SimpleXMLRPCRequestHandler):
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
        # Suppress raw HTTP POST request logs to prevent console clutter
        pass


class ThreadedXMLRPCServer(ThreadingMixIn, Server):
    daemon_threads = True


# Variables
wkr_ips = set()  # Set to store worker IPs 
active_threads = []
test = "test.py"
hostname = socket.gethostname()
job_queue = queue.Queue()  # Queue to manage jobs
master_ip = str(socket.gethostbyname(hostname))
ip_parts = master_ip.split('.')
master_ip = '.'.join(ip_parts[:-1]) + '.'  # Get the first three octets of the IP address

# Configurable Script Paths (defaulting to Script.py and Data.txt)
single_script_path = "Script.py"
ai_script_path = "Script.py"
ai_data_path = "Data.txt"



def upload_chunk(filename, chunk, mode):
    client_ip = getattr(_thread_local, "client_ip", "Unknown")
    scriptdir = os.path.dirname(__file__)
    filepath = os.path.join(scriptdir, filename)
    data = chunk.data if hasattr(chunk, "data") else chunk

    with _upload_lock:
        now = time.time()
        if mode == "wb" or filename not in _upload_tracker:
            _upload_tracker[filename] = {
                "start": now,
                "bytes": len(data),
                "last": now,
                "ip": client_ip,
            }
        else:
            _upload_tracker[filename]["bytes"] += len(data)
            _upload_tracker[filename]["last"] = now

    with open(filepath, mode) as f:
        f.write(data)
    return True


def report_inbound_transfer(filename, client_ip):
    with _upload_lock:
        info = _upload_tracker.pop(filename, None)
    if info:
        elapsed = max(0.001, info["last"] - info["start"])
        size_str = format_bytes(info["bytes"])
        print(f"[TRANSFER IN ] Received {filename} from {client_ip} | Size: {size_str} | Inbound Transfer: {elapsed:.2f}s", flush=True)
    elif os.path.exists(os.path.join(os.path.dirname(__file__), filename)):
        size = os.path.getsize(os.path.join(os.path.dirname(__file__), filename))
        print(f"[TRANSFER IN ] Received {filename} from {client_ip} | Size: {format_bytes(size)}", flush=True)


def upload_file_in_chunks(worker, local_path, remote_filename):
    t0 = time.time()
    total_bytes = 0
    with open(local_path, "rb") as f:
        mode = "wb"
        while True:
            chunk = f.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            worker.upload_chunk(remote_filename, Client.Binary(chunk), mode)
            mode = "ab" # append for subsequent chunks
    elapsed = max(0.001, time.time() - t0)
    print(f"[TRANSFER OUT] Uploaded {remote_filename} | Size: {format_bytes(total_bytes)} | Upload Time: {elapsed:.2f}s", flush=True)


def _run_job_file(filepath):
    result = sp.run(["python", filepath], capture_output=True, text=True)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass
    if result.returncode != 0 and result.stderr:
        return result.stdout + result.stderr if result.stdout else result.stderr
    return result.stdout


def _run_ai_job_file(scriptpath, datapath, l_rate):
    result = sp.run(["python", scriptpath, datapath, str(l_rate)], capture_output=True, text=True)
    if os.path.exists(scriptpath):
        try:
            os.remove(scriptpath)
        except OSError:
            pass
    if os.path.exists(datapath):
        try:
            os.remove(datapath)
        except OSError:
            pass
    if result.returncode != 0 and result.stderr:
        return result.stdout + result.stderr if result.stdout else result.stderr
    return result.stdout


def is_handshake_target(target_path):
    try:
        if os.path.exists(target_path):
            with open(target_path, "r") as f:
                content = f.read().strip()
                if content in ('print("Test")', "print('Test')", 'print("Test\\n")'):
                    return True
        elif target_path.strip() in ('print("Test")', "print('Test')", 'print("Test\\n")'):
            return True
    except Exception:
        pass
    return False


def execute_job(job_target="temp_job.py"):
    client_ip = getattr(_thread_local, "client_ip", "Unknown")
    scriptdir = os.path.dirname(__file__)
    # Support both filenames and raw code payloads
    if "\n" in job_target or (not job_target.endswith(".py") and not os.path.exists(os.path.join(scriptdir, job_target))):
        unique_file = os.path.join(scriptdir, f"temp_job_{uuid.uuid4().hex[:8]}.py")
        with open(unique_file, "w") as f:
            f.write(job_target)
        target_path = unique_file
    else:
        target_path = os.path.join(scriptdir, job_target)

    if is_handshake_target(target_path):
        job_name = "Virtual Handshake"
    else:
        job_name = f"Job ({os.path.basename(target_path)})"

    # Display inbound transfer time for received file
    report_inbound_transfer(os.path.basename(target_path), client_ip)

    res_q = queue.Queue()
    new_depth = job_queue.qsize() + 1
    print(f"[QUEUE +1    ] Enqueued {job_name} from {client_ip} | Queue Size: {new_depth}", flush=True)
    job_queue.put((_run_job_file, (target_path,), res_q, client_ip, job_name))

    success, result = res_q.get()

    _thread_local.is_job_request = True
    _thread_local.job_name = job_name
    _thread_local.job_done_time = time.time()
    _thread_local.response_len = len(result.encode("utf-8")) if isinstance(result, str) else 0

    if not success:
        raise Exception(result)
    return result


def execute_ai_job(l_rate, script_filename="temp_job.py", data_filename="temp_data.txt"):
    client_ip = getattr(_thread_local, "client_ip", "Unknown")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir, script_filename)
    datapath = os.path.join(scriptdir, data_filename)

    job_name = f"AI Training Job (lr={l_rate})"

    # Display inbound transfer times for received files
    report_inbound_transfer(script_filename, client_ip)
    report_inbound_transfer(data_filename, client_ip)

    res_q = queue.Queue()
    new_depth = job_queue.qsize() + 1
    print(f"[QUEUE +1    ] Enqueued {job_name} from {client_ip} | Queue Size: {new_depth}", flush=True)
    job_queue.put((_run_ai_job_file, (scriptpath, datapath, l_rate), res_q, client_ip, job_name))

    success, result = res_q.get()

    _thread_local.is_job_request = True
    _thread_local.job_name = job_name
    _thread_local.job_done_time = time.time()
    _thread_local.response_len = len(result.encode("utf-8")) if isinstance(result, str) else 0

    if not success:
        raise Exception(result)
    return result



def get_queue_size():
    return job_queue.qsize()


def helper(i, ip, scriptpath, datapath=None):
    if not os.path.isabs(scriptpath):
        scriptdir = os.path.dirname(__file__)
        scriptpath = os.path.join(scriptdir, scriptpath)
    if datapath and not os.path.isabs(datapath):
        scriptdir = os.path.dirname(__file__)
        datapath = os.path.join(scriptdir, datapath)

    if i == 0:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            upload_file_in_chunks(worker, scriptpath, "temp_job.py")
            t_call = time.time()
            response = worker.execute_job()
            rt = time.time() - t_call
            print(f"[TRANSFER IN ] Received response from {ip} | Size: {format_bytes(len(response))} | Round-Trip: {rt:.2f}s", flush=True)
            if response == "Test\n":
                print(f"Worker {ip} is online and ready.", flush=True)
            else:
                print(f"Worker {ip} responded with unexpected output: {response}", flush=True)
        except Exception as e:
            print(f"Error with worker {ip}: {e}", flush=True)
    elif i == 1:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            print(f"Uploading script to {ip}...", flush=True)
            job_file = f"temp_job_{uuid.uuid4().hex[:8]}.py"
            upload_file_in_chunks(worker, scriptpath, job_file)
            t_call = time.time()
            response = worker.execute_job(job_file)
            rt = time.time() - t_call
            print(f"[TRANSFER IN ] Received response from {ip} | Size: {format_bytes(len(response))} | Round-Trip: {rt:.2f}s", flush=True)
            print(f"Worker output from {ip}:\n{response}", flush=True)
        except Exception as e:
            print(f"Error with worker {ip}: {e}", flush=True)
    elif i == 2:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            l_rate = input(f"Learning rate for AI model going to ip_address: {ip}, is: ").strip()
            print(f"Uploading script and data to {ip}...", flush=True)
            job_file = f"temp_job_{uuid.uuid4().hex[:8]}.py"
            data_file = f"temp_data_{uuid.uuid4().hex[:8]}.txt"
            upload_file_in_chunks(worker, scriptpath, job_file)
            upload_file_in_chunks(worker, datapath, data_file)
            t_call = time.time()
            response = worker.execute_ai_job(l_rate, job_file, data_file)
            rt = time.time() - t_call
            print(f"[TRANSFER IN ] Received response from {ip} | Size: {format_bytes(len(response))} | Round-Trip: {rt:.2f}s", flush=True)
            print(f"Worker output from {ip}:\n{response}", flush=True)
        except Exception as e:
            print(f"Error with worker {ip}: {e}", flush=True)



def ai_train(script_name=None, data_name=None):
    script_to_run = script_name if script_name else ai_script_path
    data_to_run = data_name if data_name else ai_data_path
    print(f"Starting distributed AI training job using script '{script_to_run}' and data '{data_to_run}'...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = script_to_run if os.path.isabs(script_to_run) else os.path.join(scriptdir, script_to_run)
    datapath = data_to_run if os.path.isabs(data_to_run) else os.path.join(scriptdir, data_to_run)
    if not os.path.exists(scriptpath):
        print(f"Error: Script file '{scriptpath}' not found.")
        return
    if not os.path.exists(datapath):
        print(f"Error: Data file '{datapath}' not found.")
        return
    try:
        print("Sending job...")
        threads = []
        for ip in list(wkr_ips):
            t = thread.Thread(target=helper, args=(2, ip, scriptpath, datapath))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        print(f"Error: {e}")


def single_job(name=None):
    script_to_run = name if name else single_script_path
    print(f"Starting distributed single job using script '{script_to_run}'...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = script_to_run if os.path.isabs(script_to_run) else os.path.join(scriptdir, script_to_run)
    if not os.path.exists(scriptpath):
        print(f"Error: Script file '{scriptpath}' not found.")
        return
    try:
        print("Sending job...")
        threads = []
        for ip in list(wkr_ips):
            t = thread.Thread(target=helper, args=(1, ip, scriptpath))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
    except Exception as e:
        print(f"Error: {e}")


def Menu():
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
            if ip == 'localhost':
                new_ip = ip
            else:
                new_ip = str(master_ip) + ip
            wkr_ips.add(new_ip)
            helper(0, new_ip, test)  # Call helper to run the test script on the new worker
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
            print("4. Switch AI to tiny AI test (ai_script.py / ai_data.txt)")
            print("5. Reset all to defaults (Script.py / Data.txt)")
            print("6. Back to Master Menu")
            cfg = input("Select option (1-6): ").strip()
            if cfg == '1':
                val = input("Enter path/filename for Single Job script (leave blank to cancel): ").strip()
                if val:
                    single_script_path = val
                    print(f"Single Job script set to: {single_script_path}")
            elif cfg == '2':
                val = input("Enter path/filename for AI Training script (leave blank to cancel): ").strip()
                if val:
                    ai_script_path = val
                    print(f"AI Training script set to: {ai_script_path}")
            elif cfg == '3':
                val = input("Enter path/filename for AI Training dataset (leave blank to cancel): ").strip()
                if val:
                    ai_data_path = val
                    print(f"AI Training dataset set to: {ai_data_path}")
            elif cfg == '4':
                ai_script_path = "ai_script.py"
                ai_data_path = "ai_data.txt"
                print("AI job configured to use: ai_script.py and ai_data.txt")
            elif cfg == '5':
                single_script_path = "Script.py"
                ai_script_path = "Script.py"
                ai_data_path = "Data.txt"
                print("All scripts reset to defaults: Script.py and Data.txt")
        elif menu_choice == 'S':
            print("Single task, type (S)ingle")
            print("AI train, type (A)I")
            c = input("Choice: ").strip().upper()
            if c == 'S':
                single_job()
            elif c == 'A':
                ai_train()
            else:
                print("Invalid choice")
        elif menu_choice == 'Q':
            print("Exiting Master mode...")
            break 
        else:
            print("Invalid choice, please try again.")


def worker():
    global job_queue
    stop_event = thread.Event()

    print("\n" + "=" * 62)
    print("                 WORKER NODE ONLINE")
    print("   Listening on Port 8000 | Managing Job Queue")
    print("   Press [ENTER] at any time to stop worker mode.")
    print("=" * 62 + "\n")

    def process_queue():
        while not stop_event.is_set():
            try:
                task = job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            func, args, res_q, client_ip, job_name = task
            q_remaining = job_queue.qsize()
            print(f"[EXECUTE     ] Starting {job_name} from {client_ip} | Queue Remaining: {q_remaining}", flush=True)

            start_time = time.time()
            try:
                result = func(*args)
                elapsed = time.time() - start_time
                print(f"[SUCCESS     ] Completed {job_name} from {client_ip} | Execution Time: {elapsed:.2f}s | Queue Size: {job_queue.qsize()}", flush=True)
                res_q.put((True, result))
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"[ERROR       ] Failed {job_name} from {client_ip} | Execution Time: {elapsed:.2f}s | Error: {e} | Queue Size: {job_queue.qsize()}", flush=True)
                res_q.put((False, str(e)))
            finally:
                job_queue.task_done()


    queue_worker_thread = thread.Thread(target=process_queue, daemon=True)
    queue_worker_thread.start()

    server = ThreadedXMLRPCServer(("0.0.0.0", 8000), requestHandler=CleanRequestHandler, allow_none=True, logRequests=False)
    server.register_function(upload_chunk, "upload_chunk")
    server.register_function(execute_job, "execute_job")
    server.register_function(execute_ai_job, "execute_ai_job")
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

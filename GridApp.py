import os as os
import subprocess as sp
import xmlrpc.client as Client
import threading as thread
import socket as socket
from xmlrpc.server import SimpleXMLRPCServer as Server
from socketserver import ThreadingMixIn
import queue as queue
import uuid as uuid


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


def upload_chunk(filename, chunk, mode):
    scriptdir = os.path.dirname(__file__)
    filepath = os.path.join(scriptdir, filename)
    data = chunk.data if hasattr(chunk, 'data') else chunk
    with open(filepath, mode) as f:
        f.write(data)
    return True

def upload_file_in_chunks(worker, local_path, remote_filename):
    with open(local_path, "rb") as f:
        mode = "wb"
        while True:
            chunk = f.read(1024 * 1024) # 1MB chunks
            if not chunk:
                break
            worker.upload_chunk(remote_filename, Client.Binary(chunk), mode)
            mode = "ab" # append for subsequent chunks

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


def execute_job(job_target="temp_job.py"):
    scriptdir = os.path.dirname(__file__)
    # Support both filenames and raw code payloads
    if "\n" in job_target or (not job_target.endswith(".py") and not os.path.exists(os.path.join(scriptdir, job_target))):
        unique_file = os.path.join(scriptdir, f"temp_job_{uuid.uuid4().hex[:8]}.py")
        with open(unique_file, "w") as f:
            f.write(job_target)
        target_path = unique_file
    else:
        target_path = os.path.join(scriptdir, job_target)

    res_q = queue.Queue()
    print(f"Queuing job for {os.path.basename(target_path)} (queue size: {job_queue.qsize() + 1})...")
    job_queue.put((_run_job_file, (target_path,), res_q))
    success, result = res_q.get()
    if not success:
        raise Exception(result)
    return result


def execute_ai_job(l_rate, script_filename="temp_job.py", data_filename="temp_data.txt"):
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir, script_filename)
    datapath = os.path.join(scriptdir, data_filename)

    res_q = queue.Queue()
    print(f"Queuing AI job with learning rate {l_rate} (queue size: {job_queue.qsize() + 1})...")
    job_queue.put((_run_ai_job_file, (scriptpath, datapath, l_rate), res_q))
    success, result = res_q.get()
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
            response = worker.execute_job()
            if response == "Test\n":
                print(f"Worker {ip} is online and ready.")
            else:
                print(f"Worker {ip} responded with unexpected output: {response}")
        except Exception as e:
            print(f"Error with worker {ip}: {e}")
    elif i == 1:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            print(f"Uploading script to {ip}...")
            job_file = f"temp_job_{uuid.uuid4().hex[:8]}.py"
            upload_file_in_chunks(worker, scriptpath, job_file)
            response = worker.execute_job(job_file)
            print(f"Worker output from {ip}:", response)
        except Exception as e:
            print(f"Error with worker {ip}: {e}")
    elif i == 2:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            l_rate = input(f"Learning rate for AI model going to ip_address: {ip}, is: ")
            print(f"Uploading script and data to {ip}...")
            job_file = f"temp_job_{uuid.uuid4().hex[:8]}.py"
            data_file = f"temp_data_{uuid.uuid4().hex[:8]}.txt"
            upload_file_in_chunks(worker, scriptpath, job_file)
            upload_file_in_chunks(worker, datapath, data_file)
            response = worker.execute_ai_job(l_rate, job_file, data_file)
            print(f"Worker output from {ip}:", response)
        except Exception as e:
            print(f"Error with worker {ip}: {e}")


def ai_train(script_name="ai_script.py", data_name="ai_data.txt"):
    print("Starting the distributed AI training job...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir, script_name)
    if not os.path.exists(scriptpath):
        scriptpath = os.path.join(scriptdir, "Script.py")
    datapath = os.path.join(scriptdir, data_name)
    if not os.path.exists(datapath):
        datapath = os.path.join(scriptdir, "Data.txt")
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



def single_job(name="Script.py"):
    print("Starting the distributed job...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir, name)
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
    while True:
        print("\n--- Master Node Management ---")
        print("(A)dd an IP")
        print("(R)emove an IP")
        print("(V)iew current IPs")
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

    def process_queue():
        print("Worker queue processor started. Ready to execute queued jobs.")
        while not stop_event.is_set():
            try:
                task = job_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            func, args, res_q = task
            try:
                print(f"Worker dequeued a job. Executing... (Remaining in queue: {job_queue.qsize()})")
                result = func(*args)
                res_q.put((True, result))
            except Exception as e:
                print(f"Error during job execution: {e}")
                res_q.put((False, str(e)))
            finally:
                job_queue.task_done()
        print("Worker queue processor stopped.")

    queue_worker_thread = thread.Thread(target=process_queue, daemon=True)
    queue_worker_thread.start()

    server = ThreadedXMLRPCServer(("0.0.0.0", 8000), allow_none=True)
    print("Worker is listening on port 8000...")
    server.register_function(upload_chunk, "upload_chunk")
    server.register_function(execute_job, "execute_job")
    server.register_function(execute_ai_job, "execute_ai_job")
    server.register_function(get_queue_size, "get_queue_size")

    server_thread = thread.Thread(target=server.serve_forever)
    print("Server Starting")
    server_thread.start()

    input("Worker is running... Press [ENTER] to stop and switch modes.\n")

    print("Shutting down worker...")
    server.shutdown()
    server_thread.join()
    try:
        server.server_close()
    except Exception:
        pass
    stop_event.set()
    queue_worker_thread.join(timeout=2)
    print("Worker stopped.")


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

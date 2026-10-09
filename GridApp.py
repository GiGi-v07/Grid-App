import os
import subprocess as sp
import xmlrpc.client as Client
import threading as thread
from xmlrpc.server import SimpleXMLRPCServer as Server


# Variables
wkr_ips = set()  # Set to store worker IPs 
active_threads = []
scriptdir = os.path.dirname(os.path.abspath(__file__))
test = os.path.join(scriptdir, "test.py")


def upload_chunk(filename, chunk, mode):
    filepath = os.path.join(scriptdir, filename)
    with open(filepath, mode) as f:
        f.write(chunk.data)
    return True

def upload_file_in_chunks(worker, local_path, remote_filename):
    with open(local_path, "rb") as f:
        mode = "wb"
        while True:
            chunk = f.read(1024 * 1024)  # 1MB chunks
            if not chunk:
                break
            worker.upload_chunk(remote_filename, Client.Binary(chunk), mode)
            mode = "ab"  # append for subsequent chunks

def execute_job():
    filepath = os.path.join(scriptdir, "temp_job.py")
    result = sp.run(["python", filepath], capture_output=True, text=True)
    if os.path.exists(filepath):
        os.remove(filepath)
    output = result.stdout
    if result.stderr:
        output = (output + "\n[STDERR]:\n" + result.stderr) if output else result.stderr
    return output


def execute_ai_job(l_rate):
    scriptpath = os.path.join(scriptdir, "temp_job.py")
    datapath = os.path.join(scriptdir, "temp_data.txt")
    result = sp.run(["python", scriptpath, datapath, str(l_rate)], capture_output=True, text=True)
    if os.path.exists(scriptpath):
        os.remove(scriptpath)
    if os.path.exists(datapath):
        os.remove(datapath)
    output = result.stdout
    if result.stderr:
        output = (output + "\n[STDERR]:\n" + result.stderr) if output else result.stderr
    return output

def helper(i, ip, scriptpath, datapath=None, l_rate="0.01"):
    if i == 0:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            upload_file_in_chunks(worker, scriptpath, "temp_job.py")
            response = worker.execute_job()
            if response and response.strip() == "Test":
                print(f"Worker {ip} is online and ready.")
                return True
            else:
                print(f"Worker {ip} responded with unexpected output: {response}")
                return False
        except Exception as e:
            print(f"Error connecting to worker {ip}: {e}")
            return False
    elif i == 1:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            print(f"Uploading script to {ip}...")
            upload_file_in_chunks(worker, scriptpath, "temp_job.py")
            response = worker.execute_job()
            print(f"Worker output from {ip}:\n{response}")
        except Exception as e:
            print(f"Error with worker {ip}: {e}")
    elif i == 2:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            print(f"Uploading AI model script and dataset to {ip}...")
            upload_file_in_chunks(worker, scriptpath, "temp_job.py")
            upload_file_in_chunks(worker, datapath, "temp_data.txt")
            print(f"Worker {ip} is training the AI model (lr={l_rate})...")
            response = worker.execute_ai_job(l_rate)
            print(f"\n--- Worker Output from {ip} ---\n{response}")
        except Exception as e:
            print(f"Error with worker {ip}: {e}")


def ai_train():
    if not wkr_ips:
        print("No workers added yet! Please add at least one worker IP using (A).")
        return
    print("\nStarting Distributed AI Training Job...")
    l_rate_input = input("Enter learning rate for AI model [default: 0.01]: ").strip()
    l_rate = l_rate_input if l_rate_input else "0.01"

    scriptpath = os.path.join(scriptdir, "Script.py")
    datapath = os.path.join(scriptdir, "Data.txt")

    if not os.path.exists(datapath):
        print(f"Dataset not found at {datapath}. Generating synthetic dataset...")
        try:
            import Generate
            Generate.generate_dataset(datapath)
        except Exception as e:
            print(f"Failed to generate dataset: {e}")
            return

    try:
        print(f"Dispatching training jobs to {len(wkr_ips)} worker(s) with learning rate {l_rate}...")
        threads = []
        for ip in list(wkr_ips):
            t = thread.Thread(target=helper, args=(2, ip, scriptpath, datapath, l_rate))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print("\nAll distributed AI training jobs completed.")
    except Exception as e:
        print(f"Error in distributed AI training: {e}")


def single_job(name="Script.py"):
    if not wkr_ips:
        print("No workers added yet! Please add at least one worker IP using (A).")
        return
    print("\nStarting distributed single job...")
    scriptpath = os.path.join(scriptdir, name)
    try:
        print(f"Dispatching job to {len(wkr_ips)} worker(s)...")
        threads = []
        for ip in list(wkr_ips):
            t = thread.Thread(target=helper, args=(1, ip, scriptpath))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        print("\nAll worker jobs completed.")
    except Exception as e:
        print(f"Error in single job: {e}")


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
            new_ip = input("Enter the Worker IP to add (e.g. localhost or 192.168.x.x): ").strip()
            if not new_ip:
                print("IP address cannot be empty.")
                continue
            print(f"Testing connection to worker at {new_ip}:8000...")
            online = helper(0, new_ip, test)
            if online:
                wkr_ips.add(new_ip)
                print(f"Added {new_ip} to active workers.")
            else:
                add_anyway = input(f"Worker {new_ip} did not pass handshake. Add anyway? (y/N): ").strip().lower()
                if add_anyway == 'y':
                    wkr_ips.add(new_ip)
                    print(f"Added {new_ip} (offline/unverified).")
        elif menu_choice == 'R':
            del_ip = input("Enter the Worker IP to remove: ").strip()
            if del_ip in wkr_ips:
                wkr_ips.remove(del_ip)
                print(f"Removed {del_ip}.")
            else:
                print("Error: IP not found in the list.")
        elif menu_choice == 'V':
            if wkr_ips:
                print(f"Current Worker IPs ({len(wkr_ips)}): {list(wkr_ips)}")
            else:
                print("Current Worker IPs: None registered yet.")
        elif menu_choice == 'S':
            print("\nSelect Job Type:")
            print("(S)ingle Task")
            print("(A)I Training")
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
        

if __name__ == "__main__":
    while True:
        print("\n===============================")
        print("          GRID APP             ")
        print("===============================")
        print("Run this laptop as:")
        print("(M)aster Node")
        print("(W)orker Node")
        print("(Q)uit Program")
        ch = input("Select an option: ").strip().upper()
        if ch == 'Q':
            print("Exiting GridApp.")
            break
        elif ch == 'W':
            server = Server(("0.0.0.0", 8000), allow_none=True)
            print("\nWorker is listening on port 8000...")
            server.register_function(upload_chunk, "upload_chunk")
            server.register_function(execute_job, "execute_job")
            server.register_function(execute_ai_job, "execute_ai_job")
            server_thread = thread.Thread(target=server.serve_forever)
            print("Server Started.")
            server_thread.start()

            input("Worker is running... Press [ENTER] to stop and switch modes.\n")

            print("Shutting down worker...")
            server.shutdown()
            server_thread.join()
            print("Worker stopped.")
        elif ch == 'M':
            Menu()
        else:
            print("Invalid choice.")
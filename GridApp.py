import os as os
import subprocess as sp
import xmlrpc.client as Client
import threading as thread
from xmlrpc.server import SimpleXMLRPCServer as Server


# Variables
wkr_ips = set()  # Set to store worker IPs 
active_threads = []
test = "test.py"



def upload_chunk(filename, chunk, mode):
    scriptdir = os.path.dirname(__file__)
    filepath = os.path.join(scriptdir, filename)
    with open(filepath, mode) as f:
        f.write(chunk.data)
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

def execute_job():
    scriptdir = os.path.dirname(__file__)
    filepath = os.path.join(scriptdir,"temp_job.py")
    result = sp.run(["python",filepath],capture_output=True,text=True)
    os.remove(filepath)
    return result.stdout


def execute_ai_job(l_rate):
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir,"temp_job.py")
    datapath = os.path.join(scriptdir,"temp_data.txt")
    result = sp.run(["python",scriptpath,datapath,l_rate],capture_output=True,text=True)
    os.remove(scriptpath)
    os.remove(datapath)
    return result.stdout

def helper(i, ip, scriptpath, datapath=None):
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
            upload_file_in_chunks(worker, scriptpath, "temp_job.py")
            response = worker.execute_job()
            print(f"Worker output from {ip}:", response)
        except Exception as e:
            print(f"Error with worker {ip}: {e}")
    elif i == 2:
        try:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            l_rate = input(f"Learning rate for AI model going to ip_address: {ip}, is: ")
            print(f"Uploading script and data to {ip}...")
            upload_file_in_chunks(worker, scriptpath, "temp_job.py")
            upload_file_in_chunks(worker, datapath, "temp_data.txt")
            response = worker.execute_ai_job(l_rate)
            print(f"Worker output from {ip}:", response)
        except Exception as e:
            print(f"Error with worker {ip}: {e}")


def ai_train():
    print("Starting the distributed job...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir,"Script.py")
    datapath = os.path.join(scriptdir,"Data.txt")
    try:
        print("Sending job...")
        for ip in wkr_ips:
            t = thread.Thread(target=helper, args=(2,ip, scriptpath, datapath))
            active_threads.append(t)
            t.start()
        for t in active_threads:
            t.join()
            active_threads.remove(t)
    except Exception as e:
        print(f"Error: {e}")


def single_job(name="Script.py"):
    print("Starting the distributed job...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir, name)
    try:
        print("Sending job...")
        for ip in wkr_ips:
            t = thread.Thread(target=helper, args=(1,ip, scriptpath))
            active_threads.append(t)
            t.start()
        for t in active_threads:
            t.join()
            active_threads.remove(t)
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
            new_ip = input("Enter the Worker IP to add: ").strip()
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
        

while True:
    print("Run this laptop as (M)aster, (W)orker or (Q)uit Program?")
    ch = input("Select an option: ").strip().upper()
    if ch == 'Q':
        break
    elif ch == 'W':
        server = Server(("0.0.0.0",8000), allow_none=True)
        print("Worker is listening on port 8000...")
        server.register_function(upload_chunk, "upload_chunk")
        server.register_function(execute_job, "execute_job")
        server.register_function(execute_ai_job, "execute_ai_job")
        server_thread = thread.Thread(target=server.serve_forever)
        print("Server Starting")
        server_thread.start()

        input("Worker is running... Press [ENTER] to stop and switch modes.\n")

        print("Shutting down worker...")
        server.shutdown()
        server_thread.join()
    elif ch == 'M':
        Menu()
    else:
        print("Invalid choice.")
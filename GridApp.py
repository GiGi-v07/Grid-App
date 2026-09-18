import os as os
import subprocess as sp
import xmlrpc.client as Client
import threading as thread
from xmlrpc.server import SimpleXMLRPCServer as Server

# Variables
wkr_ips = [] 




def execute_job(code):
    scriptdir = os.path.dirname(__file__)
    filepath = os.path.join(scriptdir,"temp_job.py")
    with open(filepath,"w") as file:
        file.write(code)
    result = sp.run(["python",filepath],capture_output=True,text=True)
    os.remove(filepath)
    return result.stdout


def execute_ai_job(code,data,l_rate):
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir,"temp_job.py")
    datapath = os.path.join(scriptdir,"temp_data.txt")
    with open(scriptpath,"w") as script:
        script.write(code)
    with open(datapath,"w") as data_obj:
        data_obj.write(data)
    result = sp.run(["python",scriptpath,datapath,l_rate],capture_output=True,text=True)
    os.remove(scriptpath)
    os.remove(datapath)
    return result.stdout


def ai_train():
    print("Starting the distributed job...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir,"Script.py")
    datapath = os.path.join(scriptdir,"Data.txt")
    try:
        with open(scriptpath,"r") as script:
            payload = script.read()
        with open(datapath,"r") as data:
            data_payload = data.read()        
        print("Sending job...")
        for ip in wkr_ips:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            l_rate = input(f"Learning rate for AI model going to ip_address: {ip}, is: ")
            response = worker.execute_ai_job(payload,data_payload,l_rate)
            print("Worker output:", response)
    except Exception as e:
        print(f"Error: {e}")


def single_job():
    print("Starting the distributed job...")
    scriptdir = os.path.dirname(__file__)
    scriptpath = os.path.join(scriptdir,"Script.py")
    try:
        with open(scriptpath,"r") as script:
            payload = script.read()
        print("Sending job...")
        for ip in wkr_ips:
            worker = Client.ServerProxy(f"http://{ip}:8000")
            response = worker.execute_job(payload)
            print("Worker output:", response)
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
            wkr_ips.append(new_ip)
            print(f"Added {new_ip} to the grid.")
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
        server.register_function(execute_job,"execute_job")
        server.register_function(execute_ai_job,"execute_ai_job")
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
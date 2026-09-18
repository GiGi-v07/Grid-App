import xmlrpc.client as Client
import os as os
worker = Client.ServerProxy("http://localhost:8000/")

scriptdir = os.path.dirname(__file__)
filepath = os.path.join(scriptdir,"Script.py")

with open(filepath,"r") as file:
    payload = file.read()
print("Sending job...")
response = worker.execute_job(payload)
print("Worker output:", response)
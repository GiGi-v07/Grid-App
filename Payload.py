import os as os
import subprocess as sp
from xmlrpc.server import SimpleXMLRPCServer as Server



def execute_job(code):
    scriptdir = os.path.dirname(__file__)
    filepath = os.path.join(scriptdir,"temp_job.py")
    with open(filepath,"w") as file:
        file.write(code)
    result = sp.run(["python",filepath],capture_output=True,text=True)
    os.remove(filepath)
    return result.stdout




server = Server(("0.0.0.0",8000))
print("Worker is listening on port 8000...")
server.register_function(execute_job,"execute_job")
server.serve_forever()
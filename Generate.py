import os as os 
scriptdir = os.path.dirname(__file__)
filepath = os.path.join(scriptdir,"Data.txt")

with open (filepath,"w") as file:
    for i in range(1000000):
        file.write(str(i)+"\n")
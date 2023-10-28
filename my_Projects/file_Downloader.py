import requests
from tqdm import tqdm
r=requests.get("https://www.win-rar.com/fileadmin/winrar-versions/winrar/winrar-x64-624.exe", stream=True) 
totalExpectedBytes=int(r.headers["content-length"])
progress_Bar=tqdm(total=totalExpectedBytes, unit="B", unit_scale=True, colour="green")
with open("winrar223.exe", 'wb') as fd:
    for chunk in r.iter_content(chunk_size=1024):
        progress_Bar.update(len(chunk))
        fd.write(chunk)
progress_Bar.close()

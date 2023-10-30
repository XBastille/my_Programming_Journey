import requests
import json
from pywebio.input import *
from pywebio.output import *
from pywebio import *
def get_Fun_Fact():
    clear()
    put_html("<p align=""left""><h2><img src=""data:image/png;base64,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"" width=""11%"">  \ Fun Fact Generator</h2></p>") 
    headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    r=requests.get("https://uselessfacts.jsph.pl/random.json?language=en", headers=headers)
    data=json.loads(r.text)
    uf=data["text"]
    put_text(uf).style("color:blue; font-size: 30px")
    put_button("click me boi!", onclick=get_Fun_Fact)
start_server(get_Fun_Fact, port=8080, debug=True, remote_access=True)
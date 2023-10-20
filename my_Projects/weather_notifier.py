import requests
from bs4 import BeautifulSoup
from win10toast import ToastNotifier
n=ToastNotifier()
headers={'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
r=requests.get("https://weather.com/en-IN/weather/today/l/68509ddcc58030eeb09cd5130641916746139e9174ab3527650e99b108d95ed9", headers=headers)
soup=BeautifulSoup(r.text, "html.parser")
current_info1=str(soup.find("div", class_="CurrentConditions--header--kbXKR").get_text())
current_info2=str(soup.find("span", class_="CurrentConditions--tempValue--MHmYY").get_text())
current_info3=str(soup.find("div", class_="CurrentConditions--phraseValue--mZC_p").get_text())
current_info4=str(soup.find("div", class_="CurrentConditions--tempHiLoValue--3T1DG").get_text())
res=(f"{current_info1}\n{current_info2}\n{current_info3}\n{current_info4}")
n.show_toast("WEATHER UPDATE", res, duration=10)

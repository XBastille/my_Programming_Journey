import requests
from bs4 import BeautifulSoup
#proxies are required for bulk scraping (will learn later)
'''html_doc = """<html><head><title>The Dormouse's story</nice></head>
<body>
<p class="title"><b>The Dormouse's story</b></p>

<p class="story">Once upon a time there were three little sisters; and their names were
<a href="https://example.com/elsie" class="sister" id="link1">Elsie</a>,
<a href="https://example.com/lacie" class="sister" id="link2">Lacie</a> and
<a href="https://example.com/tillie" class="sister" id="link3">Tillie</a>;
and they lived at the bottom of a well.</p>
<p class="story">...</p>
"""
soup=BeautifulSoup(html_doc, "html.parser")
#print(soup.prettify())
print(soup.title)
print(soup.get_text())  #get all the text available in the website
print(soup.find_all("a"))
for links in soup.find_all("a"):    #extracting all the links from the website
    print(links.get("href"))'''
headers=headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
r=requests.get("https://timesofindia.indiatimes.com/city/bhubaneswar", headers=headers)
#our requests was failed as amazon didn't permit to scrap, so, we had to simulate a fake browser by providing a user-Agent in order to bypass it
'''with open ("parsing2.html", "r", encoding="utf-8") as f:
    l=f.read()'''
soup=BeautifulSoup(r.text, "lxml")
print(soup.a)     #here it refers to tag i.e </a>
'''print(soup.find_all("div"))'''
print(soup.title.string)
print(soup.span)
for links in soup.find_all("a"):    #extracting all the links from the website
    print(links.get("href"))
m=soup.find(class_="uZEMf").get_text().split(",")
for i,j in enumerate(m):
    print(f"{i}: {j}")
print(soup.find(class_="lSIdy col_l_6 col_m_6").find("a").get("href"))

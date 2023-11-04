from pywebio import *
from pywebio.input import *
from pywebio.output import *
'''age=input("Enter you age:-", type=NUMBER)
password=input("Enter your password:", type=PASSWORD)
gift=select("which gift do you want?", ["A Car", "A Dog"])
area=checkbox("what is your branch?", options=["CSIT", "CSE", "ECE", "EEE", "IT"])
area2=radio("what is your branch?", options=["CSIT", "CSE", "ECE", "EEE", "IT"])
text=textarea("Text Area", rows=3, placeholder="Some text")
img=file_upload("select image:", accept="image/*")              #we can put placeholder as well
#n=input("What's your name?", type=TEXT, placeholder="Enter your name here", help_text="Full name is required", required=True)
def check_age(p):
    if p<18:
        return "Too young"
    elif p>60:
        return "Too old"
#age=input("Enter your age:-", type=NUMBER, validate=check_age)
def check_form(g):
    if len(g["name"])>8:
        return ("name", "Name too big")
    elif g["age"]<=0:
        return ("age", "age can't be negetive")
data=input_group("Basic info",[
    input("Enter your name:-", name="name", required=True), 
    input("Enter your age:-", name="age", type=NUMBER)
], validate=check_form)
code=textarea("code edit", code={"mode":"python", "theme": "darcula"}, rows=10, value="import something\n# write your python code")
#output
put_text(data["name"], ":", data["age"])
put_text("hello world")
put_table([
    ["Name", "Age"],
    ["Bibhor", "18"],
    ["gora", "65"]
])
#put_image(open('/path/to/some/image.png', 'rb').read())  # local image 
#put_image('http://example.com/some-image.png')  # internet image 
put_markdown("~~strikethrough~~")        #what it does is it puts a straight line over characters (the ~ does the strike)
put_file("hello_world.txt", b"hello world!")     #here the first arguement is the file download link and we can name it as we want, 2nd arguement is the contents of the file
popup("popup title", "popup text content")
toast("new message")
put_table([
    ['Type', 'Content'],
    ['html', put_html('X<sup>2</sup>')],      #imp
    ['text', '<hr/>'],  # equal to ['text', put_text('<hr/>')]
    ['buttons', put_buttons(['A', 'B'], onclick=...)],   #imp
    ['markdown', put_markdown('`Awesome PyWebIO!`')],    #imp
    ['file', put_file('hello.text', b'hello world')],
    ['table', put_table([
        ['A', 'B'], 
        ['C', 'D']])]
])
popup('Popup title', [
    put_html('<h3>Popup Content</h3>'),    #imp
    'plain html: <br/>',  # Equivalent to: put_text('plain html: <br/>')
    put_table([['A', 'B'], ['C', 'D']]),
    put_button('close_popup()', onclick=close_popup)     #imp
])
with put_collapse("This is the title"):     #we use put_collapse to hidden and unhidden some contents by clicking the symbol on  it
    for i in range(4):
        put_text(i)
    put_table([
        ['Commodity', 'Price'],
        ['Apple', '5.5'],
        ['Banana', '7']
    ])
#we can use "with" for popups, toast
with use_scope("scope1"):
    put_text("this is the text in scope1")
put_text("this is text in parent scope of scope1")
with use_scope("scope1"):
    put_text("this is text2 in scope1")
with use_scope('scope2', clear=True):  # enter the existing scope and clear the previous content
    put_text('text in scope2')
from datetime import datetime
@use_scope('time', clear=True)
def show_time():
    put_text(datetime.now())
with use_scope('A'):
    put_text('Text in scope A')
    with use_scope('B'):        #nested scope
        put_text('Text in scope B')
with use_scope('C'):
    put_text('Text in scope C')
put_table([
    ['Name', 'Hobbies'],
    ['Tom', put_scope('hobby', content=put_text('Coding'))]  # hobby is initialized to coding
])
with use_scope('hobby', clear=True):   #clear=True is important to clear the first content
    put_text('Movie')  # hobby is reset to Movie
    put_text('Music')
    put_text('Drama')
# append Music, Drama to hobby
with use_scope('hobby'):
    put_text('Music')
    put_text('Drama')               #both withs give the same result
# insert the Coding into the top of the hobby
put_markdown('**Coding**', scope='hobby', position=0)    #this is the alternate way to use scope instead of using context manager.
#in conclusion, put_scope can be used inside any put_XXX()
#it is not allowed to have two scopes with the same name in the application
#clear(scope) : Clear the contents of the scope
#remove(scope) : Remove scope
#scroll_to(scope) : Scroll the page to the scope
put_row([
    put_column([
        put_code("A"),
        put_row([
            put_code("B1"), None,  #None represents the space between the output
            put_code("B2"), None,
            put_code("B3"),
        ]),
        put_code("C"),
    ]), None, 
    put_code("D"), None,
    put_code("E")
], size="40%60%")
#we didn't write None in B3 or E because it is the last element and writing None won't have any effect
#when we write put_row it means we get exactly one row and we can have as many column in that row. Same applies for put_column
put_text("hello").style("color: red; font-size: 20px")
#in combined output
put_row([
    put_text("hello").style("color: red"),
    put_markdown("markdown")
]).style("margin-top: 20px")'''
def main():   #pywebio application function
    name=input("what's your name?")
    put_text("hello", name)
start_server(main, port=8080, debug=True, remote_access=True)       #debug=True will the server to automatically reload if code changes.
#remote_access=True will give us a public, shareable address for the current application
#If you never call start_server() or path_deploy() in your code, then you are running PyWebIO application as script mode.




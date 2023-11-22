import asyncio
import telegram
from telegram import Update
import logging
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
'''async def main():
    bot=telegram.Bot("6869832786:AAGRf1gvzfcVh0XUl9CDSVGwOgJj4VZQmm8")           #token
    async with bot:
        print(await bot.getMe())                #getMe() returns simple information about the bot in the form of a user object
if __name__=="__main__":
    asyncio.run(main())
async def main():
    bot=telegram.Bot("6869832786:AAGRf1gvzfcVh0XUl9CDSVGwOgJj4VZQmm8")           
    async with bot:
        print((await bot.get_updates())[-1].message.from_user.id) 
if __name__=="__main__":
    asyncio.run(main
async def main():
    bot=telegram.Bot("6869832786:AAGRf1gvzfcVh0XUl9CDSVGwOgJj4VZQmm8")           
    async with bot:
        await bot.send_message(text="HI BIBHOR", chat_id=6559351208)
if __name__=="__main__":
    asyncio.run(main())'''
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
#now adding functionailty to it
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="hello boi, how you doing?")
async def echo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text=update.message.text)
async def caps(update: Update,context: ContextTypes.DEFAULT_TYPE):
    text_caps=" ".join(context.args).upper()                         #we used join because we receive args as list
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
#the above function will be called everytime the bot receives a telegram message i.e /start command
#the function will receive two parameter 1. update which is an object that contains all the information and data that are coming from telegram itself (like the message, the user etc)
#2. context, another object that contains the info and data of the status of the library(like the bot, application, the job queue)
if __name__=="__main__":
    application=ApplicationBuilder().token("6869832786:AAGRf1gvzfcVh0XUl9CDSVGwOgJj4VZQmm8").build()             #now application will manage all the updates fetched by updaters
    #now application is created using ApplicationiBuider() class and Updater is also created that will link with asycio.queue
    start_handler=CommandHandler("start", start)              #this will let handler to listen to our command, CommandHandler (one of the provided Handler subclasses)
    echo_handler=MessageHandler(filters.TEXT & (~filters.COMMAND), echo) 
    caps_handler=CommandHandler("caps", caps)
    unknown_handler=MessageHandler(filters.COMMAND, unknown)
    #filters module contains a no. of so called filters that filter incoming message like text, images, videos, status update etc
    #any message that returns True for least one of the filters passed to messageHandler will be accepted
    application.add_handler(start_handler)  
    application.add_handler(echo_handler)
    application.add_handler(caps_handler)
    application.add_handler(unknown_handler, group=0)
    application.run_polling()         #this will run the bot until we hit ctrl+c

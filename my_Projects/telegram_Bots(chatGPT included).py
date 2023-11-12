import asyncio
import openai
import logging
import telegram
from telegram import Update
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes, MessageHandler, filters
logging.basicConfig(level=logging.INFO, format="%(name)s - %(levelname)s - %(message)s")
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user=update.message.from_user
    await context.bot.send_message(chat_id=update.effective_chat.id, text=f"""hello {user["first_name"]}, how you doing?
    type or click /show_commands to get show all the commands available""")
async def show_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="""Available Commands :- 
    /get_link - gives the link of the website (note:- only website with .com)
    /chatGPT - use chatGPT and it's features 
    /caps - turns your text into capital""")
async def caps(update: Update,context: ContextTypes.DEFAULT_TYPE):
    text_caps=" ".join(context.args).upper()                     
    await context.bot.send_message(chat_id=update.effective_chat.id, text=text_caps)
async def get_link(update: Update,context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)!=0:  
        link="".join(context.args).lower()            
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"https://www.{link}.com/")
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text='sorry I did not get any website name. Type /get_link "website name"')
async def chatGPT(update: Update,context: ContextTypes.DEFAULT_TYPE):
    if len(context.args)!=0:
        prompt=" ".join(context.args) 
        openai.api_key="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" 
        response=openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role":"user", "content":prompt}]
        )
        await context.bot.send_message(chat_id=update.effective_chat.id, text=response.choices[0].message.content.strip())
    else:
        await context.bot.send_message(chat_id=update.effective_chat.id, text='sorry I did not get any prompt, please type /chatGPT "prompt"')
async def unknown(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(chat_id=update.effective_chat.id, text="Sorry, I didn't understand that command.")
if __name__=="__main__":
    application=ApplicationBuilder().token("6869832786:AAGRf1gvzfcVh0XUl9CDSVGwOgJj4VZQmm8").build()
    start_handler=CommandHandler("start", start)
    show_commands_handler=CommandHandler("show_commands", show_commands)
    get_link_handler=CommandHandler("get_link", get_link)
    caps_handler=CommandHandler("caps", caps)
    unknown_handler=MessageHandler(filters.COMMAND, unknown)
    chatGPT_handler=CommandHandler("chatGPT", chatGPT)
    application.add_handler(start_handler)
    application.add_handler(show_commands_handler)
    application.add_handler(caps_handler)
    application.add_handler(get_link_handler)
    application.add_handler(chatGPT_handler)
    application.add_handler(unknown_handler, group=0)
    application.run_polling()
'''from typing import Dict
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    ConversationHandler,
    MessageHandler,
    PicklePersistence,
    filters,
)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
CHOOSING, TYPING_REPLY, TYPING_CHOICE = range(3)
reply_keyboard = [
    ["Age", "Favourite colour"],
    ["Number of siblings", "Something else..."],
    ["Done"],
]
markup = ReplyKeyboardMarkup(reply_keyboard, one_time_keyboard=True)
def facts_to_str(user_data: Dict[str, str]) -> str:
    facts=[f"{key} - {value}" for key, value in user_data.items()]
    return "\n".join(facts).join(["\n", "\n"])
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_text = "Hi! My name is x_Bastille."
    if context.user_data:
        reply_text+=(
            f" You already told me your {', '.join(context.user_data.keys())}. Why don't you "
            "tell me something more about yourself? Or change anything I already know."
        )
    else:
        reply_text+=(
            " I will hold a more complex conversation with you. Why don't you tell me "
            "something about yourself?"
        )
    await update.message.reply_text(reply_text, reply_markup=markup)
    return CHOOSING
async def regular_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text.lower()
    context.user_data["choice"] = text
    if context.user_data.get(text):
        reply_text = (
            f"Your {text}? I already know the following about that: {context.user_data[text]}"
        )
    else:
        reply_text = f"Your {text}? Yes, I would love to hear about that!"
    await update.message.reply_text(reply_text)
    return TYPING_REPLY
async def custom_choice(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    await update.message.reply_text(
        'Alright, please send me the category first, for example "Most impressive skill"'
    )
    return TYPING_CHOICE
async def received_information(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    text = update.message.text
    category = context.user_data["choice"]
    context.user_data[category] = text.lower()
    del context.user_data["choice"]
    await update.message.reply_text(
        "Neat! Just so you know, this is what you already told me:"
        f"{facts_to_str(context.user_data)}"
        "You can tell me more, or change your opinion on something.",
        reply_markup=markup,
    )
    return CHOOSING
async def show_data(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display the gathered info."""
    await update.message.reply_text(
        f"This is what you already told me: {facts_to_str(context.user_data)}"
    )
async def done(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if "choice" in context.user_data:
        del context.user_data["choice"]
    await update.message.reply_text(
        f"I learned these facts about you: {facts_to_str(context.user_data)}Until next time!",
        reply_markup=ReplyKeyboardRemove(),
    )
    return ConversationHandler.END
def main() -> None:
    persistence=PicklePersistence(filepath="conversationbot")
    application=Application.builder().token("6869832786:AAGRf1gvzfcVh0XUl9CDSVGwOgJj4VZQmm8").persistence(persistence).build()
    conv_handler=ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            CHOOSING: [
                MessageHandler(
                    filters.Regex("^(Age|Favourite colour|Number of siblings)$"), regular_choice
                ),
                MessageHandler(filters.Regex("^Something else...$"), custom_choice),
            ],
            TYPING_CHOICE: [
                MessageHandler(
                    filters.TEXT & ~(filters.COMMAND | filters.Regex("^Done$")), regular_choice
                )
            ],
            TYPING_REPLY: [
                MessageHandler(
                    filters.TEXT & ~(filters.COMMAND | filters.Regex("^Done$")),
                    received_information,
                )
            ],
        },
        fallbacks=[MessageHandler(filters.Regex("^Done$"), done)],
        name="my_conversation",
        persistent=True,
    )
    application.add_handler(conv_handler)
    show_data_handler = CommandHandler("show_data", show_data)
    application.add_handler(show_data_handler)
    application.run_polling(allowed_updates=Update.ALL_TYPES)
if __name__ == "__main__":
    main()'''

    

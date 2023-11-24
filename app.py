import os
import logging
import asyncio
import re
from aiogram import Bot, Dispatcher, types, executor
import openai
from openai import AsyncOpenAI
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage

logging.basicConfig(level=logging.INFO)
load_dotenv()
TOKEN = os.getenv("TG_TOKEN")
BOT_ID = int(os.getenv("BOT_ID"))

# Инициализация вашей системы
logging.info("Инициализация началась")
storage_cntxt = StorageContext.from_defaults(persist_dir="./data/index_storage_1024")
idx = load_index_from_storage(storage_cntxt)
query_engine = idx.as_query_engine(
    include_text=True,
    response_mode="no_text",
    embedding_mode="hybrid",
    similarity_top_k=3,
)
logging.info("Завершена")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)
client = AsyncOpenAI()

# Создаем асинхронную очередь
message_queue = asyncio.Queue()

async def message_worker():
    """
    Message queue handler
    """
    while True:
        # Получаем задачу из очереди
        chat_id, text = await message_queue.get()

        # Отправляем сообщение
        await bot.send_message(chat_id, text, parse_mode="HTML")

        # Ждем 6 секунд перед обработкой следующей задачи
        await asyncio.sleep(6)

async def on_startup(dispatcher):
    """
    Run the task in the context of the current event cycle
    """
    asyncio.create_task(message_worker())

async def keep_typing(chat_id, interval=5):
    """
    Function to keep the "print" effect.
    """
    while True:
        await bot.send_chat_action(chat_id, 'typing')
        await asyncio.sleep(interval)

async def answer(user_message: str, reply_to_message=None):
    """
    Common function for getting answer from GPT
    """
    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    regex = re.compile(r"Вопрос: (.*?) \n\n", re.DOTALL)
    if not reply_to_message is None:
        message = f"{reply_to_message}\n\n<b>Вопрос:</b><i>{escape_html(user_message)}<i> \n\n" \
                  f"<b>Ответ:</b>"
        retrival_query = regex.findall(reply_to_message)[0] + escape_html(user_message)
        retrival = await query_engine.aquery(retrival_query)
    else:
        message = user_message
        retrival = await query_engine.aquery(message)
    information = [
        (i.text, i.metadata["url"], i.metadata["title"]) for i in retrival.source_nodes
    ]

    for text in [text for text, _, _ in information]:
        logging.info(text)

    information_text = escape_html(" ".join([text for text, _, _ in information]))
    information_url = "\n".join(
        set(
            f'&#x25CF; <a href="{url}">{escape_html(title)}</a>'
            for _, url, title in information
        )
    )

    context_prompt = (
        "Ниже мы предоставили контекстную информацию\n"
        "---------------------\n"
        f"{information_text}"
        "\n---------------------\n"
        f"Учитывая эту информацию, ответьте, пожалуйста, на вопрос: {message}\n"
        "\n---------------------\n"
        "Ответ на вопрос должен быть развернутым, полным, и охватывать множество "
        "аспектов заданного вопроса"
        "Внимание! В ответе нельзя упоминать конекстную информацию! "
        "Пользователь не знает о ее наличии!"
    )

    logging.info("Сообщение сформировано и отправлено в OpenAI")
    model_name = "gpt-3.5-turbo-1106"

    context_response = await client.chat.completions.create(
        model=model_name, temperature=0, messages=[{"role": "user", "content": context_prompt}]
    )
    logging.info(context_response.choices[0].message.content)
    # logging.info(context_response.choices[0]['message']['content'])

    evaluate_promt = (
        "Есть ответы на вопросы некоторых пользователей. Даны только ответы."
        "Задание: необходимо определить, получилось ли у пользователя ответить на вопрос, "
        "является ли его ответ уверенным. Если получилось ответить, ответь YES, если нет -"
        "NO"
        "\n---------------------\n"
        "MESSAGE #1: Да, результаты действий, выполненных в Jupyter Notebook, "
        "можно подключить к даталенсу.\n"
        "YOUR ANSWER TO MESSAGE #1: YES\n"
        "MESSAGE #2: Из предоставленной контекстной информации нельзя однозначно определить, "
        "рассматривается ли тема с временными рядами в курсах корпорации Karpov inc..\n"
        "YOUR ANSWER TO MESSAGE #2: NO\n"
        "MESSAGE #3: Извините, я не могу ответить на этот вопрос без дополнительной информации. \n"
        "YOUR ANSWER TO MESSAGE #3: NO\n"
        f"MESSAGE #4: {context_response.choices[0].message.content}\n"
        f"YOUR ANSWER TO MESSAGE #4:"
    )
    evaluate_response = await client.chat.completions.create(
        model=model_name, temperature=0, messages=[{"role": "user", "content": evaluate_promt}]
    )
    logging.info(evaluate_response.choices[0].message.content)

    dont_match_start_phrase = "К сожалению, я не могу ответить на этот вопрос, " \
                              "основываясь на роликах с YouTube-канала Karpov.Courses, " \
                              "но могу сам ответить на него.\n"

    if evaluate_response.choices[0].message.content.strip() == "YES":
        main_response = context_response.choices[0].message.content
        extended_answer = f"<b>Подробнее здесь:</b> \n\n{information_url}"
    else:
        gpt_response = await client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[{"role": "user", "content": message}]
        )
        main_response = dont_match_start_phrase + gpt_response.choices[0].message.content
        extended_answer = ""

    template_answer = (
            f"<b>Вопрос:</b> <i>{escape_html(user_message)}</i> \n\n"
            f"<b>Ответ:</b> {main_response} \n\n" +
            extended_answer
    )
    return template_answer

@dp.message_handler(commands=["start", "help"])
async def send_welcome(message: types.Message):
    """
    Sends a welcome message
    """
    greeting = "Привет, я бот KarpovAI!\n" \
               "Я являюсь QA-системой на основе контента YouTube-канала Karpov.Courses.\n" \
               "Для того, чтобы я ответил на твой вопрос, тегни меня: @karpovAI_bot\n" \
               "Бот создан студентами <a href='https://karpov.courses/simulator-ml'>SimulatorML</a>"
    await message_queue.put((message.chat.id, greeting))

@dp.message_handler(lambda message: "@karpovAI_bot" in message.text)
async def handle_tag(message: types.Message):
    """
    KarpovAI bot tag function
    """
    logging.info("Сообщение принято")
    typing_task = asyncio.create_task(keep_typing(message.chat.id))
    try:
        user_message = message.text.replace("@karpovAI_bot", "").strip()
        try:
            template_answer = await answer(user_message)
        except openai.APITimeoutError:
            template_answer = "Сервис пока не доступен. Попробуйте обратиться позже"
    finally:
        typing_task.cancel()
    await message_queue.put((message.chat.id, template_answer))


@dp.message_handler(lambda message: message.reply_to_message and
                                    message.reply_to_message.from_user.id == BOT_ID)
async def handle_reply(message: types.Message):
    """
    KarpovAI bot reply function
    """
    logging.info("Сообщение принято")
    typing_task = asyncio.create_task(keep_typing(message.chat.id))
    try:
        original_message = message.reply_to_message.text
        user_reply = message.text
        try:
            template_answer = await answer(user_reply, reply_to_message=original_message)
        except openai.APITimeoutError:
            template_answer = "Сервис пока не доступен. Попробуйте обратиться позже"
    finally:
        typing_task.cancel()
    await message_queue.put((message.chat.id, template_answer))

if __name__ == "__main__":
    executor.start_polling(dp, skip_updates=True, on_startup=on_startup)

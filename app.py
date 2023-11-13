import os
import logging
from aiogram import Bot, Dispatcher, types, executor
import openai
from dotenv import load_dotenv
from llama_index import StorageContext, load_index_from_storage

logging.basicConfig(level=logging.INFO)
load_dotenv()
TOKEN = os.getenv("TG_TOKEN")
openai.api_key = os.getenv("API_KEY")

# Инициализация вашей системы
logging.info("Инициализация началась")
# print("Инициализация началась")
storage_cntxt = StorageContext.from_defaults(persist_dir="./data/index_storage_1024")
idx = load_index_from_storage(storage_cntxt)
query_engine = idx.as_query_engine(
    include_text=True,
    response_mode="no_text",
    embedding_mode="hybrid",
    similarity_top_k=3,
)
logging.info("Завершена")
# print("Завершена")
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(lambda message: "@karpovAI_bot" in message.text)
async def answer(message: types.Message):
    """
    KarpovAI bot
    """
    logging.info("Сообщение принято")
    # print("Сообщение принято")
    user_message = message.text.replace("@karpovAI_bot", "").strip()
    retrival = query_engine.query(user_message)
    information = [
        (i.text, i.metadata["url"], i.metadata["title"]) for i in retrival.source_nodes
    ]

    def escape_html(text: str) -> str:
        return (
            text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    for text in [text for text, _, _ in information]:
        logging.info(text)

    information_text = escape_html(" ".join([text for text, _, _ in information]))
    information_url = "\n".join(
        set(
            [
                f'&#x25CF; <a href="{url}">{escape_html(title)}</a>'
                for _, url, title in information
            ]
        )
    )

    context_prompt = (
        "Ниже мы предоставили контекстную информацию\n"
        "---------------------\n"
        f"{information_text}"
        "\n---------------------\n"
        f"Учитывая эту информацию, ответьте, пожалуйста, на вопрос: {user_message}\n"
        "Ответ на вопрос должен быть развернутым, полным, и охватывать множество "
        "аспектов заданного вопроса"
        "Внимание! В ответе нельзя упоминать конекстную информацию! "
        "Пользователь не знает о ее наличии!"
    )

    logging.info("Сообщение сформировано и отправлено в OpenAI")
    model_name = "gpt-3.5-turbo-1106" # "gpt-3.5-turbo"
    context_response = openai.ChatCompletion.create(
        model=model_name, temperature=0, messages=[{"role": "user", "content": context_prompt}]
    )
    logging.info(context_response.choices[0]['message']['content'])

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
        f"MESSAGE #3: {context_response.choices[0]['message']['content']}\n"
        f"YOUR ANSWER TO MESSAGE #3:"
    )
    evaluate_response = openai.ChatCompletion.create(
        model=model_name, temperature=0, messages=[{"role": "user", "content": evaluate_promt}]
    )
    logging.info(evaluate_response.choices[0]['message']['content'])

    dont_match_start_phrase = "К сожалению, я не могу ответить на этот вопрос, " \
                              "основываясь на роликах с YouTube-канала Karpov.Courses, " \
                              "но могу сам ответить на него.\n"

    if evaluate_response.choices[0]['message']['content'].strip() == "YES":
        main_response = context_response.choices[0]['message']['content']
        extended_answer = f"<b>Подробнее здесь:</b> \n\n{information_url}"
    else:
        main_response = dont_match_start_phrase + openai.ChatCompletion.create(
            model=model_name,
            temperature=0,
            messages=[{"role": "user", "content": user_message}]
        ).choices[0]['message']['content']
        extended_answer = ""

    template_answer = (
        f"<b>Вопрос:</b> <i>{escape_html(user_message)}</i> \n\n"
        f"<b>Ответ:</b> {main_response} \n\n" +
        extended_answer
    )
    await message.answer(template_answer, parse_mode="HTML")


if __name__ == "__main__":

    executor.start_polling(dp, skip_updates=True)

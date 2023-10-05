import logging
from aiogram import Bot, Dispatcher, types
import openai
from dotenv import load_dotenv
import os
from llama_index import StorageContext, load_index_from_storage

load_dotenv()
TOKEN = os.getenv("TG_TOKEN")
openai.api_key = os.getenv("API_KEY")

# Инициализация вашей системы
logging.info("Инициализация началась")
storage_cntxt = StorageContext.from_defaults(persist_dir="./data/index_storage_1024")
idx = load_index_from_storage(storage_cntxt)
query_engine = idx.as_query_engine(
    include_text=True,
    response_mode="no_text",
    embedding_mode="hybrid",
    similarity_top_k=5,
)
logging.info("Завершена")
logging.basicConfig(level=logging.INFO)
bot = Bot(token=TOKEN)
dp = Dispatcher(bot)


@dp.message_handler(lambda message: "@karpovAI_bot" in message.text)
async def answer(message: types.Message):
    logging.info("Сообщение принято")
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

    information_text = escape_html(" ".join([text for text, _, _ in information]))
    information_url = "\n".join(
        set(
            [
                f'&#x25CF; <a href="{url}">{escape_html(title)}</a>'
                for _, url, title in information
            ]
        )
    )

    template_prompt = (
        "Ниже мы предоставили контекстную информацию\n"
        "---------------------\n"
        f"{information_text}"
        "\n---------------------\n"
        f"Учитывая эту информацию, ответьте коротко, пожалуйста, на вопрос: {user_message}\n"
    )
    logging.info("Сообщение сформировано и отправлено в OpenAI")
    model_name = "gpt-3.5-turbo"
    response = openai.ChatCompletion.create(
        model=model_name, messages=[{"role": "user", "content": template_prompt}]
    )
    template_answer = (
        f"<b>Вопрос:</b> <i>{escape_html(user_message)}</i> \n\n"
        f"<b>Ответ:</b> <i>{response.choices[0]['message']['content']}</i> \n\n"
        "<b>Подробнее здесь:</b> \n\n"
        f"{information_url}"
    )
    await message.answer(template_answer, parse_mode="HTML")


if __name__ == "__main__":
    from aiogram import executor

    executor.start_polling(dp, skip_updates=True)

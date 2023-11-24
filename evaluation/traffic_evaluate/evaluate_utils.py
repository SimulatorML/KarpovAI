"""
Скрипт с функциями для оценки трафика тг-бота.

Порядок оценки трафика:
1. Генерируем вопросы к каждому видео.
2. Получаем поисковый индекс для каждого вопроса.
3. Парсим основной тг-чат Karpov.Courses за последний месяц, получаем json с сообщениями.
Парсер отдельным файлом.
4. С помощью gpt-3.5-turbo отбираем сообщения, которые содержат вопрос.
5. Для каждого вопроса из тг-чата получаем один retrival из индекса с вопросами
(Этот retrival будет ближайшим по similarity к вопросу из чата).
6. Замеряем similarity между вопросом из чата и ближайшим сгенерированным вопросом.
Собираем датасет с колонками 'chat_question', 'gen_question', 'similarity'.
7. Исследуем значения similarity, выбираем отсечку.

Функции:
- process_nodes - for each video, discards the last node if it is too short
    to prevent non-relevant questions appearing.
- generate_video_questions - функция для генерации вопросов к каждому видео.
- get_questions_index - функция для получения индекса.
- filter_questions_from_chat - функция отбора вопросов из тг-чата с помощью gpt-3.5-turbo.
- get_questions_dataset - функция получения датасета для исследования релевантных боту вопросов.
"""
from typing import List
# from typing import DataFrame # - not works at my machine...
import pandas as pd # my substitute for typing DataFrame
from llama_index.schema import BaseNode

# parse json file with transcribed videos
# and save a modified file
import json

import pandas as pd
from dotenv import load_dotenv
import openai

from llama_index.schema import BaseNode

# to pass the model object into the functions
from llama_index.llms import OpenAI

# initialize the Document object to be used by SimpleNodeParser
from llama_index import Document

# to split the video text into the nodes
from llama_index.node_parser import SimpleNodeParser

# make node: question pairs
from llama_index.evaluation import generate_question_context_pairs
from llama_index.evaluation import RelevancyEvaluator, FaithfulnessEvaluator
from llama_index import StorageContext, load_index_from_storage, VectorStoreIndex, ServiceContext
from pytube import YouTube

load_dotenv()
openai.api_key = os.getenv("API_KEY")


def process_nodes(nodes: List[BaseNode]) -> List[BaseNode]:
    """
    Processes a list of nodes based on the following criteria:
    
    1. If a node has the same title as the previous one, it's added to the result list 
       only if its word count is more than 50% of the previous node's word count.
    2. If a node has a different title, it's simply added to the result list.

    Parameters:
    - nodes (list): A list of BaseNode objects.

    Returns:
    - list: A processed list of BaseNode objects.
    """
    
    # Initialize an empty list to store the processed nodes
    result = []

    # Initialize variables to keep track of the previous node's title and word count
    prev_title = None
    prev_word_count = 0

    # Iterate through the list of nodes
    for node in nodes:
        current_title = node.metadata['title']
        current_word_count = len(node.text.split())

        # Check if the current node has the same title as the previous one
        if current_title == prev_title:
            if current_word_count > 0.5 * prev_word_count:
                result.append(node)
        else:
            result.append(node)

        # Update the previous title and word count for the next iteration
        prev_title = current_title
        prev_word_count = current_word_count

    return result

def generate_video_questions(
        video_info_path: str,
        llm: OpenAI,
        chunk_size: int = 3 * 1024,
        video_info_output_path: str = None,
        test_version: bool = True
        ) -> None:
    """
    Generates questions for the text of each video from video_info.json, first splitting the text
    into chunk_size tokens. Saves the generated questions back into video_info.json.
    The resulting format of video_info.json is:

    [
        {
            "url": [<url of video>],
            "title": [<title of video>],
            "description": [<description of video>],
            "audio_path": [<path to audio track of video>],
            "text": [<text of video>],
            "control_questions": [<generated questions>]
        },
    ]

    Parameters
    ----------
    video_info_path: str
        path to video_info.json
    llm: OpenAI object
        a model for question generation
    chunk_size: int
        size of the nodes to produce in tokens
    video_info_output_path: str
        path to the final json file with questions
    test_version: bool
        if False, parses the whole json file

    Returns
    -------

    """

    # create an output json path
    if not video_info_output_path:
        video_info_output_path = video_info_path.replace('.json', '_with_questions.json')

    # open json with transcribed videos and parse
    with open(video_info_path, "r", encoding="utf-8") as f:
        video_json = json.load(f)

    # when testing, do not run the function on thr whole json dataset
    if test_version:
        video_json = video_json[:3]

    assert len(video_json) == 3

    # a list of Documents to be used by SimpleNodeParser
    # to make a list of Nodes
    docs = [
        Document(
            text="".join(video_dict["text"]),
            metadata={"url": video_dict["url"][0], "title": video_dict["title"][0]},
        )
        for video_dict in video_json
    ]

    # parse the docs into the nodes
    parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
    nodes = parser.get_nodes_from_documents(docs)

    # process the nodes to discard the last node of the video 
    # if it is very shortened during splitting
    nodes = process_nodes(nodes)

    # make a prompt template
    qa_generate_prompt_tmpl = """\
    Внизу указана контекстная информация.

    ---------------------
    {context_str}
    ---------------------

    На основе приведенного текста составь только один вопрос, \
    на который можно ответить с помощью текста. \
    Вопрос должен покрыть как можно больше аспектов в тексте. \
    Он должен быть только на основе привденного текста \
    и относиться к области анализа данных."
    """

    # query based on the nodes list
    # num_questions_per_chunk will anyway not appear in the query, 
    # can be any value :)
    qa_dataset = generate_question_context_pairs(
        nodes=nodes, 
        llm=llm, 
        qa_generate_prompt_tmpl=qa_generate_prompt_tmpl,
        num_questions_per_chunk=1 
        )

    # pairs node_id: video_url
    node_id_url = {node.id_: node.metadata['url'] for node in nodes}

    # pairs question_id: node_id
    question_id_node_id = {
        question_id: qa_dataset.relevant_docs[question_id][0] 
        for question_id in qa_dataset.relevant_docs
        }
    
    # pairs question_id: video_url
    question_id_url = {
        question_id: node_id_url[question_id_node_id[question_id]] 
        for question_id in question_id_node_id
        }
    
    # invert the 'question_id_url'
    # pairs video_url: list[question_id]
    url_question_id = {}

    for key, value in question_id_url.items():
        if value in url_question_id:
            url_question_id[value].append(key)
        else:
            url_question_id[value] = [key]
    
    # replace question_id by question_text
    # pairs video_url: list[question]

    url_question = {
        url: [
            qa_dataset.queries[question_id] 
              for question_id in url_question_id[url]
              ] 
              for url in url_question_id
              }
    
    # extend the original json
    # by creating 'control_questions': list[question] pairs
    # for each video's dict
    for video_dict in video_json:
        video_dict['control_questions'] = url_question[video_dict['url'][0]]
    
    # dump the final json to a new file
    with open(video_info_output_path, "w", encoding="utf-8") as f:
        json.dump(video_json, f, ensure_ascii=False, indent=4)

        
def get_questions_index(index_folder_path: str, video_info_path: str) -> None:
    """
    Заносит каждый вопрос к видео из video_info.json в поисковый индекс с помощью llama_index.
    В качестве метаданных также заносится ссылка и название видео.

    Parameters
    ----------
    index_folder_path
    video_info_path

    Returns
    -------

    """

def filter_questions_from_chat(parsed_chat_path: str, filtered_questions_path: str) -> None:
    """
    Отбирает с помощью gpt-3.5-turbo сообщения из чата, в которых содержится вопрос
    Перед подачей в gpt-3.5-turbo обрабатывает тексты сообщений, в которых содержатся
    метаданные: ссылки, форматирование и т.д.

    Parameters
    ----------
    parsed_chat_path: str
        path to json with parsed messages
    filtered_questions_path
        path to filtered messages containing the question

    Returns
    -------

    """
    with open(parsed_chat_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)

    # Convert metadata from text (links, bold, etc) to text
    for message in data_json:
        if isinstance(message["text"], list):
            text_list = []
            for text in message["text"]:
                if isinstance(text, dict):
                    text_list.append(text["text"])
                else:
                    text_list.append(text)
            message["text"] = " ".join(text_list)

    promt = """Контекст:\n
        Есть сообщения из чата онлайн-школы по анализу данных и машинному обучению\n
        Задание:\n
        Необходимо определить, является ли полученное сообщение вопросом по анализу данных или \
        машинному обучению, отвечая на который, можно также приложить различные учебные \
        материалы, ссылки, видео и т.д.\n
        Необходимо учесть несколько нюансов:\n
        - Не каждое сообщение, удовлетворяющее условию, является вопросом в явном виде.\n
        - Часто искомые сообщения начинаются с обращения к студентам и просьбой решить какую-нибудь \
        проблему.\n
        - Также искомые сообщения обычно развернутые, и в них детально описывается суть проблемы.\n
        - Но искомыми сообщениями не являются наводящие вопросы или ответы других студентов.\n
        Если сообщение является искомым вопросом, ответить YES, если не является - \
        ответить NO\n
        ---------------------\n
        MESSAGE #1: всем привет! кто пользуется datalens для визуализации, \
        поделитесь плз своим мнением\n
        YOUR ANSWER TO MESSAGE #1: YES\n
        MESSAGE #2: пилите кастомный дашборд на питоне с помощью библиотеки plotly dash, \
        разворачивайте его локально или на сервере\n
        YOUR ANSWER TO MESSAGE #2: NO\n
        MESSAGE #3: Всем привет. Подскажите, не смог найти решение в инете. \
        Не смог в юпитере установить библиотеку catboost. \
        Выскакивает ошибка: ERROR: Could not build wheels for catboost, \
        which is required to install pyproject.toml-based projects. \
        Работаю на mac m1. Что делать?)\n
        YOUR ANSWER TO MESSAGE #3: YES\n
        MESSAGE #4: А, и это ошибка инсталляции собственно?\n
        YOUR ANSWER TO MESSAGE #4: NO\n"""

    if os.path.exists(filtered_questions_path):
        with open(filtered_questions_path, "r", encoding="utf-8") as f:
            filtered_data = json.load(f)
            last_id = filtered_data["id"]
    else:
        filtered_data = {"id": 0, "content": []}
        last_id = 0
    for number_message, message in enumerate(data_json, 1):
        if last_id >= number_message:
            continue
        filtered_data.update({"id": number_message})
        print(f"Processed {number_message} message")
        template = f"MESSAGE #5: {message['text']}\nYOUR ANSWER TO MESSAGE #5: "
        response_object = openai.ChatCompletion.create(
            model=llm_model, messages=[{"role": "user", "content": (promt + template)}]
        )
        response = response_object.choices[0]['message']['content']
        print(template, response)
        if response.strip() == "YES":
            filtered_data["content"].append(message)
        with open(filtered_questions_path, "w", encoding="utf-8") as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=4)

def get_questions_dataset(filtered_questions_path: str,
                          index_folder_path: str,
                          question_dataset_path: str):
    """
    Составляет датафрейм для исследования и отбора релевантных боту вопросов из чата.
    Этапы:
    - Для каждого вопроса из тг-чата получаем один retrival из индекса с вопросами
    (Этот retrival будет ближайшим по similarity к вопросу из чата).
    - Получаем similarity между вопросом из чата и ближайшим сгенерированным вопросом
    с помощью retrival.source_nodes[0].get_score().
    - Собираем датасет с колонками 'chat_question', 'gen_question', 'similarity'.

    Parameters
    ----------
    filtered_questions_path
    index_folder_path

    Returns
    -------
    DataFrame
        columns: 'chat_question', 'gen_question', 'similarity'

    """
    with open(filtered_questions_path, "r", encoding="utf-8") as f:
        data_json = json.load(f)["content"]

    storage_context = StorageContext.from_defaults(persist_dir=index_folder_path)
    index = load_index_from_storage(storage_context)
    retriever = index.as_retriever(similarity_top_k=1)
    print("Index is loaded")

    gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")
    service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)
    rel_evaluator_gpt3 = RelevancyEvaluator(service_context=service_context_gpt3)
    faith_evaluator_gpt3 = FaithfulnessEvaluator(service_context=service_context_gpt3)

    if os.path.exists(question_dataset_path):
        with open(question_dataset_path, "r", encoding="utf-8") as f:
            question_dataset = json.load(f)
            last_id = question_dataset["id"]
    else:
        question_dataset = {"id": 0,
                            "content":{
                                "chat_question": [],
                                "retrieved_node": [],
                                "video_source": [],
                                "response": [],
                                "similarity": [],
                                "faithfulness": [],
                                "relevancy": []}
                            }
        last_id = 0
    for number_message, message in enumerate(data_json, 1):
        if last_id >= number_message:
            continue
        question_dataset.update({"id": number_message})
        print(f"Processed {number_message} question")
        chat_question = message["text"]
        if not chat_question:
            continue
        retrival = retriever.retrieve(chat_question)[0]
        retrieved_node = retrival.node

        information_text = retrieved_node.text
        information_url = f"Karpov.courses: {retrieved_node.metadata['url']} " \
                          f"- {retrieved_node.metadata['title']})"

        template = (
            "Ниже мы предоставили контекстную информацию\n"
            "---------------------\n"
            f"{information_text}"
            "\n---------------------\n"
            f"Учитывая эту информацию, ответьте, пожалуйста, на вопрос: {chat_question}\n"
        )
        model_name = "gpt-3.5-turbo"
        response = openai.ChatCompletion.create(
            model=model_name, messages=[{"role": "user", "content": template}]
        ).choices[0]['message']['content']

        relevancy_score = rel_evaluator_gpt3.evaluate(chat_question,
                                                      response,
                                                      [information_text]).passing
        faithfulness_score = faith_evaluator_gpt3.evaluate(chat_question,
                                                           response,
                                                           [information_text]).passing
        question_dataset["content"]["chat_question"].append(chat_question)
        question_dataset["content"]["retrieved_node"].append(information_text)
        question_dataset["content"]["video_source"].append(information_url)
        question_dataset["content"]["response"].append(response)
        question_dataset["content"]["similarity"].append(retrival.get_score())
        question_dataset["content"]["faithfulness"].append(faithfulness_score)
        question_dataset["content"]["relevancy"].append(relevancy_score)
        with open(question_dataset_path, "w", encoding="utf-8") as f:
            json.dump(question_dataset, f, ensure_ascii=False, indent=4)

def get_chat_statistics(parsed_chat_path: str, filtered_questions_path: str) -> dict:
    """
    Gets chat statistics in the form of a dictionary with the following keys:
    number of messages, number of unique users
    """
    with open(parsed_chat_path, "r", encoding="utf-8") as f:
        parsed_data = json.load(f)
    msg_number = len(parsed_data)
    unique_users = set()
    for message in parsed_data:
        unique_users.add(message["from_id"])
    unique_users_number = len(unique_users)

    with open(parsed_chat_path, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)
    qst_number = len(filtered_data)
    unique_qst_users = set()
    for message in filtered_data:
        unique_qst_users.add(message["from_id"])
    unique_qst_users_number = len(unique_users)

    return {"msg_number": msg_number,
            "unique_users_number": unique_users_number,
            "questions_number": qst_number,
            "unique_questions_user_number": unique_qst_users_number}

def get_channel_info(url_file_path, csv_file_path):
    """
    Get video info - title, length, publish_date
    """
    videos = []
    with open(url_file_path, "r", encoding="utf-8") as f:
        for line in f:
            videos.append(line.strip())
    video_data = {
        "url": [],
        "title": [],
        "length": [],
        "publish_date": []
    }
    for idx, video_url in enumerate(videos):
        yt = YouTube(video_url)
        video_data["url"].append(video_url)
        video_data["title"].append(yt.title)
        print(f"{idx} video {yt.title}")
        video_data["length"].append(yt.length)
        video_data["publish_date"].append(yt.publish_date)
    df_video_data = pd.DataFrame(video_data)
    df_video_data.to_csv(csv_file_path, index=False)

if __name__ == "__main__":
    # get_questions_index("questions_index_storage", "video_info_test.json")
    # filter_questions_from_chat("chat_KK.json", "chat_KK_filtered.json", "gpt-3.5-turbo")
    # print(get_chat_statistics("chat_KK.json"))
    # get_questions_dataset("chat_KK_filtered.json",
    #                       "../data/index_storage_1024",
    #                       "question_dataset.json")
    get_channel_info("../data/urls_of_channel_videos.txt", "date_lenght_video_info.csv")

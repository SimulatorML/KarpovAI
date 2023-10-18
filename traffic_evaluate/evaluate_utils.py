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

# to pass the model object into the functions
from llama_index.llms import OpenAI

# initialize the Document object to be used by SimpleNodeParser
from llama_index import Document

# to split the video text into the nodes
from llama_index.node_parser import SimpleNodeParser

# make node: question pairs
from llama_index.evaluation import generate_question_context_pairs


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

def get_questions_dataset(filtered_questions_path: str, index_folder_path: str) -> pd.DataFrame:

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

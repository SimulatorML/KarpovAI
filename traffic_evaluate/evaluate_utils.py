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
- question_gen - функция для генерации вопросов к каждому видео.
- get_questions_index - функция для получения индекса.
- filter_questions_from_chat - функция отбора вопросов из тг-чата с помощью gpt-3.5-turbo.
- get_questions_dataset - функция получения датасета для исследования релевантных боту вопросов.
"""
from typing import DataFrame


def question_gen(video_info_path: str) -> None:
    """
    Генерирует вопросы к тексту каждого видео из video_info.json, разбивая текст предварительно
    на 1024*3 токенов. Сохраняет сгенерированные вопросы в video_info.json.
    Шаблон получившегося video_info.json:
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

    Returns
    -------

    """

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

def get_questions_dataset(filtered_questions_path: str, index_folder_path: str) -> DataFrame:
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

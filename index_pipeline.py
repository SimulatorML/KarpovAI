import os
import json
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
import openai
from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index import StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex, ServiceContext
from parsers.parser_transcribe import ParserTranscribe, get_video_urls

load_dotenv()
openai.api_key = os.getenv("API_KEY")

@dataclass()
class IndexPipeline:
    """
    Запускает пайплайн получения индекса -
    от определения новых видео в YouTube до сохранения индекса.
    """
    path_to_save: str
    url_file_path: str
    json_video_info_path: str
    index_folder: str
    chunk_size: int = 200
    chunk_overlap: int = 50

    def _get_download_urls(self, channel_url) -> List[str]:
        """Определяет список видео для скачивания"""

        # Determine the set of downloaded videos
        downloaded_videos = set()
        if os.path.exists(self.url_file_path):
            with open(self.url_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    downloaded_videos.add(line.strip())

        # Determine the set of videos to download
        new_videos = set(get_video_urls(channel_url)) - downloaded_videos

        # Add new videos to the file
        with open(self.url_file_path, "a", encoding="utf-8") as f:
            for video in new_videos:
                f.write(video + "\n")

        return new_videos

    def _transcribe_videos(self, new_videos):
        """
        Скачивает и транскрибирует видео из списка self.new_videos.
        Текст и метаданные сохраняются в json-файл.
        """
        transcriber = ParserTranscribe(self.path_to_save, self.json_video_info_path)
        for url in new_videos:
            transcriber.get_transcribe_video(url)

    def _get_index(self, new_videos):
        """
        Функция должна работать следующим образом:

        1. Если есть сохраненный индекс в self.storage_index_path -
        загружает его.
        2. По списку new_videos находит документы в json и добавляет их в индекс.
        Или создает новый индекс, если self.storage_index_path не существует
        3. Сохраняет индекс
        """

        # Загружаем индекс
        vector_store_path = os.path.join(self.index_folder, "vector_store.json")
        if os.path.exists(vector_store_path):
            storage_context = StorageContext.from_defaults(persist_dir=self.index_folder)
            index = load_index_from_storage(storage_context)
        else:
            # Или создаем пустой
            node_parser = SimpleNodeParser.from_defaults(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            service_context = ServiceContext.from_defaults(node_parser=node_parser)
            index = VectorStoreIndex([], service_context=service_context)

        # Выбираем документы для добавления в индекс
        with open(self.json_video_info_path, "r", encoding="utf-8") as f:
            data_json = json.load(f)
        to_download_data = []
        for i in new_videos:
            for j in data_json:
                if i == j["url"]:
                    to_download_data.append(j)

        # Формируем документы
        documents = [
            Document(
                text=data["text"][0],
                metadata={"url": data["url"][0], "title": data["title"][0]},
            )
            for data in to_download_data
        ]

        # Добавляем документы в индекс
        for doc in documents:
            index.insert(doc)

        # Сохраняем индекс
        index.storage_context.persist(self.index_folder)


    def run(self, channel_url):
        """Запускает пайплайн получения индекса"""
        new_videos = self._get_download_urls(channel_url)
        self._transcribe_videos(new_videos)
        self._get_index(new_videos)

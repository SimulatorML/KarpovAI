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
from data_pipelines.parser_transcribe import ParserTranscribe, get_video_urls

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
        new_videos = list(set(get_video_urls(channel_url)) - downloaded_videos)

        # Add new videos to the file
        with open(self.url_file_path, "a", encoding="utf-8") as f:
            for video in new_videos:
                f.write(video + "\n")

        return new_videos

    def _transcribe_videos(self, new_videos: List[str]) -> None:
        """
        Скачивает и транскрибирует видео из списка self.new_videos.
        Текст и метаданные сохраняются в json-файл.
        """
        transcriber = ParserTranscribe(self.path_to_save, self.json_video_info_path)
        for i, url in enumerate(new_videos):
            print(f"Transcribe {i} video")
            transcriber.get_transcribe_video(url)

    def _get_index(self, new_videos: List[str]) -> None:
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
                if i == j["url"][0]:
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
        for i, doc in enumerate(documents):
            print(f"Add {i} video to index")
            index.insert(doc)

        # Сохраняем индекс
        index.storage_context.persist(self.index_folder)


    def run(self, channel_url: str, test: bool=False) -> None:
        """Запускает пайплайн получения индекса"""
        new_videos = self._get_download_urls(channel_url)
        if new_videos:
            if test:
                new_videos = new_videos[:2]
            self._transcribe_videos(new_videos)
            self._get_index(new_videos)
        else:
            print("No new videos to download")

if __name__ == "__main__":
    pipe = IndexPipeline(
        "../data/audio",
        "../data/urls_of_channel_videos.txt",
        "../data/video_info.json",
        "../data/index_storage_2048",
        chunk_size=2048
    )
    # pipe.run("https://www.youtube.com/c/karpovcourses")
    # storage_cntxt = StorageContext.from_defaults(persist_dir="../data/index_storage_1500")
    # idx = load_index_from_storage(storage_cntxt)
    # print("Index is loaded")
    # query_engine = idx.as_query_engine(
    #     include_text=True,
    #     response_mode="no_text",
    #     embedding_mode="hybrid",
    #     similarity_top_k=5,
    # )
    #
    # while True:
    #     question = input()
    #     if question == "exit":
    #         break
    #     retrival = query_engine.query(
    #         question,
    #     )
    #     print(f"Q: {question}")
    #     information = [
    #         (i.text, i.metadata["url"], i.metadata["title"]) for i in retrival.source_nodes
    #     ]
    #     for i in information:
    #         print(i[2], "\n", i[0])


    # with open("../data/video_info.json", "r", encoding="utf-8") as file:
    #     data_list = json.load(file)
    # crawling_urls = []
    # for itm in data_list:
    #     if itm["text"]:
    #         crawling_urls.append(itm["url"][0])
    # print(f"Number of crawling urls: {len(crawling_urls)}")
    #
    # all_videos = set()
    # with open("../data/urls_of_channel_videos.txt", "r", encoding="utf-8") as file:
    #     for line in file:
    #         all_videos.add(line.strip())
    # print(f"Total number urls: {len(all_videos)}")
    #
    # not_crawling = list(all_videos - set(crawling_urls))
    # print(f"Number of not crawling urls {len(not_crawling)}")
    # pipe._transcribe_videos(not_crawling)
    #
    #
    with open("../data/video_info.json", "r", encoding="utf-8") as file:
        data_list = json.load(file)
    transcribe_urls = []
    for itm in data_list:
        if itm["text"]:
            transcribe_urls.append(itm["url"][0])
    pipe._get_index(transcribe_urls)
    # ./data/audio/9W1v-DkXriY
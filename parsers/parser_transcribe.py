import os
import json
import subprocess
import time
from dataclasses import dataclass
from typing import List
from dotenv import load_dotenv
from pytube import YouTube
import openai
from parsers.channel_parser import ChannelParser

load_dotenv()
openai.api_key = os.getenv("API_KEY")

@dataclass()
class ParserTranscribe:
    """
    A pipeline for downloading and transcribing audio from YouTube videos.
    Has two public methods:
    - get_transcribe_video(url_of_video: str) - downloads and transcribes audio
    and saves data to a json file
    - get_video_urls(channel_url: str) - get list of all video urls from youtube-channel
    """
    path_to_save: str # Путь к папке с аудио
    json_video_info_path: str # Путь к json-файлу
    segment_time: int = 900 # Длительность сегмента при нарезке аудио
    max_attempts: int = 5  # максимальное количество попыток
    delay: int = 10  # задержка между попытками в секундах


    def _get_video_info(self, video_url: str) -> dict:
        """
        Get info from Youtube-video - title, description, audio track
        Parameters
        ----------
        url: str
          url of Youtube-video

        Returns
        -------
        dict
          keys:
          - "url": str
          - "title": str
          - "description": str
          - "audio_path": str
            saved audio path
        """
        yt = YouTube(video_url)
        audio_name = yt.video_id + ".mp4"
        audio_path = os.path.join(self.path_to_save, audio_name)
        title = yt.title
        print(f"Downloading {title}...")
        # The next string 'yt.streams...' must be before the following.
        # Surprisingly, but the fact is, if this is not observed,
        # then it is impossible to get a description.
        # Download mp4 audio file:
        yt.streams.filter(only_audio=True).first().download(self.path_to_save, audio_name)
        description = yt.description
        return {
            "url": video_url,
            "title": title,
            "description": description,
            "audio_path": audio_path
        }

    def _download_channel_audio_track(self, url_of_video: str) -> str:
        """
        Download audio track from YouTube-video and save info to json.
        Return path to downloaded audio track
        """
        if os.path.exists(self.json_video_info_path):
            with open(self.json_video_info_path, "r", encoding="utf-8") as f:
                video_info = json.load(f)
        else:
            video_info = []
        url_info = self._get_video_info(url_of_video)
        print(f"Path to mp4 file: {url_info['audio_path']}\n")

        video_info_item = {"url": [], "title": [], "description": [], "audio_path": [], "text": []}
        for key, value in url_info.items():
            video_info_item[key].append(value)

        # Проверяем, что такого url в json нет
        exist_url = True
        if video_info:
            for itm in video_info:
                exist_url = exist_url and (itm["url"][0] != video_info_item["url"][0])
        print(exist_url)
        if exist_url:
            video_info.append(video_info_item)

        with open(self.json_video_info_path, "w", encoding="utf-8") as f:
            json.dump(video_info, f, ensure_ascii=False, indent=4)

        return video_info_item["audio_path"][0]

    def _split_audio(self, file_path, output_pattern):
        """
        Разделяет аудиофайл на сегменты с использованием утилиты ffmpeg.

        Parameters
        ----------
        file_path : str
            Путь к исходному аудиофайлу, который необходимо разделить.
        output_pattern : str
            Шаблон имени выходного файла. Должен содержать `%03d` для нумерации сегментов
            (например, "output_segment%03d.mp3").

        Returns
        -------
        None

        """
        cmd = [
            "ffmpeg",
            "-i",
            file_path,
            "-f",
            "segment",
            "-segment_time",
            str(self.segment_time),
            "-c:a",
            "libmp3lame",
            output_pattern,
        ]
        subprocess.run(cmd, check=False)

    def _transcribe_with_whisper(self, audio_path):
        """
        Транскрибирует аудиофайл с использованием модели Whisper от OpenAI.

        Функция пытается транскрибировать аудио многократно
        (до max_attempts раз) в случае ошибок API.

        Parameters
        ----------
        audio_path : str
            Путь к аудиофайлу, который необходимо транскрибировать.

        Returns
        -------
        str
            Транскрибированный текст аудиофайла.

        Raises
        ------
        openai.error.APIError
            Если после всех попыток транскрибировать аудио возникает ошибка.

        """

        for attempt in range(self.max_attempts):
            try:
                with open(audio_path, "rb") as audio_file:
                    transcript = openai.Audio.transcribe(
                        file=audio_file,
                        model="whisper-1",
                        response_format="text",
                        language=["ru"],
                        prompt="Это организация Karpov Courses. Машинное обучение,"
                               " Machine learning, Дата саентист, data scientist,"
                               " dev ops, девопс, stepik.org, степик",
                    )
                return transcript
            except openai.error.APIError as e:
                print(f"Ошибка при обращении к API (попытка {attempt + 1}): {e}")
                if attempt < self.max_attempts - 1:  # если это не последняя попытка
                    print(f"Ожидание {self.delay} секунд перед следующей попыткой...")
                    time.sleep(self.delay)
                else:
                    print("Превышено максимальное количество попыток.")
                    raise  # повторно вызываем исключение, чтобы сообщить о проблеме
        return None

    def _get_transcribe(self, audio_path: str):
        file_name = os.path.basename(audio_path)
        if file_name.endswith(".mp4"):
            file_path = os.path.join(self.path_to_save, file_name)
            output_pattern = os.path.join(
                self.path_to_save, f"{file_name[:-4]}_segment%03d.mp3"
            )
            self._split_audio(file_path, output_pattern)

            # Собираем все транскрибации для данного файла
            transcriptions = []
            segment_files = sorted(
                [
                    f
                    for f in os.listdir(self.path_to_save)
                    if f.startswith(file_name[:-4] + "_segment")
                ]
            )
            for segment_file in segment_files:
                if segment_file.startswith(file_name[:-4] + "_segment"):
                    segment_path = os.path.join(self.path_to_save, segment_file)
                    transcriptions.append(self._transcribe_with_whisper(segment_path))
                    os.remove(
                        segment_path
                    )  # Удаляем спличенный файл после транскрибации
            os.remove(audio_path) # Удаляем исходный файл

            # Сохраняем все транскрибации в один текстовый файл
            print(transcriptions)
            with open(self.json_video_info_path, "r", encoding="utf-8") as f:
                data_list = json.load(f)

            for item in data_list:
                if item["audio_path"][0] == file_path:
                    item["text"] = transcriptions
                    break

            with open(self.json_video_info_path, "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)

    def get_transcribe_video(self, url_of_video: str):
        """Основная функция для полного запуска пайплайна транскрибации видео по ссылке"""
        audio_path = self._download_channel_audio_track(url_of_video)
        self._get_transcribe(audio_path)

    def get_video_urls(self, channel_url: str) -> List[str]:
        """Get list of all video urls from youtube-channel"""
        channel = ChannelParser(channel_url)
        return channel.video_urls

if __name__ == "__main__":
    parser = ParserTranscribe(
        "../audio",
        "../video_info.json"
    )
    parser.get_transcribe_video("https://www.youtube.com/watch?v=OXtOhjeiTzw")
    parser.get_transcribe_video("https://www.youtube.com/watch?v=9W1v-DkXriY")

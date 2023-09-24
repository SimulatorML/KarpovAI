import os
import subprocess
from dotenv import load_dotenv
import openai
import json
import time

load_dotenv()
openai.api_key = os.getenv("API_KEY")
AUDIO_DIR = "audio"
SEGMENT_TIME = 900


def split_audio(file_path, output_pattern):
    """
    Разделяет аудиофайл на сегменты с использованием утилиты ffmpeg.

    Parameters
    ----------
    file_path : str
        Путь к исходному аудиофайлу, который необходимо разделить.
    output_pattern : str
        Шаблон имени выходного файла. Должен содержать `%03d` для номерации сегментов (например, "output_segment%03d.mp3").

    Returns
    -------
    None

    Notes
    -----
    Функция использует глобальную переменную SEGMENT_TIME для определения длительности каждого сегмента.
    """
    cmd = [
        "ffmpeg",
        "-i",
        file_path,
        "-f",
        "segment",
        "-segment_time",
        str(SEGMENT_TIME),
        "-c:a",
        "libmp3lame",
        output_pattern,
    ]
    subprocess.run(cmd)


def transcribe_with_whisper(audio_path):
    """
    Транскрибирует аудиофайл с использованием модели Whisper от OpenAI.

    Функция пытается транскрибировать аудио многократно (до max_attempts раз) в случае ошибок API.

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

    Notes
    -----
    Функция использует глобальные переменные max_attempts и delay для определения количества попыток и задержки между ними.
    """
    max_attempts = 5  # максимальное количество попыток
    delay = 10  # задержка между попытками в секундах

    for attempt in range(max_attempts):
        try:
            with open(audio_path, "rb") as audio_file:
                transcript = openai.Audio.transcribe(
                    file=audio_file,
                    model="whisper-1",
                    response_format="text",
                    language=["ru"],
                    prompt="Это организация Karpov Courses. Машинное обучение, Machine learning, Дата саентист, data scientist, dev ops, девопс, stepik.org, степик",
                )
            return transcript
        except openai.error.APIError as e:
            print(f"Ошибка при обращении к API (попытка {attempt + 1}): {e}")
            if attempt < max_attempts - 1:  # если это не последняя попытка
                print(f"Ожидание {delay} секунд перед следующей попыткой...")
                time.sleep(delay)
            else:
                print("Превышено максимальное количество попыток.")
                raise  # повторно вызываем исключение, чтобы сообщить о проблеме


def main():
    """
    Основная функция для обработки аудиофайлов в указанной директории.

    Функция выполняет следующие действия:
    1. Проверяет наличие указанной директории с аудиофайлами.
    2. Разделяет каждый аудиофайл на сегменты.
    3. Транскрибирует каждый сегмент с помощью модели Whisper от OpenAI.
    4. Сохраняет транскрибированный текст в текстовый файл.
    5. Обновляет информацию в файле video_info.json.

    Notes
    -----
    Функция использует глобальные переменные, такие как AUDIO_DIR, для определения пути к директории с аудиофайлами.
    """
    if not os.path.exists(AUDIO_DIR):
        print(f"Папка {AUDIO_DIR} не найдена.")
        return

    for file_name in os.listdir(AUDIO_DIR):
        if file_name.endswith(".mp3"):
            file_path = os.path.join(AUDIO_DIR, file_name)
            output_pattern = os.path.join(
                AUDIO_DIR, f"{file_name[:-4]}_segment%03d.mp3"
            )
            split_audio(file_path, output_pattern)

            # Собираем все транскрибации для данного файла
            transcriptions = []
            segment_files = sorted(
                [
                    f
                    for f in os.listdir(AUDIO_DIR)
                    if f.startswith(file_name[:-4] + "_segment")
                ]
            )
            for segment_file in segment_files:
                if segment_file.startswith(file_name[:-4] + "_segment"):
                    segment_path = os.path.join(AUDIO_DIR, segment_file)
                    transcriptions.append(transcribe_with_whisper(segment_path))
                    os.remove(
                        segment_path
                    )  # Удаляем спличенный файл после транскрибации

            # Сохраняем все транскрибации в один текстовый файл
            print(transcriptions)
            with open("video_info.json", "r", encoding="utf-8") as f:
                data_list = json.load(f)

            for item in data_list:
                if item["audio_path"][0] == file_path:
                    item["text"] = transcriptions
                    break

            with open(
                os.path.join(AUDIO_DIR, f"{file_name[:-4]}.txt"), "w", encoding="utf-8"
            ) as txt_file:
                txt_file.write("\n\n".join(transcriptions))

            with open("video_info.json", "w", encoding="utf-8") as f:
                json.dump(data_list, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()

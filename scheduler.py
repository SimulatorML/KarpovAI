import os
import time
import schedule
import subprocess

APP_FILE = "app.py"

def stop_bot():
    """
    Stopping the bot
    """
    # Остановка бота
    try:
        subprocess.run(["pkill", "-f", "python " + APP_FILE])
        print("Bot has been stopped.")
    except Exception as e:
        print(f"Error stopping the bot: {e}")

def update_index():
    """
    Check if new indexes have been added. If they are, stop the bot, add them and restart it
    """
    # Открываем файл с url-ами и запоминаем его состояние
    with open(r'data/urls_of_channel_videos.txt', 'r') as file:
        old_content = file.read()

    # Запускаем пайплайн. Если новые видео будут найдены, в файл добавятся их url-ы
    subprocess.run(['python', r'data_pipelines/index_pipeline.py'])

    # Открываем файл с url-ами после изменения
    with open(r'data/urls_of_channel_videos.txt', 'r') as file:
        new_content = file.read()

    # Если файл с url-ами изменился, останавливаем бота, добавляем новые индексы, перезапускаем бота
    if old_content == new_content:
        print("No new videos found.")
    else:
        try:
            stop_bot()  # Останавливаем бот
            subprocess.Popen(["python", APP_FILE])  # Перезапускаем бот
            print("Bot has been updated and restarted.")
        except Exception as e:
            print(f"Error updating and restarting the bot: {e}")

# Планируем обновление индексов
schedule.every().monday.at("00:00").do(update_index)

if __name__ == "__main__":
    # Запускаем бот
    subprocess.Popen(["python", APP_FILE])

    # Запускаем шедулер
    while True:
        schedule.run_pending()
        time.sleep(1)

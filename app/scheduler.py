import time
import logging
from logging.handlers import RotatingFileHandler
import subprocess
from dataclasses import dataclass
import schedule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(module)s: %(message)s",
    handlers=[
        RotatingFileHandler("logs/app.log", maxBytes=5000000, backupCount=2),
        logging.StreamHandler()
    ]
)


@dataclass()
class Scheduler:
    """Index update scheduler class"""
    app_file: str = "app.py"
    index_update_file: str = '../data_pipelines/index_pipeline.py'
    url_file_path: str = '../data/urls_of_channel_videos.txt'
    bot_process: subprocess.Popen = None

    def stop_bot(self):
        """Stopping the bot"""
        if self.bot_process:
            self.bot_process.terminate()  # Используем метод terminate() для остановки процесса
            logging.info("Bot has been stopped.")
        else:
            logging.info("Bot is not running.")

    def update_index(self):
        """Check if new indexes have been added.
        If they are, stop the bot, add them and restart it"""
        # Открываем файл с url-ами и запоминаем его состояние
        with open(self.url_file_path, 'r', encoding="utf-8") as file:
            old_content = file.read()

        # Запускаем пайплайн. Если новые видео будут найдены, в файл добавятся их url-ы
        subprocess.run(['python', self.index_update_file], check=False)

        # Открываем файл с url-ами после изменения
        with open(self.url_file_path, 'r', encoding="utf-8") as file:
            new_content = file.read()

        # Если файл с url-ами изменился, останавливаем бота,
        # добавляем новые индексы, перезапускаем бота
        if old_content == new_content:
            logging.info("No new videos found.")
        else:
            self.stop_bot()  # Сначала останавливаем бот

            self.bot_process = subprocess.Popen(["python", self.app_file])  # Перезапускаем бот
            logging.info("Bot has been updated and restarted.")

    def run(self):
        """Run scheduler"""
        self.bot_process = subprocess.Popen(["python", self.app_file])

        # Планируем обновление индексов
        schedule.every().sunday.at("03:00").do(self.update_index)
        # schedule.every().day.at("23:16").do(self.update_index)

        # Запускаем шедулер
        while True:
            schedule.run_pending()
            time.sleep(1)


if __name__ == "__main__":
    scheduler = Scheduler()
    scheduler.run()

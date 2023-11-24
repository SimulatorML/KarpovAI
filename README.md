# KarpovAI
---
Telegram bot aimed at helping within data science-related queries. Runs with GPT 3.5 (default) under the hood, powered with a Retrieval Augmented Generation (RAG) system connected to a vector database of transcribed data science videos corpus. With the help of llamaindex, it retrieves the most relevant node and asks GPT to answer the question based on the context, if possible.

## Features
- **Contextual Data Science Responses:** Leverages a comprehensive database of video transcriptions for informed answers.
- **RAG and GPT Integration:** Combines the strengths of RAG and GPT for nuanced query understanding and response generation.
- **Telegram Bot Interface:** Easy-to-use interface on Telegram for querying data science topics.

## Project Structure
- `app.py`: Launches the bot in a specified Telegram chat.
- `evaluation/`
  - `evaluator.py`: Demonstrates the evaluation of the RAG system.
  - `evaluate/_json.json`: Contains the output from the evaluation system.
  - `traffic_evaluate/`: Tools for evaluating bot demand in a given Telegram chat (for internal use).
- `demo/rag.py`: Demonstrates the functionality of the RAG system.
- `data/`: Contains data for creating the vector database, including transcribed videos split into nodes with metadata and control questions, and video URLs.
  - `index_storage/`: Stores the vector database of the nodes.
- `data_pipelines/`: End-to-end pipeline scripts for video parsing, transcription (`parser_transcribe.py`), and indexing of transcripts (`index_pipeline.py`).
- `pytube/`: A fixed version of the pytube package used for fetching audio channels from YouTube videos.

## Getting Started
### Prerequisites
- Python 3.8+
- Telegram Bot API Token
- Access to GPT with OpenAI API token

### Installation
1. Clone the repository:
`git clone [repository URL]`
2. Install the dependencies:
`pip install -r requirements.txt`


### Configuration
- Set up your .env file with the following keys:
    - TG_TOKEN: Bot Telegram Token
    - BOT_ID: ID of the Bot
    - API_KEY: OpenAI API KEY

## Usage
Start the Bot in a specified chat by executing `app.py`


## The main tools used in the project
- ### Python
- ### GPT 3.5
- ### Llamaindex
- ### Aiogram

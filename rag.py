from llama_index import Document
from llama_index.node_parser import SimpleNodeParser
from llama_index import VectorStoreIndex, get_response_synthesizer, ServiceContext
from llama_index.retrievers import VectorIndexRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.postprocessor import SimilarityPostprocessor
from dotenv import load_dotenv
import json
import openai
import os

load_dotenv()
openai.api_key = os.getenv("API_KEY")
with open("video_info.json", "r", encoding="utf-8") as f:
    data_json = json.load(f)
# 1
documents = [
    Document(
        text=data["text"][0],
        metadata={"url": data["url"][0], "title": data["title"][0]},
    )
    for data in data_json
]
# 2
node_parser = SimpleNodeParser.from_defaults(chunk_size=200, chunk_overlap=50)
service_context = ServiceContext.from_defaults(node_parser=node_parser)

# 3
index = VectorStoreIndex.from_documents(documents, service_context=service_context)

query_engine = index.as_query_engine(
    include_text=True,
    response_mode="no_text",
    embedding_mode="hybrid",
    similarity_top_k=3,
)
question = "Какие софт скиллы нужны для дата саентиста"
retrival = query_engine.query(
    question,
)
print(f"Q: {question}")
information = [
    (i.text, i.metadata["url"], i.metadata["title"]) for i in retrival.source_nodes
]
information_text = " ".join([text for text, _, _ in information])
information_url = "/n".join(
    set([f"Karpov.courses: {url} - {title})" for _, url, title in information])
)

template = (
    "Ниже мы предоставили контекстную информацию\n"
    "---------------------\n"
    f"{information_text}"
    "\n---------------------\n"
    f"Учитывая эту информацию, ответьте, пожалуйста, на вопрос: {question}\n"
)
model_name = "gpt-3.5-turbo"
response = openai.ChatCompletion.create(
    model=model_name, messages=[{"role": "user", "content": template}]
)
print(f"А:{response.choices[0]['message']['content']}\n\n{information_url}")

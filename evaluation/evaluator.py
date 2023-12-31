import os
import json
# import pandas as pd
from dotenv import load_dotenv
import openai
from llama_index import Document
from llama_index.llms import OpenAI
from llama_index.evaluation import DatasetGenerator, RelevancyEvaluator, FaithfulnessEvaluator
# from llama_index.node_parser import SimpleNodeParser
from llama_index import StorageContext, load_index_from_storage
from llama_index import ServiceContext
# from data_pipelines.parser_transcribe import ParserTranscribe, get_video_urls

load_dotenv()
openai.api_key = os.getenv("API_KEY")

storage_cntxt = StorageContext.from_defaults(persist_dir="data/index_storage_1024")
idx = load_index_from_storage(storage_cntxt)
print("Index is loaded")
with open("data/video_info.json", "r", encoding="utf-8") as f:
    data = json.load(f)
data = data[:2]
docs = [
    Document(
        text=d["text"][0],
        metadata={"url": d["url"][0], "title": d["title"][0]},
    )
    for d in data
]
data_generator = DatasetGenerator.from_documents(
    docs,
    num_questions_per_chunk=2,
    question_gen_query="На русском языке"
)
eval_questions = data_generator.generate_questions_from_nodes()
gpt3 = OpenAI(temperature=0, model="gpt-3.5-turbo")
service_context_gpt3 = ServiceContext.from_defaults(llm=gpt3)
rel_evaluator_gpt3 = RelevancyEvaluator(service_context=service_context_gpt3)
faith_evaluator_gpt3 = FaithfulnessEvaluator(service_context=service_context_gpt3)

query_engine = idx.as_query_engine(
    include_text=True,
    response_mode="no_text",
    embedding_mode="hybrid",
    similarity_top_k=3,
)

evaluate_table = []

for question in eval_questions:
    evaluate_question = {}
    retrival = query_engine.query(
        question,
    )
    information = [
        (i.text, i.metadata["url"], i.metadata["title"]) for i in retrival.source_nodes
    ]
    contexts = [i.text for i in retrival.source_nodes]
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
    ).choices[0]['message']['content']
    relevancy_score = rel_evaluator_gpt3.evaluate(question, response, contexts).passing
    faithfulness_score = faith_evaluator_gpt3.evaluate(question, response, contexts).passing
    evaluate_question["question"] = question
    evaluate_question["sourses"] = information_text
    evaluate_question["video_sourse"] = information_url
    evaluate_question["response"] = response
    evaluate_question["faithfulness"] = faithfulness_score
    evaluate_question["relevancy"] = relevancy_score
    evaluate_table.append(evaluate_question)
with open("evaluate_json.json", "w", encoding="utf-8") as f:
    json.dump(evaluate_table, f, ensure_ascii=False, indent=4)

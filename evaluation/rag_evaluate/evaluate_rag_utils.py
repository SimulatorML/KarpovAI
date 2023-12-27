"""
Utilites for RAG system evaluation.
Question generation:
    class NodeSampler:
        sample informative nodes
    class QAGenerator:
        generate questions, link them to the initial video json
Evaluation:
    class Validator:
        get the context and answers to the provided questions,
        evaluate the retrival/answering using the provided evaluators.
"""


# typing
from typing import List, Sequence, Dict

import asyncio
import json

# for sampling of the nodes
import numpy as np

# imports for typing
from llama_index.schema import BaseNode
from llama_index.indices.base import BaseIndex
from llama_index.evaluation.base import BaseEvaluator
from llama_index.evaluation import EmbeddingQAFinetuneDataset
from llama_index.embeddings.openai import OpenAIEmbedding

from llama_index import (
                    Document,
                    StorageContext,
                    load_index_from_storage,
                    ServiceContext
                )
from llama_index.llms import OpenAI
from llama_index.node_parser import SimpleNodeParser
from llama_index.evaluation import (
                            generate_question_context_pairs,
                            RelevancyEvaluator,
                            FaithfulnessEvaluator
                            )
from llama_index.indices.postprocessor import PrevNextNodePostprocessor


class NodeSampler:
    """
    Given a set of documents in json format, split into nodes
    and sample those having relevant keywords
    """
    def __init__(self, relevant_keywords_path: str, not_relevant_keywords_path: str):
        # get relevant and irrelevant keywords
        with open(relevant_keywords_path, 'r') as file:
            self.relevant_keywords = [line.strip().lower() for line in file]

        with open(not_relevant_keywords_path, 'r') as file:
            self.not_relevant_keywords = [line.strip().lower() for line in file]
        # for future sampling
        self.rng = np.random.default_rng()

    def _count_words(self, text: str, word_list: list) -> int:
        """Counts occurrences of words from the word_list in the text."""
        return sum(text.count(word) for word in word_list)

    def _node_weight(self, node: BaseNode) -> int:
        """Reads a node text and returns its weight
        based on the count of relevant and irrelevant words."""
        text = (node.metadata['title'][0] + ' ' + ''.join(node.text)).lower()
        relevant_count = self._count_words(text, self.relevant_keywords)
        irrelevant_count = self._count_words(text, self.not_relevant_keywords)

        # feel free to change this way of gettings weights
        weight = max(0, relevant_count - irrelevant_count ** 2)

        # modify the node by adding weight info
        node.metadata['weight'] = weight

        return weight

    def _filtered_nodes(self, nodes: Sequence[BaseNode]) -> List[BaseNode]:
        """
        Process a list of nodes based on the following criteria:
        
        1. If a node has the same title as the previous one, it's added to the result list 
        only if its word count is more than 50% of the previous node's word count.
        2. If a node has a different title, it's simply added to the result list.

        Parameters:
        - nodes (list): A list of BaseNode objects.

        Returns:
        - list: A processed list of BaseNode objects.
        """

        # Initialize an empty list to store the processed nodes
        result = []

        # Initialize variables to keep track of the previous node's title and word count
        prev_title = None
        prev_word_count = 0

        # Iterate through the list of nodes
        for node in nodes:
            current_title = node.metadata['title']
            current_word_count = len(node.text.split())

            # Check if the current node has the same title as the previous one
            if current_title == prev_title:
                if current_word_count > 0.5 * prev_word_count:
                    result.append(node)
            else:
                result.append(node)

            # Update the previous title and word count for the next iteration
            prev_title = current_title
            prev_word_count = current_word_count

        return result

    def informative_nodes(
            self,
            video_info_path: str,
            chunk_size: int = 1024,
            fraction: float = 0.2,
            ) -> List[BaseNode]:
        """
        Given the corpus of the documents stored in json format,
        1. Split each document into nodes of `chunk_size` tokens,
        2. Specify weight to each node based on its information
        (count of relevant and irrelevant words),
        3. Sample nodes (fraction size) without replacement with the provided weights.
        4. Output the sampled nodes

        Parameters
        ----------
        video_info_path: str
            path to video_info.json
        chunk_size: int
            (default 1024)
            size of the nodes (in tokens) to produce
        fraction: float
            (default 0.2)
            fraction from all nodes to be sampled

        Returns
        -------
        List[BaseNode]
            list of nodes selected by sampling
        """

        # open json with transcribed videos and parse
        with open(video_info_path, "r", encoding="utf-8") as f:
            video_json = json.load(f)

        # a list of Documents to be used by SimpleNodeParser
        # to make a list of Nodes
        docs = [
            Document(
                text="".join(video_dict["text"]),
                metadata={"url": video_dict["url"][0], "title": video_dict["title"][0]},
            )
            for video_dict in video_json
        ]

        # parse the docs into the nodes
        parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        nodes = parser.get_nodes_from_documents(docs)

        # process the nodes to discard the last node of the video
        # if it is very shortened during splitting
        nodes = self._filtered_nodes(nodes)

        # Process nodes and calculate weights
        weights = []
        for node in nodes:
            weight = self._node_weight(node)
            weights.append(weight)

        # Normalize weights to use in random sampling (if they are not all negative)
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights if total_weight > 0]

        # Randomly sample nodes
        sample_size = int(fraction * len(nodes))  # fraction of the nodes
        sampled_nodes = self.rng.choice(
            nodes,
            size=sample_size,
            replace=False,
            p=normalized_weights
            )

        # sort the sampled nodes based on their weights (relevancy)
        return sorted(sampled_nodes, key=lambda x: x.metadata['weight'], reverse=True)


class QAGenerator:
    """
    given a sequence of nodes (context), generates questions to them.
    can save the resulting `EmbeddingQAFinetuneDataset` json notation
    with the correct encoding.
    link the initial set of documents to the generated questions.
    """

    def __init__(
        self,
        llm: OpenAI = None,
        qa_generate_prompt_tmpl: str = None,
    ):
        """
        llm: OpenAI
            default(None)
            language model for generation
            if None, 'gpt-3.5-turbo-1106' is used
        qa_generate_prompt_tmpl: str
            (default None)
            prompt for question generation
            if None, uses standard one
        """

        if not llm:
            # initialize the model
            llm = OpenAI(temperature=0, model="gpt-3.5-turbo-1106")

        if not qa_generate_prompt_tmpl:
            # make a prompt template
            qa_generate_prompt_tmpl = """\
            Ты - эксперт в составлении вопросов в области анализа данных.
            Внизу указана контекстная информация.

            ---------------------
            {context_str}
            ---------------------

            На основе приведенного текста составь только один вопрос, \
            на который можно ответить с помощью текста. \
            Вопрос должен покрыть как можно больше аспектов в тексте. \
            Он должен быть только на основе приведенного текста \
            и относиться к области анализа данных.
            Пожалуйста, ограничь размер вопроса 10 словами."
            """

        self.llm = llm
        self.qa_generate_prompt_tmpl = qa_generate_prompt_tmpl

    def _generate_questions(
        self,
        nodes: Sequence[BaseNode],
        save_json: bool = True,
        qa_json_path: str = None,
    ) -> EmbeddingQAFinetuneDataset:
        """
        Generates questions to the nodes based on the prompt and llm.

        Parameters:
        -----------
        nodes: Sequence[BaseNode]
            nodes to generate questions to
        save_json: bool
            (default True)
            whether to save resulting EmbeddingQAFinetuneDataset as json
        qa_json_path: str
            (default None)
            filename to save resulting EmbeddingQAFinetuneDataset json
        
        Returns:
        --------
        EmbeddingQAFinetuneDataset
            question: node dataset
        """

        # query based on the nodes.
        # num_questions_per_chunk will anyway not appear in the query,
        # it can be any value :)
        qa_dataset = generate_question_context_pairs(
            nodes=nodes,
            llm=self.llm,
            qa_generate_prompt_tmpl=self.qa_generate_prompt_tmpl,
            num_questions_per_chunk=1
            )

        # save EmbeddingQAFinetuneDataset json if required
        if save_json:
            if not qa_json_path:
                qa_json_path = 'question_context_pairs.json'
            with open(qa_json_path, 'w', encoding='utf8') as file:
                file.write(qa_dataset.json(ensure_ascii=False, indent=4))

        return qa_dataset

    def generate_document_question_pairs(
        self,
        nodes: Sequence[BaseNode],
        doc_json_path: str,
        doc_question_path: str,
        save_json: bool = True,
        qa_json_path: str = None,
    ) -> None:
        """
        Generates questions to the nodes based on the prompt and llm.
        Saves EmbeddingQAFinetuneDataset object if needed.
        Links questions to the initial video documents.

        Parameters:
        -----------
        nodes: Sequence[BaseNode]
            nodes to generate questions to
        doc_json_path: str
            path to the json file with video docs
        doc_question_path: str
            path at which to save modified video docs json
            with added questions
        save_json: bool
            (default True)
            whether to save resulting EmbeddingQAFinetuneDataset as json
        qa_json_path: str
            (default None)
            filename to save resulting EmbeddingQAFinetuneDataset json
        
        Returns:
        --------
        EmbeddingQAFinetuneDataset
            question: node dataset
        """
        # open json with transcribed videos and parse
        with open(doc_json_path, "r", encoding="utf-8") as f:
            doc_json = json.load(f)

        # get QA pairs as EmbeddingQAFinetuneDataset
        qa_dataset = self._generate_questions(
            nodes=nodes,
            save_json=save_json,
            qa_json_path=qa_json_path
        )

        # link node_id: video_url
        node_id_url = {node.id_: node.metadata['url'] for node in nodes}

        # link question_id: node_id
        question_id_node_id = {
            question_id: qa_dataset.relevant_docs[question_id][0]
            for question_id in qa_dataset.relevant_docs
            }

        # link question_id: video_url
        question_id_url = {
            question_id: node_id_url[question_id_node_id[question_id]]
            for question_id in question_id_node_id
            }

        # invert the 'question_id_url'
        # link video_url: list[question_id]
        url_question_id = {}

        for key, value in question_id_url.items():
            if value in url_question_id:
                url_question_id[value].append(key)
            else:
                url_question_id[value] = [key]

        # replace question_id by question_text
        # link video_url: list[question]
        url_question = {
            url: [
                qa_dataset.queries[question_id]
                    for question_id in url_question_id[url]
                    ]
                    for url in url_question_id
                    }

        # extend the original json
        # by creating 'control_questions': list[question] pairs
        # for each video's dict
        for doc in doc_json:
            key = doc['url'][0]
            if key in url_question:
                doc['control_questions'] = url_question[key]
            else:
                doc['control_questions'] = None

        # dump the final json to a new file
        with open(doc_question_path, "w", encoding="utf-8") as f:
            json.dump(doc_json, f, ensure_ascii=False, indent=4)

        return qa_dataset


class Validator:
    """
    Given a set of questions, pulls the most relevant node(s),
    asks LLM to answer questions based on the provided context
    and validates the resulting answers
    """
    def __init__(
        self,
        client,
        semaphore: asyncio.Semaphore,
        index: BaseIndex = None,
        persist_dir: str = None,
        has_node_postprocessors: bool = False,
        num_nodes: int = 1,
        node_postprocessors_mode: str = 'both',
        embed_model = OpenAIEmbedding,
        context_prompt: str = None,
        top_k: int = 3,
    ):
        """
        Parameters:
        -----------
        client: OpenAI async client
        semaphore: asyncio.Semaphore
            Semaphore object to control concurrent requests
        index: BaseIndex
            (defaults None)
            Index database instantiated
            If None, tries to load from the directory provided.
        persist_dir: str
            (default None)
            directory of an index database
        has_node_postprocessors: bool
            (default False)
            Whether to include neighbouring nodes
            from the original document
            for the context.
        num_nodes: int
            (default 1)
            if has_node_postprocessors set True,
            indicates number of neighbouring nodes to retrieve.
        node_postprocessors_mode: str
            (default 'both')
            if has_node_postprocessors set True,
            indicates whether to go forward ('next'),
            backward ('previous'), or both ('both').
        embed_model
            (default OpenAIEmbedding)
            model to get embeddings, used in the index.
            Not used if the index was provided.
        context_prompt: str
            (default None)
            prompt to use for answer generation
            if None, uses default prompt
        top_k: int
            (default 3)
            number of most relevant nodes to retrieve
        
        Raises:
        -------
        ValueError
            if both index and persist_dir are None.
        """
        # get an index if not provided
        if not index:
            if not persist_dir:
                raise ValueError(
                    'Provide either index or directory to load index from, none was provided.'
                    )
            storage_context = StorageContext.from_defaults(persist_dir=persist_dir)
            service_context = ServiceContext.from_defaults(embed_model=embed_model)
            index = load_index_from_storage(storage_context, service_context=service_context)

        # get a prompt if not provided
        if not context_prompt:
            context_prompt = (
                "Ниже мы предоставили контекстную информацию\n"
                "---------------------\n"
                "{information_text}"
                "\n---------------------\n"
                "Учитывая эту информацию, ответьте, пожалуйста, на вопрос: {query}\n"
                "\n---------------------\n"
                "Ответ на вопрос должен быть развернутым, полным, и охватывать множество "
                "аспектов заданного вопроса"
            )

        # params to create query engine
        params = {
            "include_text": True,
            "response_mode": "no_text",
            "embedding_mode": "hybrid",
            "similarity_top_k": top_k
        }

        # if retrieving neighboring nodes as well
        if has_node_postprocessors:
            postprocessor = PrevNextNodePostprocessor(
            docstore=index.docstore,
            num_nodes=num_nodes,  # number of nodes to fetch when looking forwards or backwards
            mode=node_postprocessors_mode,  # can be either 'next', 'previous', or 'both'
        )
            params['node_postprocessors'] = [postprocessor]

        self.query_engine = index.as_query_engine(**params)

        self.context_prompt = context_prompt
        self.client = client
        self.semaphore = semaphore

    async def answer_evaluated(
        self,
        query: str,
        output_table: List[dict],
        model_name: str = "gpt-3.5-turbo-1106",
        evaluators: Dict[str, BaseEvaluator] = None,
    ):
        """
        Get an answer for a given query based on the retrieved context.
        Also, validates the answer using provided evaluators.

        Parameters:
        -----------
        query: str
            query to answer
        output_table: List[dict]
            list to append results to.
            results are in a form of key-value pairs:
                'query': question asked
                'sources': node text used for answering
                'sources_urls': video urls of the nodes
                'response': llm response
                'metric_name': value of a metric_name from evaluators
        model_name: str
            (default gpt-3.5-turbo-1106)
            name of llm for answering
        evaluators: Dict[str, BaseEvaluator]
            (default None)
            dict with metric name: Evaluator pairs
            if None, relevancy and faithfullness are used
            with RelevancyEvaluator and FaithfullnessEvaluator.                      
        """
        async with self.semaphore:
            # to be used in output
            evaluate_result = {}

            # get the relevant nodes
            retrival = await self.query_engine.aquery(query)
            information = [
                (i.text, i.metadata["url"], i.metadata["title"]) for i in retrival.source_nodes
            ]

            # get the text and URL metadata of the retrieved nodes
            information_text = " ".join([text for text, _, _ in information])
            information_urls = list(
                {
                f"Karpov.courses: {url} - {title})" for _, url, title in information
            }
            )

            contexts = [node.text for node in retrival.source_nodes]

            # add the retrieved data to the output
            evaluate_result["query"] = query
            evaluate_result["sources"] = contexts
            evaluate_result["sources_urls"] = information_urls

            # query the model with the provided query and retrieved context
            response = await self.client.chat.completions.create(
                model=model_name,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": self.context_prompt.format_map(
                        {
                            'information_text': information_text,
                            'query': query
                        }
                    )
                    }
                    ]
            )

            # get the response, add it to the output
            response = response.choices[0].message.content
            evaluate_result["response"] = response

            # evaluate and add the result to the output
            for metric_name, evaluator in evaluators.items():
                evaluate_result[metric_name] = await evaluator.aevaluate(
                    query,
                    response,
                    contexts
                    )
                evaluate_result[metric_name] = evaluate_result[metric_name].passing

            output_table.append(evaluate_result)

    async def answers_evaluated_list(
        self,
        queries: Sequence[str],
        output_path: str,
        model_name: str = "gpt-3.5-turbo-1106",
        evaluators: Dict[str, BaseEvaluator] = None,
    ):
        """
        Get answers for given queries based on the retrieved contexts.
        Also, validate the answers using provided evaluators.

        Parameters:
        -----------
        query: str
            query to answer
        output_path: str
            path to put result to as JSON.
            result is list of dictionaries of key-value pairs:
                'query': question asked
                'sources': node text used for answering
                'sources_url': video url of the node
                'response': llm response
                'metric_name': value of a metric_name from evaluators  
        model_name: str
            (default gpt-3.5-turbo-1106)
            name of llm for answering
        evaluators: Dict[str, BaseEvaluator]
            (default None)
            dict with metric name: Evaluator pairs
            if None, relevancy and faithfullness are used
            with RelevancyEvaluator and FaithfullnessEvaluator.  
        """
        # get the evaluators if not provided
        if not evaluators:
            evaluators = {
                'relevancy': RelevancyEvaluator(),
                'faithfullness': FaithfulnessEvaluator()
            }
        # to be used by answer_evaluated()
        output_table = []

        # parallelize the retrieving/querying tasks
        async with asyncio.TaskGroup() as tg:
            _ = [
                tg.create_task(
                self.answer_evaluated(
                    query=query,
                    output_table=output_table,
                    model_name=model_name,
                    evaluators=evaluators,
                )
            )
            for query in queries
            ]

        # write resulting dataset to the JSON
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_table, f, ensure_ascii=False, indent=4)

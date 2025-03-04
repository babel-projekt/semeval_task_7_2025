import ast
import json
from typing import Any, Dict, List, Tuple
import os
import numpy as np
import pandas as pd
import torch
from rerankers import Document, Reranker
from sentence_transformers import SentenceTransformer
from together import Together
from dotenv import load_dotenv

load_dotenv()

LLM_RERANKING_BASE_PROMPT = "You are an expert fact-checker and information retrieval specialist.  \
        Your task is to analyze a query and a set of articles to identify the most relevant ones for fact-checking purposes. \
    \n\nTask:\n1. Review the query that needs fact-checking\n2. \
    Analyze the candidate articles provided\n3. \
    Select the 10 most relevant articles that would be most useful for fact-checking the query\n \
    4. Return ONLY the article IDs of these 10 articles in a tab-separated format\n\nImportant Instructions:\n- Focus on selecting articles that:\n  * \
    Directly address the claim in the query\n  * Provide factual evidence or counter-evidence\n \
    * Come from reliable sources\n  * Contain specific details relevant to the query\n  * \
    Cover different a   spects of the claim for comprehensive fact-checking\n- \
    Output format must be EXACTLY:\n  * Only article IDs\n  * Tab-separated\n  * \
    One line only\n  * Top 10 articles in order of relevance\n  * \
    No explanations or additional text\n\nQuery for fact-checking: {query} "


def determine_max_batch_size() -> int:
    if torch.cuda.is_available():
        print("CUDA is available")
        free_memory = torch.cuda.get_device_properties(
            0
        ).total_memory - torch.cuda.memory_allocated(0)
        max_batch_size = min(256, int((free_memory * 0.7) / (4 * 1024)))
    else:
        print("CUDA unavailable")
        max_batch_size = 32  # Default for CPU
    return max_batch_size


client_base = Together(api_key=os.getenv("TOGETHER_API_KEY"))

def get_matching_facts(
    model: SentenceTransformer,
    post_data: str,
    fact_embeddings: np.ndarray,
    fact_ids: List[str],
    max_batch_size: int,
    prompt_name: str = None,
    top_k: int = 10,
) -> List[str]:
    post_embedding = model.encode(
        [post_data],
        batch_size=max_batch_size,
        show_progress_bar=True,
        prompt_name=prompt_name,
    )
    similarities = model.similarity(post_embedding, fact_embeddings)[0].tolist()
    sort_index = [
        b[0] for b in sorted(enumerate(similarities), key=lambda i: i[1], reverse=True)
    ][:top_k]
    return [fact_ids[idx] for idx in sort_index]


def embed_fact_search_space(
    model: SentenceTransformer,
    fact_search_space: Dict[str, Dict[str, Dict[str, Any]]],
    max_batch_size: int,
    prompt_name: str = None,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    for lang in fact_search_space.keys():
        # Collect all texts and their corresponding fact check IDs
        texts: List[str] = []
        fact_ids: List[str] = []

        for fact_check_id in fact_search_space[lang].keys():
            texts.append(fact_search_space[lang][fact_check_id]["data"])
            fact_ids.append(fact_check_id)

        embeddings = model.encode(
            texts,
            batch_size=max_batch_size,
            show_progress_bar=True,
            prompt_name=prompt_name,
        )
        for i, fact_check_id in enumerate(fact_ids):
            fact_search_space[lang][fact_check_id]["embedding"] = embeddings[i]

        # Free memory
        del texts
        del fact_ids
    return fact_search_space


def rerank_docs(ranker, query, documents, to_consider=200, top_n=10, return_ids=True):
    docs = [Document(text=text["text"], doc_id=text["id"]) for text in documents][
        :to_consider
    ]
    results = ranker.rank(query=query, docs=docs)
    results = results.top_k(top_n)
    if return_ids:
        return [result.doc_id for result in results]
    return results


def rerank_docs_llm(
    client_base: Together,
    query: str,
    documents: List[Dict[str, Any]],
    to_consider: int = 100,
    top_n: int = 10,
    return_ids: bool = True,
    model_name: str = "Qwen/Qwen2.5-72B-Instruct-Turbo",
) -> List[str]:
    docs_to_consider = documents[:to_consider]
    base_prompt = LLM_RERANKING_BASE_PROMPT.format(query=query)
    candidate_articles = "\nCandidate Articles:"
    for doc in docs_to_consider:
        candidate_articles += f"\nID: {doc['id']}\tTEXT: {doc['text']}"

    # Add final instructions
    final_instructions = "\nOutput the top 10 most relevant article IDs in this exact format:\nID1\tID2\tID3\tID4\tID5\tID6\tID7\tID8\tID9\tID10\nONLY RETURN tab seperated IDs....NOTHING ELSE, PLEASE NOTHING ELSE......"

    # Combine all parts of the prompt
    final_prompt = base_prompt + candidate_articles + final_instructions

    # Get response from LLM
    response = client_base.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": final_prompt}],
    )
    response_str = response.choices[0].message.content
    # Parse the response to get document IDs
    # Clean up any newlines and split by tabs
    doc_ids = response_str.replace("\n", " ").strip().split("\t")

    # Convert string IDs to integers if needed and take only top_n
    try:
        doc_ids = [int(doc_id) for doc_id in doc_ids[:top_n]]
    except ValueError:
        # If conversion fails, just use the strings
        doc_ids = doc_ids[:top_n]
    return doc_ids if return_ids else docs_to_consider


TEST_DATA_PATH = "data_test"

with open(f"{TEST_DATA_PATH}/tasks.json", "r") as f:
    tasks = json.load(f)

posts_df = pd.read_csv(f"{TEST_DATA_PATH}/posts.csv")
fact_checks_df = pd.read_csv(f"{TEST_DATA_PATH}/fact_checks.csv")

post_dict = posts_df.set_index("post_id").T.to_dict()
fact_dict = fact_checks_df.set_index("fact_check_id").T.to_dict()

monolingual_tasks = tasks["monolingual"]
crosslingual_tasks = tasks["crosslingual"]

all_monolingual_data = {}

for lang in monolingual_tasks.keys():
    all_monolingual_data[lang] = {}
    all_monolingual_data[lang]["post_ids"] = monolingual_tasks[lang]["posts_test"]
    all_monolingual_data[lang]["fact_check_ids"] = monolingual_tasks[lang][
        "fact_checks"
    ]
    all_monolingual_data[lang]["posts"] = {}
    all_monolingual_data[lang]["fact_checks"] = {}
    for post_id in all_monolingual_data[lang]["post_ids"]:
        post_data = post_dict[post_id]
        instances, ocr, verdicts, text = (
            post_data["instances"],
            post_data["ocr"],
            post_data["verdicts"],
            post_data["text"],
        )
        text = ast.literal_eval(text)
        ocr = ast.literal_eval(ocr) if isinstance(ocr, str) else ocr
        original_text, translated_text, lang_prob = text
        original_ocr, translated_ocr, lang_prob = (
            ocr if isinstance(ocr, str) else (None, None, None)
        )
        all_monolingual_data[lang]["posts"][post_id] = {}
        all_monolingual_data[lang]["posts"][post_id] = {
            "original_text": original_text,
            "g_translated_text": translated_text,
            "lang_prob": lang_prob,
            "verdicts": verdicts,
            "ocr": original_ocr,
            "g_translated_ocr": translated_ocr,
            "data": translated_text + " " + str(translated_ocr),
        }
    for fact_check_id in all_monolingual_data[lang]["fact_check_ids"]:
        fact_check_data = fact_dict[fact_check_id]
        fact_claim, fact_instances, fact_title = (
            fact_check_data["claim"],
            fact_check_data["instances"],
            fact_check_data["title"],
        )
        fact_claim = (
            ast.literal_eval(fact_claim)
            if isinstance(fact_claim, str)
            else (None, None, None)
        )
        fact_title = (
            ast.literal_eval(fact_title)
            if isinstance(fact_title, str)
            else (None, None, None)
        )
        original_fact_claim, translated_fact_claim, lang_prob = fact_claim
        original_fact_title, translated_fact_title, lang_prob = fact_title
        all_monolingual_data[lang]["fact_checks"][fact_check_id] = {
            "original_fact_claim": original_fact_claim,
            "g_translated_fact_claim": translated_fact_claim,
            "lang_prob": lang_prob,
            "original_fact_title": original_fact_title,
            "g_translated_fact_title": translated_fact_title,
            "lang_prob": lang_prob,
            "data": translated_fact_claim + " " + str(translated_fact_title),
        }
    # delete the post_ids and fact_check_ids from the data
    del all_monolingual_data[lang]["post_ids"]
    del all_monolingual_data[lang]["fact_check_ids"]

MODEL_NAMES = ["NovaSearch/stella_en_400M_v5"]
RERANKER_NAMES = [
    {
        "name": "cross-encoder",
    }
    # {
    #     "name": "mixedbread-ai/mxbai-rerank-large-v1",
    #     "model_type": "cross-encoder",
    # },
    # {
    #     "name": "t5",
    # },
    # {"name": "unicamp-dl/InRanker-base", "model_type": "t5"},
    # {"name": "colbert"},
    # {"name": "Qwen/Qwen2.5-72B-Instruct-Turbo"},
]

fact_search_space = {}
for lang in monolingual_tasks.keys():
    fact_search_space[lang] = {}
    for fact_check_id in all_monolingual_data[lang]["fact_checks"].keys():
        fact_search_space[lang][fact_check_id] = {}
        fact_search_space[lang][fact_check_id]["data"] = all_monolingual_data[lang][
            "fact_checks"
        ][fact_check_id]["data"]

# embed the fact search space
for model_name in MODEL_NAMES:
    for reranker in RERANKER_NAMES:
        results = {}
        model = SentenceTransformer(model_name, trust_remote_code=True)
        max_batch_size = determine_max_batch_size()
        fact_search_space = embed_fact_search_space(
            model, fact_search_space, max_batch_size, prompt_name=None
        )
        fact_embeddings = {}
        for lang in fact_search_space.keys():
            fact_ids = fact_search_space[lang].keys()
            fact_embeddings[lang] = np.array(
                [fact_search_space[lang][fact_id]["embedding"] for fact_id in fact_ids]
            )

        for lang in monolingual_tasks.keys():
            # Get the fact IDs for this specific language
            fact_ids = list(fact_search_space[lang].keys())

            for post_id in monolingual_tasks[lang]["posts_test"]:
                post_data = all_monolingual_data[lang]["posts"][post_id]["data"]
                # find the top 10 fact checks using the cosine similarity between the post and the fact check
                top_k = 100
                results[post_id] = get_matching_facts(
                    model=model,
                    post_data=post_data,
                    fact_embeddings=fact_embeddings[lang],
                    fact_ids=fact_ids,
                    max_batch_size=max_batch_size,
                    prompt_name=None,
                    top_k=top_k,
                )
                fact_docs = [
                    {"text": fact_search_space[lang][fact_id]["data"], "id": fact_id}
                    for fact_id in results[post_id]
                ]
                # if reranker["name"] == "Qwen/Qwen2.5-72B-Instruct-Turbo":
                #     reranked_docs = rerank_docs_llm(
                #         client_base=client_base,
                #         query=post_data,
                #         documents=fact_docs,
                #         to_consider=top_k,
                #         top_n=10,
                #         return_ids=True,
                #     )
                # else:
                #     ranker = Reranker(
                #         reranker["name"], model_type=reranker.get("model_type", None)
                #     )
                #     reranked_docs = rerank_docs(
                #         ranker=ranker,
                #         query=post_data,
                #         documents=fact_docs,
                #         to_consider=top_k,
                #         top_n=10,
                #         return_ids=True,
                #     )
                results[post_id] = fact_docs
        with open(
            f"{model_name.split('/')[-1]}_{top_k}_monolingual_predictions.json",
            "w",
        ) as outfile:
            json.dump(results, outfile)

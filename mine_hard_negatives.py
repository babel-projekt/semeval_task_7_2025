# Script to mine hard negatives from the translated query dataset

import os

# Set the environment variable
# os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

from datasets import load_dataset, concatenate_datasets

eng_dataset = load_dataset(
    "csv",
    data_files="negatives_base_data_with_translated_queries_v3_v.csv",
    split="train",
)
from sentence_transformers.util import mine_hard_negatives
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("NovaSearch/stella_en_400M_v5", trust_remote_code=True)

dataset_en = mine_hard_negatives(
    dataset=eng_dataset,
    anchor_column_name="query",
    positive_column_name="positive",
    model=model,
    range_min=10,
    range_max=100,
    margin=0.01,
    num_negatives=20,
    sampling_strategy="top",
    batch_size=128,
)

dataset = concatenate_datasets([dataset_en])
dataset.to_csv("en_dataset_stella_400_20_translated_query_v3_w_v.csv")

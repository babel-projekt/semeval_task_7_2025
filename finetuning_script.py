# Script to finetune the embedding model on the mined hard negatives

from datasets import load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
import os

os.environ["WANDB_DISABLED"] = "True"

from sentence_transformers.losses import TripletLoss

from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator

model = SentenceTransformer("dunzhang/stella_en_400M_v5", trust_remote_code=True)

dataset = load_dataset(
    "csv",
    data_files="en_dataset_stella_400_20_translated_query_v3_w_v.csv",
    split="train",
)
train_test_split = dataset.train_test_split(test_size=0.1, seed=42)  # 10% for dev set
train_dataset = train_test_split["train"]
eval_dataset = train_test_split["test"]
loss = TripletLoss(model=model)

args = SentenceTransformerTrainingArguments(
    # Required parameter:
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    learning_rate=5e-05,
    max_grad_norm=1.0,
    lr_scheduler_type="linear",
    fp16=False,
    bf16=False,
    optim="adamw_torch",
    batch_sampler=BatchSamplers.NO_DUPLICATES,
    eval_strategy="steps",
    eval_steps=1000,
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
    logging_steps=100,
)

# 6. (Optional) Create an evaluator & evaluate the base model
dev_evaluator = TripletEvaluator(
    anchors=eval_dataset["query"],
    positives=eval_dataset["answer"],
    negatives=eval_dataset["negative"],
)
dev_evaluator(model)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    loss=loss,
    evaluator=dev_evaluator,
)
trainer.train()

model.save("models/stella_en_400M_10_negatives")

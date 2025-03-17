<div align="center">

# TIFIN India at SemEval-2025: Harnessing Translation to Overcome Multilingual IR Challenges in Fact-Checked Claim Retrieval

[![Workshop](https://img.shields.io/badge/SemEval-2025-blue)](https://semeval.github.io/)  
[![Proceedings](https://img.shields.io/badge/ACL%20Anthology-SemEval%202025-green)](https://www.aclweb.org/anthology/venues/semeval/)  
[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://drive.google.com/file/d/1Mcpbwi7p5RitKrKV4qZhlucavENHSuRf/view?usp=sharing)

</div>

## Abstract
We address the challenge of retrieving previously fact-checked claims in monolingual and crosslingual settings - a critical task given the global prevalence of disinformation. Our approach follows a two-stage strategy: a reliable baseline retrieval system using a finetuned embedding model and an LLM-based reranker. Our key contribution is demonstrating how LLM-based translation can overcome the hurdles of multilingual information retrieval. Additionally, we focus on ensuring that the bulk of the pipeline can be replicated on a consumer GPU. Our final integrated system achieved a success@10 score of 0.938 (∼0.94) and 0.81025 on the monolingual and crosslingual test sets respectively.

## Setup

### 1. Create the Conda Environment

Run the following command to create a Conda environment with Python 3.10:

```bash
conda create --name claim_retrieval python=3.10 -y
conda activate claim_retrieval
pip install -r requirements.txt
```

### 2. Run the scripts

To train the model, first run the script to mine hard negatives, and then proceed with fine-tuning.

```bash
python mine_hard_negatives.py
python finetuning_script.py
```

## Dataset and Model Details

Our work utilized multiple datasets and a fine-tuned model, which are publicly available on Hugging Face. Below is a detailed description of each dataset and model, along with the corresponding links.

### Datasets

#### 1. Mined Negatives for Training
- **Description**: This dataset contains (anchor, positive, negative) triplets used for fine-tuning the embedding model. 20 negatives were mined per anchor-positive pair.
- **Link**: [Mined Negatives Task 7](https://huggingface.co/datasets/prasannad28/mined_negatives_task7)

#### 2. Base Data for Negative Mining
- **Description**: This dataset contains anchor and positive pairs that were used to mine hard negatives.
- **Link**: [Negatives Base Data](https://huggingface.co/datasets/prasannad28/negatives_base_data)

#### 3. Translated Facts
- **Description**: This dataset contains LLM-based translations (via Aya-Expanse) of the full fact-space for SemEval Task 7 - Multilingual and Crosslingual Fact-Checked Claim Retrieval. However, it was not used in the final pipeline due to a lack of performance gains. Further refinements were not pursued due to high computational costs.
- **Link**: [Translated Facts](https://huggingface.co/datasets/prasannad28/translated_facts)

#### 4. Translated Facts Test Set
- **Description**: This dataset contains the test set of translated facts, used for evaluation.
- **Link**: [Translated Facts Test Set](https://huggingface.co/datasets/prasannad28/translated_facts_test_set)

#### 5. Augmented Posts Test Set
- **Description**: This dataset contains augmented social media posts, used for testing retrieval performance.
- **Link**: [Augmented Posts Test Set](https://huggingface.co/datasets/prasannad28/augmented_posts_test_set)

### Model

#### Final Fine-Tuned Model
- **Description**: This is the final model trained using positive and negative pairs from a multilingual fact-checking dataset.
- **Link**: [Final Fine-Tuned Model](https://huggingface.co/prasannad28/stella-en-ft-v1.0)

#### Usage Example
```python
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('prasannad28/stella-en-aug-20-v0.6i')
```


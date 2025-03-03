<div align="center">

# Harnessing Translation to Overcome Multilingual IR Challenges in Fact-Checked Claim Retrieval

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

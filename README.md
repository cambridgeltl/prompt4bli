# Prompt4BLI
This repository is the official PyTorch implementation of the following paper:

Yaoyiran Li, Anna Korhonen, and Ivan Vulić. 2023. *On Bilingual Lexicon Induction with Large Language Models*. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023). [[arXiv]](https://arxiv.org/abs/2310.13995) [[OpenReview]](https://openreview.net/group?id=EMNLP/2023/Conference)

**Prompt4BLI** aims to address the Bilingual Lexicon Induction (BLI) / Word Translation tasks with **autoregressive Large Language Models (LLMs)**. We for the first time demonstrate that prompting multilingual LLMs for BLI outperforms traditional BLI approaches which rely on calculating cross-lingual word embeddings (CLWEs). While we show that prompting off-the-shelf LLMs can already establish new state-of-the-art BLI performance on many BLI language pairs (our main experimental setup), the Prompt4BLI repo also provides code for BLI-oriented fine-tuning which can further improve the results (as a side experiment, demonstrated on smaller-scale LLMs).

Traditional methods rely on learning parameterized CLWE mappings or cross-lingual word pair scoring functions and usually tackle BLI in three setups: (1) **Supervised**, 5K seed translation pairs; (2) **Semi-Supervised**, 1K seed translation pairs; (3) **Unsupervised**, 0 seed translation pairs. (cf. our previous work [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI) and [BLICEr](https://github.com/cambridgeltl/BLICEr)). Different from traditional methods, **Prompt4BLI** only makes use of off-the-shelf LLMs, **not** requiring LLM fine-tuning nor updating any learnable parameters. Our work considers the following prompting setups:

- **Few-Shot Prompting**: We propose to retrieve a subset of the seed translation pairs (nearest neighbour retrieval) as in-context examples for prompting. Corresponds to the traditional Supervised and Semi-Supervised BLI setups where the seed bilingual dictionary size is 5K and 1K respectively.
- **Zero-Shot Prompting**: No in-context examples are used. Corresponds to the traditional Unsupervised BLI setup.

# Dependencies
- PyTorch>=1.10.1
- Transformers>=4.28.1

# LLMs Used in Our Work

| LLM | (Hugging Face) Model ID |
| -------- | -------- |
| mT5-small | "google/mt5-small" |
| mT5-base | "google/mt5-base" |
| mT5-large | "google/mt5-large" |
| mT5-xl | "google/mt5-xl" |
| mT5-xxl | "google/mt5-xxl" |
| mT0-small | "bigscience/mt0-small" |
| mT0-base | "bigscience/mt0-base" |
| mT0-large | "bigscience/mt0-large" |
| mT0-xl | "bigscience/mt0-xl" |
| mT0-xxl | "bigscience/mt0-xxl" |
| XGLM-564M | "facebook/xglm-564M" |
| XGLM-1.7B | "facebook/xglm-1.7B" |
| XGLM-2.9B | "facebook/xglm-2.9B" |
| XGLM-4.5B | "facebook/xglm-4.5B" |
| XGLM-7.5B | "facebook/xglm-7.5B" |
| mGPT | "sberbank-ai/mGPT" |
| LLaMA-7B | "huggyllama/llama-7b" |
| LLaMA-13B | "huggyllama/llama-13b" |

# Data
Following [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI/) and [BLICEr](https://github.com/cambridgeltl/BLICEr), our data is obtained from the [XLING](https://github.com/codogogo/xling-eval) (8 languages, 56 BLI directions in total) and [PanLex-BLI](https://github.com/cambridgeltl/panlex-bli) (15 lower-resource languages, 210 BLI directions in total).

Get XLING data:
```bash
sh get_xling_data.sh
```

For PanLex-BLI, please see [./get_panlex_data](./get_panlex_data), where we provide the code for deriving the monolingual word embeddings.

# Run the Code
Prepare BLI Data and Extract In-Context Examples for Few-Shot Prompting (XLING):
```bash
python run_extract_vocabularies.py
python run_extract_bli_data.py
```

Prepare BLI Data and Extract In-Context Examples for Few-Shot Prompting (PanLex-BLI):
```bash
python run_extract_vocabularies_panlex.py
python run_extract_bli_data_panlex.py
```

(Optional) Run BLI-Oriented LLM Fine-Tuning (define LLM dirs, learning rate, batch size, and random seed in run_training.py):
```bash
python run_training.py
```

Run BLI Evaluation (define seed dictionary size, n_shot, LLM dir, and language pairs to evaluate manually in run_bli.py):
```bash
python run_bli.py
```

# Citation
Please cite our paper if you find **Prompt4BLI** useful.
```bibtex
@inproceedings{li-etal-2023-on,
    title     = {On Bilingual Lexicon Induction with Large Language Models},
    author    = {Li, Yaoyiran and Korhonen, Anna and Vuli{\'c}, Ivan},
    booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},    
    year      = {2023}
}
```

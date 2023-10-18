# Prompt4BLI
This repository is the official PyTorch implementation of the following paper:

Yaoyiran Li, Anna Korhonen, and Ivan Vulić. 2023. *On Bilingual Lexicon Induction with Large Language Models*. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023). [[arXiv]](https:./) [[OpenReview]](https:./)

**Prompt4BLI** aims to address Bilingual Lexicon Induction (BLI) / Word Translation with autoregressive Large Language Models (LLMs). We for the first time demonstrate that prompting multilingual LLMs for BLI outperforms traditional BLI approaches which rely on calculating cross-lingual word embeddings (CLWEs). While we show that prompting off-the-shelf LLMs can already establish new state-of-the-art BLI performance on many BLI language pairs, the Prompt4BLI repo also provides code for BLI-oriented fine-tuning which can further improve the results.

## Citation:
Please cite our paper if you find **Prompt4BLI** useful.
```bibtex
@inproceedings{li-etal-2023-on,
    title     = {On Bilingual Lexicon Induction with Large Language Models},
    author    = {Li, Yaoyiran and Korhonen, Anna and Vuli{\'c}, Ivan},
    booktitle = {Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing},    
    year      = {2023}
}
```

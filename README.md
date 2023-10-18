# Prompt4BLI
This repository is the official PyTorch implementation of the following paper:

Yaoyiran Li, Anna Korhonen, and Ivan Vulić. 2023. *On Bilingual Lexicon Induction with Large Language Models*. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP 2023). [[arXiv]](https:./) [[OpenReview]](https:./)

**Prompt4BLI** aims to address the Bilingual Lexicon Induction (BLI) / Word Translation tasks with **autoregressive Large Language Models (LLMs)**. We for the first time demonstrate that prompting multilingual LLMs for BLI outperforms traditional BLI approaches which rely on calculating cross-lingual word embeddings (CLWEs). While we show that prompting off-the-shelf LLMs can already establish new state-of-the-art BLI performance on many BLI language pairs (our main experimental setup), the Prompt4BLI repo also provides code for BLI-oriented fine-tuning which can further improve the results (as a side experiment, demonstrated on smaller-scale LLMs).

Traditional methods rely on learning parameterized CLWE mappings or cross-lingual word pair scoring functions and usually tackle BLI in three setups: (1) **Supervised**, 5K seed translation pairs; (2) **Semi-Supervised**, 1K seed translation pairs; (3) **Unsupervised**, 0 seed translation pairs. (cf. Our previous work [ContrastiveBLI](https://github.com/cambridgeltl/ContrastiveBLI), [BLICEr](https://github.com/cambridgeltl/BLICEr)). Different from traditional methods, **Prompt4BLI** only makes use of off-the-shelf LLMs, **not** requiring LLM fine-tuning nor updating any learnable parameters. Our work considers the following prompting setups:

- **Few-Shot Prompting**: We propose to retrieve a subset of the seed translation pairs (nearest neighbour retrieval) as in-context examples for prompting. Correspond to the traditional Semi-Supervised and Semi-Supervised BLI setups when the seed bilingual dictionary size is 5K and 1K respectively.
- **Zero-Shot Prompting**: No in-context examples are used. Correspond to the traditional Unsupervised BLI setup.

# Dependencies

# Data

# Run the Code


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

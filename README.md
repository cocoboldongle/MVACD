# Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection

The dataset used in the paper: *"Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection"* (ACL 2026 Main)

[![Paper](https://img.shields.io/badge/Paper-arXiv%202509.17292-red.svg)](https://arxiv.org/abs/2509.17292)
[![KoACD](https://img.shields.io/badge/🤗%20KoACD-HuggingFace-blue.svg)](https://huggingface.co/datasets/kma80kjs1/KoACD)
[![License](https://img.shields.io/badge/License-CC_BY--NC_4.0-green.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Overview

This repository contains the datasets and implementation details for the MVACD framework, which combines Large Language Models (LLMs) with a Multiple-Instance Learning (MIL) architecture for cognitive distortion detection.

Each utterance is decomposed into **Emotion, Logic, and Behavior (ELB)** components, processed by LLMs to infer multiple distortion instances with predicted types, expressions, and salience scores. These are integrated via a **Multi-View Gated Attention** mechanism for final classification.

---

## Datasets

### KoACD (Korean Adolescent Cognitive Distortion Dataset)

- **Source:** [Kim and Kim, 2025](https://arxiv.org/abs/2505.00367) — EMNLP Findings 2025
- **Full dataset:** 🤗 [HuggingFace](https://huggingface.co/datasets/kma80kjs1/KoACD)
- **Subset used in this paper:** 5,000 utterances (500 per distortion type), expert-validated by 10 Korean psychologists

> ⚠️ **Note:** There is no fixed train/validation/test split. Each experimental run uses a different random seed for stratified splitting (80/10/10).

| Cognitive Distortion Type | Count (%) |
|---------------------------|-----------|
| Labeling | 478 (10.6%) |
| Mental Filtering | 470 (10.4%) |
| All-or-Nothing Thinking | 464 (10.3%) |
| Emotional Reasoning | 458 (10.2%) |
| Personalization | 459 (10.2%) |
| Overgeneralization | 452 (10.0%) |
| Discounting the Positive | 451 (10.0%) |
| Jumping to Conclusions | 431 (9.6%) |
| Magnification and Minimization | 432 (9.6%) |
| Should Statements | 415 (9.2%) |
| **Total** | **4,510 (100%)** |

### Therapist QA Dataset

- **Source:** [Shreevastava and Foltz, 2021](https://aclanthology.org/2021.clpsych-1.17/)
- **Processed data (ELB + LLM instances):** Available upon request
- **Usage:** Primary label per utterance; used for cross-linguistic generalization benchmarking

> ⚠️ **Note:** There is no fixed train/validation/test split. Each experimental run uses a different random seed for stratified splitting (80/10/10).

| Split | Utterances |
|-------|-----------|
| Train | 1,277 (80%) |
| Validation | 159 (10%) |
| Test | 161 (10%) |
| **Total** | **1,597** |

---

## Results

### Input Configuration Comparison (Weighted F1)

| Configuration | KoACD Val | KoACD Test | Therapist QA Val | Therapist QA Test |
|--------------|-----------|------------|-----------------|-------------------|
| Baseline | 0.504 ± 0.019 | 0.473 ± 0.015 | 0.410 ± 0.038 | 0.340 ± 0.037 |
| ELB only | 0.519 ± 0.016 | 0.483 ± 0.017 | 0.438 ± 0.028 | 0.378 ± 0.036 |
| Salience only | 0.518 ± 0.015 | 0.486 ± 0.014 | 0.428 ± 0.036 | 0.360 ± 0.035 |
| **ELB + Salience** | **0.529 ± 0.018** | **0.505 ± 0.014** | **0.460 ± 0.029** | **0.394 ± 0.034** |

### LLM Baseline Comparison (F1)

| Model | KoACD | Therapist QA |
|-------|-------|--------------|
| Gemini 2.0 Flash | 0.386 | 0.348 |
| GPT-4o | 0.325 | 0.332 |
| Claude 3.7 Sonnet | 0.272 | 0.318 |

---

## Citation

If you use this work, please cite:

```bibtex
@article{kim2025mvacd,
  title   = {Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection},
  author  = {Kim, Jun Seo and Kim, Hyemi and Oh, Woo Joo and Cho, Hongjin and Lee, Hochul and Kim, Hye Hyeon},
  journal = {arXiv preprint arXiv:2509.17292},
  year    = {2025},
  url     = {https://arxiv.org/abs/2509.17292}
}
```

If you use the KoACD dataset, please also cite:

```bibtex
@inproceedings{kim2025koacd,
  title     = {KoACD: The First Korean Adolescent Dataset for Cognitive Distortion Analysis via Role-Switching Multi-LLM Negotiation},
  author    = {Kim, Jun Seo and Kim, Hye Hyeon},
  booktitle = {Findings of the Association for Computational Linguistics: EMNLP 2025},
  pages     = {22050--22078},
  year      = {2025},
  url       = {https://arxiv.org/abs/2505.00367}
}
```

---

## Contact

- **Jun Seo Kim** — [kma80kjs@gachon.ac.kr](mailto:kma80kjs@gachon.ac.kr)
- **Hye Hyeon Kim** — [hye_hyeon@yonsei.ac.kr](mailto:hye_hyeon@yonsei.ac.kr)

---

## License

This repository is licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/) and is available for **research purposes only**.

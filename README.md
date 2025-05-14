# ALMEA: Active Learning-Enhanced Multimodal Entity Alignment with Semantically-Calibrated Modality Imputation

This repository contains the implementation of our method proposed in the NeurIPS 2025 submission titled "ALMEA: Active Learning-Enhanced Multimodal Entity Alignment with Semantically-Calibrated Modality Imputation".

> **Note**: This repository is anonymous and intended solely for the purpose of double-blind peer review.

> Multimodal knowledge graphs (MMKGs) offer enriched knowledge representation by integrating structural, visual, and textual information from heterogeneous sources. However, existing multimodal entity alignment (MMEA) approaches face significant challenges due to missing modalities and semantic inconsistencies across sources. These limitations compromise alignment robustness, especially in low-resource scenarios with limited seed pairs. To bridge the gap, we propose \textbf{ALMEA}, a novel MMEA framework that integrates semantic calibration and active learning to improve alignment. Specifically, ALMEA synthesizes embeddings for missing modalities and refines semantic representations to address inconsistencies across MMKGs. With active learning strategy, it iteratively selects optimal candidate pairs within a learnable budget, enabling more effective acquisition of modality information in low-resource scenarios. Extensive experiments on benchmark MMKG datasets demonstrate that ALMEA consistently outperforms state-of-the-art baselines, achieving an MRR improvement of approximately 4.70\%. The effectiveness of the semantic calibration was also confirmed by the non-active variant in the ablation study, with accuracy improvements of 2.02\% and 2.85\%.
<p align="center">
  <img src="picture/Figure_one_png.png" alt="ALMEA Framework" width="700"/>
</p>

## 🔧 Environment Setup

We provide a `requirements.txt` file for setting up the Python environment. Below are the main dependencies used in this project:

```bash
pip install -r requirements.txt
```

## 📦 Details

- Python (>= 3.9)
- [PyTorch](https://pytorch.org/) (~= 2.5.1 + cu121)
- [NumPy](https://numpy.org/) (~= 2.2.1)
- [Transformers](https://huggingface.co/transformers/) (~= 4.47.1)
- [TQDM](https://tqdm.github.io/) (~= 4.66.4)
- [SciPy](https://scipy.org/) (~= 1.14.1)
- [Seaborn](https://seaborn.pydata.org/) (~= 0.13.2)
- [Matplotlib](https://matplotlib.org/) (~= 3.10.0)
- [Scikit-learn](https://scikit-learn.org/) (~= 1.5.2)
- [Pandas](https://pandas.pydata.org/) (~= 2.2.3)
- [EasyDict](https://pypi.org/project/easydict/) (~= 1.13)
- [Unidecode](https://pypi.org/project/Unidecode/) (~= 1.3.8)

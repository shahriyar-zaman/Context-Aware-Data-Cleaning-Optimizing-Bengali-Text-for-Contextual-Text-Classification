# 🧠 Context-Aware Data Cleaning: Optimizing Bengali Text for Contextual Text Classification

This repository contains the implementation, datasets, and evaluation scripts for the paper:

> **Context-Aware Data Cleaning: Optimizing Bengali Text for Contextual Text Classification**  
> *Moshiur Rahman Faisal, Abdur Rahman Fahad, Shahriyar Zaman Ridoy, Jannat Sultana, Zinnat Fowzia Ria, Md Hasibur Rahman, Mohammed Arif Uddin, Rashedur M. Rahman*  
> Published in *SN Computer Science (2025)* — [DOI: 10.1007/s42979-025-03891-9](https://doi.org/10.1007/s42979-025-03891-9)

---

## 📜 Overview

The project proposes a **context-aware data-cleaning pipeline** for Bengali NLP tasks.  
Unlike traditional cleaning (which removes punctuation, stop words, HTML tags, etc.), this pipeline **preserves linguistic context** while reducing noise — leading to up to **4 % accuracy improvement** across multiple Bengali text-classification benchmarks.

---

## ✨ Key Contributions

- **Novel Pipeline:** A context-aware data-cleaning pipeline (CADC) that integrates  
  - Spelling correction  
  - Tagging of HTML and URLs (`<URL>`, `<EMAIL>`)  
  - Preservation of punctuation and emojis  
  - TF-IDF-based selective word removal  

- **Empirical Evaluation:**  
  Benchmarked on four Bengali datasets — **BEmoC**, **SentNoB**, **UBMEC**, and **EmoNoBa** — using  
  machine-learning (SVM, RF, XGBoost, Naïve Bayes), deep learning (CNN, BiLSTM), and transformer-based models (BanglaBERT, mBERT, XLM-RoBERTa).

- **Performance Boost:**  
  Context-aware cleaning outperforms traditional methods by up to 4 % in accuracy and weighted F1-score.

- **Open Algorithms & Tools:**  
  Provides ready-to-use algorithms and code for replicating context-aware cleaning in other low-resource languages.

---

## 🧩 Methodology

### 🔹 1. Baseline Models
| Category | Models | Embeddings |
|-----------|---------|------------|
| Machine Learning | SVM, Random Forest, XGBoost, Naïve Bayes | TF-IDF |
| Deep Learning | CNN, BiLSTM, CNN-BiLSTM | GloVe, fastText |
| Transformer ( Contextual ) | BanglaBERT, Multilingual BERT, XLM-RoBERTa | BERT Embeddings |

### 🔹 2. Cleaning Approaches
- **Traditional:** punctuation/symbol removal, stop-word removal, stemming, HTML/URL deletion  
- **Context-Aware (CADC):** spelling correction → tagging HTML/URLs → preserve punctuation → retain emoji → TF-IDF word filtering

### 🔹 3. Training Details
- Transformer epochs = 10; Deep Learning epochs = 30  
- Batch size = 8; learning rate = 2e-5 with cyclic LR scheduling  
- Trained on Kaggle P100 GPU  
- Metrics used: Accuracy & Weighted F1-Score

---

## ⚙️ CADC Pipeline Algorithm (Pseudocode)

```python
def CADC(text_data):
    cleaned_text_data = []
    for text in text_data:
        text = spell_correct(text)
        text = replace_html_and_urls_with_tags(text)
        text = preserve_punctuation(text)
        text = preserve_emojis(text)
        text = remove_less_important_words_TFIDF(text)
        cleaned_text_data.append(text)
    return cleaned_text_data
```

This pipeline preserves sentence-level context while eliminating noise and is fully extensible to other morphologically rich languages.

---

## 📊 Datasets Used

| Dataset | Labels | Classes | Domains | Records |
|:---------|:-------:|:--------:|:----------|:--------:|
| BEmoC | 6 | Emotion (Anger, Fear, Joy, etc.) | Bengali literary texts | 7 000 |
| SentNoB | 3 | Positive/Negative/Neutral | Social media comments | 14 142 |
| UBMEC | 6 | Emotion | Reviews (software, politics, sports etc.) | 13 436 |
| EmoNoBa | 6 | Emotion | Social media (12 domains) | 22 698 |

---

## 🧪 Results Summary

| Model Type | Traditional Cleaning | Context-Aware Cleaning | Improvement |
|:------------|:--------------------:|:----------------------:|:------------:|
| Machine Learning | No gain | + 1 – 2 % | ✔️ |
| Deep Learning | Slight drop | + 2 – 3 % | ✔️ |
| Transformers | Baseline | + 4 % avg. | 🔥 |

Context-aware data cleaning consistently improved BanglaBERT and XLM-RoBERTa accuracy by ≈ 4 % across datasets.

---

## 🧰 Repository Structure

```
context-aware-cleaning/
├── data/                 # Bengali datasets (BEmoC, SentNoB, UBMEC, EmoNoBa)
├── src/
│   ├── cadc_pipeline.py  # Main context-aware cleaning algorithm
│   ├── traditional_cleaning.py
│   ├── model_training.py
│   ├── evaluate.py
│   └── utils/
├── notebooks/
│   ├── experiments.ipynb
│   └── visualization.ipynb
├── results/
│   ├── accuracy_tables.csv
│   └── figures/
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### 1. Clone Repository
```bash
git clone https://github.com/<username>/context-aware-cleaning.git
cd context-aware-cleaning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Pipeline
```bash
python src/cadc_pipeline.py --dataset data/SentNoB.csv
```

### 4. Train and Evaluate
```bash
python src/model_training.py --model BanglaBERT --cleaning context-aware
```

---

## 📈 Evaluation Metrics

- **Accuracy** = (TP + TN) / Total Instances  
- **Weighted F1-Score** balances Precision and Recall for imbalanced datasets.

---

## 👩‍💻 Code Contributors

**Moshiur Rahman Faisal**  
**Shahriyar Zaman Ridoy**  
**Jannat Sultana**  
**Zinnat Fowzia Ria**  
**Md Hasibur Rahman**  
**Mohammed Arif Uddin**

---

## 📚 Citation

If you use this work, please cite:

```bibtex
@article{rahman2025contextaware,
  title={Context-Aware Data Cleaning: Optimizing Bengali Text for Contextual Text Classification},
  author={Moshiur Rahman Faisal and Abdur Rahman Fahad and Shahriyar Zaman Ridoy and Jannat Sultana and Zinnat Fowzia Ria and Md Hasibur Rahman and Mohammed Arif Uddin and Rashedur M. Rahman},
  journal={SN Computer Science},
  volume={6},
  pages={422},
  year={2025},
  publisher={Springer Nature}
}
```

---

## 🪴 License

© 2025 The Authors — Released under an exclusive license to Springer Nature Singapore Pte Ltd.  
For academic research use only.

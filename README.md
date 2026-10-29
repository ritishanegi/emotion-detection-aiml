# emotion-detection-aiml
# 🧠 Emotion Detection & Mental Health Classification using AI/ML

An AI/ML project that explores Reddit text data to identify linguistic patterns associated with mental health-related communities using machine learning, natural language processing (NLP), text classification, and unsupervised learning techniques.

## 📌 Project Overview

Social media platforms contain large amounts of user-generated text that can be analyzed to understand language patterns, emotions, and mental health-related discussions. This project uses Reddit data to explore text-based classification and clustering techniques for mental health-related content.

The repository includes data processing scripts, feature extraction pipelines, classification experiments, unsupervised clustering, Jupyter notebooks, and evaluation reports.

**Note:** This project is intended for educational and research purposes. Its predictions do not constitute a medical or psychological diagnosis.

## 🎯 Objectives

- Analyze Reddit text using natural language processing.
- Explore linguistic patterns in mental health-related discussions.
- Develop machine learning pipelines for text classification.
- Apply unsupervised learning to discover patterns and clusters in text data.
- Evaluate model performance using classification metrics.
- Compare model performance across different datasets and time periods.

## ⚙️ Technologies Used

- **Python** — Core programming language
- **Pandas & NumPy** — Data processing and numerical computation
- **Scikit-learn** — Machine learning and model evaluation
- **Jupyter Notebook** — Interactive experimentation
- **NLP & Text Processing** — Text cleaning and feature extraction
- **Clustering & Dimensionality Reduction** — Unsupervised pattern analysis
- **Matplotlib & Seaborn** — Data visualization, where used

## 🔍 Project Workflow

1. **Data Collection:** Work with Reddit mental health-related text datasets.
2. **Data Preprocessing:** Clean and prepare text data for analysis.
3. **Feature Extraction:** Transform text into machine-learning-compatible features.
4. **Exploratory Analysis:** Examine patterns and characteristics in the dataset.
5. **Model Training:** Apply classification algorithms to distinguish target-condition samples from control samples.
6. **Unsupervised Learning:** Explore clustering and topic-related patterns.
7. **Evaluation:** Analyze classification metrics, confusion matrices, and generalization across datasets.
8. **Results & Interpretation:** Review findings and limitations.

## 📂 Repository Structure

```text
emotion-detection-aiml/
│
├── catpro/                              # Project module
├── input/
│   └── reddit_mental_health_dataset/    # Dataset directory
│
├── Reddit_Mental_Health_Master_Pipeline.ipynb
├── Unsupervised_Clustering_Pipeline.ipynb
├── reddit_cluster.ipynb
├── reddit_data_extraction.ipynb
├── reddit_descriptive.ipynb
├── reddit_feature_extraction.ipynb
├── reddit_lda_pipeline.ipynb
│
├── classification_results.py
├── confusion_matrix.py
├── config.py
├── config_parameters.py
├── create_master_notebook.py
├── generate_results_md.py
├── generate_sample_data.py
├── load_reddit.py
├── reddit_cluster.py
├── run.py
├── run_all.py
├── results_summary.md
├── requirements.txt
├── requirements_fixed.txt
├── .gitignore
└── LICENSE
```

*The structure above summarizes the principal files visible in the repository; additional files and folders may be present.*

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- pip
- Git
- Jupyter Notebook or JupyterLab

### 1. Clone the repository

```bash
git clone https://github.com/ritishanegi/emotion-detection-aiml.git
```

### 2. Navigate to the project directory

```bash
cd emotion-detection-aiml
```

### 3. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

If dependency installation fails, check `requirements_fixed.txt` and the project's Python compatibility requirements.

### 5. Explore the notebooks

```bash
jupyter notebook
```

Open the notebooks to explore data preprocessing, feature extraction, classification, clustering, and evaluation.

To inspect the available script-based entry points, review `run.py` and `run_all.py` before executing them.

## 📊 Results & Evaluation

The project includes a results report in `results_summary.md` and scripts for generating classification results and confusion matrices.

The documented experiments include:

- In-domain classification evaluation
- Cross-period evaluation using pandemic-era data
- Cross-domain evaluation using COVID-19 support data
- Precision, recall, F1-score, and accuracy comparisons
- Analysis of learned text features

Performance varies by target category and evaluation dataset. Refer to [`results_summary.md`](results_summary.md) for detailed experimental results.

## 🔮 Future Scope

- Explore transformer-based language models for text classification.
- Improve generalization across datasets and time periods.
- Investigate emotion classification alongside mental health-related text analysis.
- Enhance data quality, feature engineering, and model evaluation.
- Develop an interactive dashboard for exploring research results.
- Strengthen privacy protection and responsible use of social media data.

## ⚠️ Ethical Considerations

- Social media text may contain sensitive personal information.
- Dataset usage must respect applicable platform policies and privacy requirements.
- Model predictions may be affected by bias, context, sarcasm, and differences in writing style.
- Classification results must not be treated as clinical diagnoses or used as a substitute for professional support.

## 🙏 Inspiration & Acknowledgements

This project was inspired by an existing project in the field of AI/ML and mental health-related text analysis. We used the original work as a reference and made our own modifications, improvements, and adaptations to suit our project objectives.

We would like to acknowledge the original creator for providing inspiration and a foundation for our work.

**Original Project:** https://github.com/danielmlow/reddit

## 🛠️ Our Contributions

Building upon the inspiration from the original work, we made changes to develop our own implementation. These changes may include modifications to the data processing pipeline, model implementation, experiments, evaluation methods, and project structure.

Our goal was to understand the existing approach, adapt it, and extend it through our own implementation and experimentation.

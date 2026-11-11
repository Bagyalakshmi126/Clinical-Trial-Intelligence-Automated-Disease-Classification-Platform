# Clinical-Trial-Intelligence-Automated-Disease-Classification-Platform
A production-ready Streamlit application leveraging TF-IDF vectorization and an optimized Random Forest classifier to automate clinical trial disease categorization and exploratory data analysis.

# 🏥 Clinical Trial Intelligence & Disease Classification Platform

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
  <img src="https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white" alt="Pandas">
  <img src="https://img.shields.io/badge/Matplotlib-%23#ffffff.svg?style=for-the-badge&logo=matplotlib&logoColor=white" alt="Matplotlib">
  <img src="https://img.shields.io/badge/Machine%2B_Learning-NLP-success?style=for-the-badge" alt="ML & NLP">
</p>

An end-to-end Machine Learning and Natural Language Processing (NLP) web application designed to automate clinical trial text classification, analyze research trends, and visualize medical datasets through an interactive Streamlit dashboard.

---

## ✨ Key Features & Capabilities
* **📝 Real-Time Text Classification:** Enter or paste any clinical trial brief summary to instantly predict its primary disease category using an optimized Random Forest classifier paired with TF-IDF vectorization.
* **📊 Exploratory Data Analysis (EDA):** Built-in interactive metrics tracking total clinical trial volume, unique disease classes, and overall model accuracy.
* **☁️ Medical Keyword Word Cloud:** Automatically extracts and visualizes prominent terminology from the trial text corpus.
* **📈 Transparent Evaluation Metrics:** Comprehensive class-wise breakdown of Precision, Recall, and F1-Scores across all medical categories.

---

## 🗄️ Dataset & Data Structure (`cleaned_data.csv`)

The platform processes a cleaned corpus of clinical trials (`60,337` records). Below is the breakdown of the core data schema utilized across the application and EDA pipelines:

| Column Name | Data Type | Description |
| :--- | :--- | :--- |
| `nct_id` | String | Unique ClinicalTrials.gov registry identifier (e.g., NCT01234567). |
| `title` | String | Main title assigned to the clinical study. |
| `official_title` | String | Full official study protocol title. |
| `brief_summary` | Text | Summary of the study purpose, design, and scope (used for TF-IDF NLP vectorization). |
| `conditions` | String / List | Specific medical conditions or health issues targeted by the trial. |
| `interventions` | String / List | Medical treatments, drugs, devices, or procedures evaluated. |
| `overall_status` | Category | Current recruitment or completion status (e.g., Completed, Recruiting). |
| `study_type` | Category | Classification of study design (e.g., Interventional, Observational). |
| `phase` | Category | Trial phase classification (Phase 1, Phase 2, Phase 3, Phase 4, etc.). |
| `sex` | Category | Target participant gender eligibility (All, Male, Female). |
| `minimum_age` / `maximum_age` | String | Age eligibility limits for trial participants. |
| `healthy_volunteers` | Boolean | Indicates whether healthy volunteers are accepted. |
| `eligibility_criteria` | Text | Detailed inclusion and exclusion criteria guidelines. |
| `clinicaltrials_url` | URL | Direct link to the official trial registry record. |
| `source_condition_query` | Target Label | Primary target class label used for supervised multi-class classification. |

---

## 📊 Visualization & EDA Insights

The dashboard incorporates pure **Matplotlib** and **WordCloud** visual modules to deliver seamless performance without heavy external UI dependencies:

1. **Disease Category Distribution (Bar Chart):** Visualizes the relative volume of clinical trials across each primary condition target.
2. **Summary Word Length Distribution (Histogram):** Evaluates token length frequencies and text length characteristics across study briefs.
3. **Medical Keyword Word Cloud:** Highlights high-frequency clinical terms and disease markers extracted from the corpus.

---

## 🛠️ Tech Stack & Architecture

* **Frontend Dashboard:** [Streamlit](https://streamlit.io/) (Multi-tab reactive UI layout)
* **Machine Learning Engine:** Scikit-Learn (`RandomForestClassifier`, `TfidfVectorizer`, `LabelEncoder`)
* **Data Processing:** Pandas, NumPy
* **Visualization & Artifacts:** Matplotlib, WordCloud, Joblib

---

## 📂 Project Structure

```text
clinical-trial-platform/
│
├── app.py                      # Main Streamlit web application script
├── cleaned_data.csv            # Processed master dataset (60,337 records)
├── clinical_trial_model.pkl    # Trained Random Forest classifier model artifact
├── tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer artifact
├── label_encoder.pkl           # Saved Label Encoder for category mapping
├── requirements.txt            # Python dependencies and package versions
└── README.md                   # Project documentation

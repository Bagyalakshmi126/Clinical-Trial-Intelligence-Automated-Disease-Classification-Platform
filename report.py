import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Page configuration
st.set_page_config(
    page_title="Clinical Trial Intelligence Platform",
    page_icon="🏥",
    layout="wide"
)

# Load saved model artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('clinical_trial_model.pkl')
    vectorizer = joblib.load('tfidf_vectorizer.pkl')
    encoder = joblib.load('label_encoder.pkl')
    return model, vectorizer, encoder

model, vectorizer, label_encoder = load_artifacts()

# Load dataset for EDA tab
@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_data.csv')
    except FileNotFoundError:
        return None

df_eda = load_data()

# App Title & Overview
st.title("🏥 Clinical Trial Intelligence & Disease Classification Platform")
st.markdown("""
An advanced analytics and machine learning dashboard designed to support medical research, clinical trial text analysis, 
and automated disease categorization.
""")

# Sidebar Navigation
tab = st.sidebar.selectbox("Navigation Menu", ["Predict Disease Category", "EDA & Visual Insights", "Model Performance"])

# ==========================================
# TAB 1: PREDICTION INTERFACE
# ==========================================
if tab == "Predict Disease Category":
    st.header("📝 Clinical Trial Summary Classification")
    st.write("Enter or paste a clinical trial brief summary below to predict its primary disease category in real time.")
    
    with st.expander("💡 Click to view an example summary"):
        st.code("This randomized clinical trial evaluates the efficacy and safety of targeted therapies for patients diagnosed with advanced breast cancer...")

    user_input = st.text_area("Clinical Trial Brief Summary:", height=150)
    
    if st.button("Classify Trial", type="primary"):
        if user_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            input_vector = vectorizer.transform([user_input])
            prediction_encoded = model.predict(input_vector)[0]
            prediction_proba = model.predict_proba(input_vector)[0]
            predicted_label = label_encoder.inverse_transform([prediction_encoded])[0]
            
            st.success(f"### Predicted Disease Category: `{predicted_label.upper()}`")
            
            # Confidence score breakdown using Matplotlib horizontal bar chart
            st.subheader("Confidence Probability Breakdown")
            proba_df = pd.DataFrame({
                'Category': label_encoder.classes_,
                'Probability': prediction_proba
            }).sort_values(by='Probability', ascending=True)
            
            fig_proba, ax_proba = plt.subplots(figsize=(8, 4))
            ax_proba.barh(proba_df['Category'], proba_df['Probability'], color='steelblue')
            ax_proba.set_xlabel('Probability Score')
            ax_proba.set_title('Model Confidence Across All Disease Categories')
            plt.tight_layout()
            st.pyplot(fig_proba)

# ==========================================
# TAB 2: EDA & VISUAL INSIGHTS
# ==========================================
elif tab == "EDA & Visual Insights":
    st.header("📊 Exploratory Data Analysis & Medical Insights")
    
    if df_eda is not None:
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Clinical Trials", f"{len(df_eda):,}")
        col2.metric("Disease Categories", f"{df_eda['source_condition_query'].nunique()}")
        col3.metric("Model Accuracy", "93.68%")
        
        st.divider()
        
        # Chart 1: Disease Category Distribution (Matplotlib Bar Chart)
        st.subheader("Disease Category Distribution")
        cat_counts = df_eda['source_condition_query'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        
        fig_cat, ax_cat = plt.subplots(figsize=(10, 5))
        ax_cat.bar(cat_counts['Category'], cat_counts['Count'], color='teal')
        ax_cat.set_xticklabels(cat_counts['Category'], rotation=45, ha='right')
        ax_cat.set_ylabel('Count')
        ax_cat.set_title('Distribution of Clinical Trials Across Disease Categories')
        plt.tight_layout()
        st.pyplot(fig_cat)
        
        # Chart 2: Summary Word Count Distribution (Matplotlib Histogram)
        st.subheader("Clinical Trial Summary Length Distribution")
        df_eda['word_count'] = df_eda['brief_summary'].apply(lambda x: len(str(x).split()))
        
        fig_hist, ax_hist = plt.subplots(figsize=(10, 4))
        ax_hist.hist(df_eda['word_count'], bins=50, color='darkcyan', edgecolor='white')
        ax_hist.set_xlabel('Word Count')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Distribution of Brief Summary Word Counts')
        plt.tight_layout()
        st.pyplot(fig_hist)
        
        st.divider()

 # Chart 3: Word Cloud for Medical Keywords
        st.subheader("☁️ Frequently Occurring Medical Keywords")
        st.markdown("This word cloud visualizes the most prominent terms extracted from the clinical trial text corpus.")
        
        # Automatically create 'cleaned_summary' from 'brief_summary' if it exists
        if 'cleaned_summary' not in df_eda.columns and 'brief_summary' in df_eda.columns:
            df_eda['cleaned_summary'] = df_eda['brief_summary']

        if 'cleaned_summary' in df_eda.columns:
            @st.cache_data
            def generate_wordcloud(data):
                sample_text = " ".join(data['cleaned_summary'].dropna().sample(n=min(5000, len(data)), random_state=42).astype(str))
                wc = WordCloud(
                    width=900, 
                    height=450, 
                    background_color='white', 
                    colormap='viridis',
                    max_words=150,
                    random_state=42
                ).generate(sample_text)
                return wc

            wordcloud = generate_wordcloud(df_eda)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            plt.tight_layout(pad=0)
            
            st.pyplot(fig)
        else:
            st.info("Ensure your dataframe contains the text column to generate the word cloud.")       
       
            
           

# ==========================================
# TAB 3: MODEL PERFORMANCE
# ==========================================
elif tab == "Model Performance":
    st.header("📈 Model Evaluation & Research Metrics")
    st.markdown("""
    - **Selected Algorithm:** Random Forest Classifier (optimized with TF-IDF unigrams & bigrams).
    - **Overall Test Accuracy:** **93.68%**
    - **Robustness:** High macro-averaged and weighted F1-scores (0.92 - 0.94) ensuring reliable classification performance across both major and minor disease classes.
    """)
    
    metrics_data = {
        "Disease Category": label_encoder.classes_,
        "Precision": [0.88, 0.94, 0.94, 0.97, 0.98, 0.94, 0.99, 0.95],
        "Recall": [0.96, 0.97, 0.88, 0.93, 0.93, 0.83, 0.80, 0.96],
        "F1-Score": [0.92, 0.95, 0.91, 0.95, 0.95, 0.88, 0.89, 0.95]
    }
    st.subheader("Detailed Class-wise Performance Metrics")
    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)
import streamlit as st
# --- HIDE STREAMLIT STYLE ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

from textblob import TextBlob
import pandas as pd

st.title("Sentiment Analysis App")

tab1, tab2 = st.tabs(["Text Analysis", "Dataset Analysis"])

with tab1:
    st.header("Single Sentence Analysis")
    text = st.text_input("Enter a Sentence: ")

    if st.button("Analyze"):
        if text:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            if polarity > 0:
                sentiment = "Positive 😁"
                st.success(f"Sentiment: {sentiment} (Polarity: {polarity})")
            elif polarity < 0:
                sentiment = "Negative 😒"
                st.error(f"Sentiment: {sentiment} (Polarity: {polarity})")
            else:
                sentiment = "Neutral 😐"
                st.info(f"Sentiment: {sentiment} (Polarity: {polarity})")
        else:
            st.warning("Please enter some text to analyze.")

with tab2:
    st.header("Dataset Analysis")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("Data Preview:")
        st.dataframe(df.head())
        
        text_column = st.selectbox("Select the column to analyze:", df.columns)
        
        if st.button("Analyze Dataset"):
            st.write("Analyzing...")
            
            def get_sentiment(text):
                blob = TextBlob(str(text))
                return blob.sentiment.polarity
            
            df['Polarity'] = df[text_column].apply(get_sentiment)
            
            def get_sentiment_label(score):
                if score > 0:
                    return "Positive"
                elif score < 0:
                    return "Negative"
                else:
                    return "Neutral"
            
            df['Sentiment'] = df['Polarity'].apply(get_sentiment_label)
            
            st.write("Analysis Complete!")
            st.dataframe(df)
            
            st.download_button(
                label="Download Results",
                data=df.to_csv(index=False).encode('utf-8'),
                file_name='sentiment_analysis_results.csv',
                mime='text/csv',

            )

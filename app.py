import streamlit as st
from transformers import pipeline


# CSS for background image and text styling
page_bg_css = '''
<style>
.stApp {
    background-color: #f0f0f0;
    color: black;
}
.stTextInput>div>div>input {
    background-color: white !important;
    color: black !important;
}
.stButton>button {
    background-color: #4CAF50;
    color: white;
    font-weight: bold;
}
</style>
'''

st.markdown(page_bg_css, unsafe_allow_html=True)

# st.markdown(page_bg_img, unsafe_allow_html=True)

# Load the sentiment analysis pipeline (this will download the model the first time)
@st.cache_resource(show_spinner=False)
def load_model():
    return pipeline('sentiment-analysis')

classifier = load_model()

st.title("SentimentSense")
st.write("Enter text below to analyze its sentiment:")

# Input text area
user_input = st.text_area("Your text here", "")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text to analyze.")
    else:
        result = classifier(user_input)[0]
        label = result['label']
        score = result['score']

        if label == 'POSITIVE':
            st.success(f"Sentiment: {label} ({score:.2f})")
        else:
            st.error(f"Sentiment: {label} ({score:.2f})")

import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import json
import time

# --- Setup ---
st.set_page_config(page_title="Gemini URL Pro (v2)", layout="wide")
st.title("🚀 Unified URL Publisher Classifier")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    client_site = st.text_input("Client Website", placeholder="e.g. www.tech-gear.com")
    batch_size = st.slider("Batch Size", 20, 100, 50)

# Initialize Client
client = None
if api_key:
    client = genai.Client(api_key=api_key)

# --- Category Definition ---
cat_text = st.text_area("Allowed Categories:", value="Tech News\nLifestyle\nE-commerce\nForum")
categories = [c.strip() for c in cat_text.split('\n') if c.strip()] + ["None"]

# --- Processing Function ---
def process_batch(url_batch, allowed_cats, client_url):
    # The new SDK handles schema validation via the GenerateConfig
    # We define a Pydantic-like structure for the response
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "category": {"type": "string", "enum": allowed_cats},
                    "is_relevant": {"type": "boolean"}
                },
                "required": ["url", "category", "is_relevant"]
            }
        }
    )

    prompt = f"Categorize these URLs. Categories: {allowed_cats}."
    if client_url:
        prompt += f" Check relevance to {client_url}."
    prompt += "\nURLs:\n" + "\n".join(url_batch)

    try:
        # Note: Using Gemini 2.0 Flash for best performance/speed in 2026
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=prompt,
            config=config
        )
        return response.parsed # New SDK can parse JSON directly if schema is provided
    except Exception as e:
        st.error(f"Batch failed: {e}")
        return []

# --- Main Logic ---
uploaded_file = st.file_uploader("Upload CSV", type="csv")
if uploaded_file and client:
    df = pd.read_csv(uploaded_file)
    url_col = st.selectbox("Select URL Column", df.columns)
    
    if st.button("Start Categorization"):
        urls = df[url_col].tolist()
        final_results = []
        progress = st.progress(0)
        
        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            res = process_batch(batch, categories, client_site)
            final_results.extend(res)
            progress.progress(min((i + batch_size) / len(urls), 1.0))
            
        st.session_state.final_df = pd.DataFrame(final_results)
        st.success("Complete!")
        st.dataframe(st.session_state.final_df)

import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import advertools as adv
import os

# --- Page Setup ---
st.set_page_config(page_title="Gemini URL Pro", layout="wide")
st.title("Publisher Classifier")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    client_site = st.text_input("Client URL", placeholder="https://www.example.com")
    batch_size = st.slider("URLs per Batch", 10, 100, 50)

# --- Advertools Client Site Crawler ---
def get_client_context(url):
    """Crawl homepage + 1 depth for context."""
    output_file = "client_info.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)

    with st.spinner(f"Extracting context from {url}..."):
        # DEPTH_LIMIT: 0 is homepage, 1 is one level deep
        adv.crawl(url, output_file, follow_links=True,
                  custom_settings={'DEPTH_LIMIT': 1, 'CLOSESPIDER_PAGECOUNT': 15})

    if os.path.exists(output_file):
        crawl_df = pd.read_json(output_file, lines=True)
        # Use available columns for text context
        cols = [c for c in ['title', 'h1', 'body_text'] if c in crawl_df.columns]
        combined = " ".join(crawl_df[cols].fillna('').astype(str).values.flatten())
        return combined[:4000] # Token safety
    return "No site context found."

# --- Gemini Processing ---
def classify_urls(url_batch, allowed_cats, context):
    client = genai.Client(api_key=api_key)

    # Strictly boolean and Enum-constrained schema
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

    prompt = f"""
    CLIENT CONTEXT: {context}

    TASK:
    1. Categorize these URLs using ONLY these categories: {allowed_cats}.
    2. Determine if each URL is topically relevant to the Client Context above (True/False).

    URLs:
    """ + "\n".join(url_batch)

    response = client.models.generate_content(
        model='gemini-3-flash-preview', # Fastest model for classification
        contents=prompt,
        config=config
    )
    return response.parsed

# --- UI and Execution ---
cat_input = st.text_area("Categories (One per line):", value="News\nBlog\nE-commerce\nReview Site")
categories = [c.strip() for c in cat_input.split('\n') if c.strip()] + ["None"]

uploaded_file = st.file_uploader("Upload CSV of URLs", type="csv")

if uploaded_file and api_key and client_site:
    df = pd.read_csv(uploaded_file)
    url_col = st.selectbox("Column with URLs", df.columns)

    if st.button("Run"):
        # Get context once
        site_context = get_client_context(client_site)

        urls = df[url_col].tolist()
        results = []
        progress = st.progress(0)

        for i in range(0, len(urls), batch_size):
            batch = urls[i:i + batch_size]
            batch_data = classify_urls(batch, categories, site_context)
            results.extend(batch_data)
            progress.progress(min((i + batch_size) / len(urls), 1.0))

        # Display and Download
        res_df = pd.DataFrame(results)
        st.subheader("Results")
        st.dataframe(res_df, use_container_width=True)

        csv = res_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Results", data=csv, file_name="categorized_urls.csv")

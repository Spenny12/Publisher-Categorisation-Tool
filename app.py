import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import advertools as adv
import os
import json

# --- Setup ---
st.set_page_config(page_title="Gemini URL Pro + Crawler", layout="wide")
st.title("Context-Aware Publisher Classifier")

with st.sidebar:
    api_key = st.text_input("Gemini API Key", type="password")
    client_url = st.text_input("Client Website", placeholder="https://www.example.com")
    batch_size = st.slider("Batch Size", 20, 100, 50)

# --- Advertools Crawling Function ---
def get_client_context(url):
    """Crawls homepage + 1 depth and returns a text summary."""
    output_file = "client_crawl.jsonl"
    if os.path.exists(output_file):
        os.remove(output_file)

    with st.spinner(f"Crawling {url} (Depth 1)..."):
        # DEPTH_LIMIT: 0 is homepage, 1 is one level deep
        adv.crawl(url, output_file, follow_links=True,
                  custom_settings={'DEPTH_LIMIT': 1, 'CLOSESPIDER_PAGECOUNT': 20})

    if os.path.exists(output_file):
        crawl_df = pd.read_json(output_file, lines=True)
        # Extract meaningful text from titles, H1s, and body
        text_cols = ['title', 'h1', 'body_text']
        available_cols = [c for c in text_cols if c in crawl_df.columns]

        combined_text = " ".join(crawl_df[available_cols].fillna('').astype(str).values.flatten())
        return combined_text[:5000] # Cap context to save tokens
    return "No context found."

# --- Gemini Processing ---
def classify_with_context(url_batch, allowed_cats, client_context):
    client = genai.Client(api_key=api_key)

    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                    "category": {"type": "string", "enum": allowed_cats},
                    "is_relevant": {"type": "boolean"},
                    "reasoning": {"type": "string"}
                },
                "required": ["url", "category", "is_relevant", "reasoning"]
            }
        }
    )

    prompt = f"""
    CONTEXT OF CLIENT WEBSITE:
    {client_context}

    TASK:
    1. Categorize the list of URLs into: {allowed_cats}.
    2. Determine if each URL is topically relevant to the Client Context provided above.
    3. Provide a brief reasoning for the relevance.

    URLs:
    """ + "\n".join(url_batch)

    response = client.models.generate_content(
        model='gemini-3-flash-preview',
        contents=prompt,
        config=config
    )
    return response.parsed

# --- Streamlit UI ---
cat_text = st.text_area("Categories:", value="News\nBlog\nReview Site\nE-commerce")
categories = [c.strip() for c in cat_text.split('\n') if c.strip()] + ["None"]

uploaded_file = st.file_uploader("Upload Publisher CSV", type="csv")

if st.button("Run") and uploaded_file and api_key:
    # Step 1: Get Context
    context = get_client_context(client_url)
    st.info("Client context successfully extracted.")

    # Step 2: Process List
    df = pd.read_csv(uploaded_file)
    urls = df.iloc[:, 0].tolist() # Assumes URLs are in the first column

    all_results = []
    progress = st.progress(0)

    for i in range(0, len(urls), batch_size):
        batch = urls[i:i + batch_size]
        res = classify_with_context(batch, categories, context)
        all_results.extend(res)
        progress.progress(min((i + batch_size) / len(urls), 1.0))

    res_df = pd.DataFrame(all_results)
    st.write(res_df)
    st.download_button("Download CSV", res_df.to_csv(index=False), "results.csv")

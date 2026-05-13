import streamlit as st
import pandas as pd
import tempfile
import subprocess
import sys
import os
import json
import urllib.parse
import google.generativeai as genai

def ensure_scheme(url):
    """Ensures the URL has an http or https scheme."""
    url = str(url).strip()
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url

st.set_page_config(page_title="Publisher Relevance and Categorisation Tool", layout="wide")

def run_isolated_crawl(urls, output_filepath):
    """
    Runs advertools in a subprocess to prevent Twisted reactor restart errors
    common in Streamlit applications. DEPTH_LIMIT is set to 1 to crawl the
    provided URL and one link down.
    """
    script_content = f"""
import advertools as adv
urls = {urls}
settings = {{
    'DEPTH_LIMIT': 1,
    'CLOSESPIDER_PAGECOUNT': 50,
    'LOG_LEVEL': 'ERROR',
    'USER_AGENT': 'CategorisationBot/1.0'
}}
adv.crawl(urls, '{output_filepath}', custom_settings=settings)
    """
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as script_file:
        script_file.write(script_content)
        script_path = script_file.name

    subprocess.run([sys.executable, script_path], check=True)
    os.remove(script_path)

def get_domain(url):
    """Extracts the network location (domain) from a URL."""
    parsed = urllib.parse.urlparse(str(url))
    return parsed.netloc if parsed.netloc else str(url)

def extract_domain_summaries(jl_filepath, original_urls):
    """Parses the JSONlines output from advertools and concatenates metadata for the LLM."""
    if not os.path.exists(jl_filepath) or os.path.getsize(jl_filepath) == 0:
        return {url: "No data retrieved." for url in original_urls}

    df = pd.read_json(jl_filepath, lines=True)
    summaries = {}

    for col in ['url', 'title', 'meta_desc']:
        if col not in df.columns:
            df[col] = ''

    df['title'] = df['title'].fillna('')
    df['meta_desc'] = df['meta_desc'].fillna('')
    df['domain'] = df['url'].apply(get_domain)

    for url in original_urls:
        target_domain = get_domain(url)
        domain_df = df[df['domain'] == target_domain]

        if domain_df.empty:
            summaries[url] = "No data retrieved."
            continue

        titles = domain_df['title'].replace('', pd.NA).dropna().unique()
        metas = domain_df['meta_desc'].replace('', pd.NA).dropna().unique()

        summary_text = "Page Titles:\n" + "\n".join(titles) + "\n\n"
        summary_text += "Meta Descriptions:\n" + "\n".join(metas)

        # Truncate to avoid exceeding token limits
        summaries[url] = summary_text[:4000]

    return summaries

def analyse_with_gemini(client_summary, publisher_summary, categories, api_key):
    """Calls Gemini 3.1 Flash Lite to determine relevance and categorise."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-3.1-flash-lite')

    prompt = f"""
    You are an expert data categorisation assistant. Analyse the provided website summaries.

    Client Website Summary:
    {client_summary}

    Publisher Website Summary:
    {publisher_summary}

    Available Categories: {', '.join(categories)}

    Task:
    1. Determine if the publisher website is topically relevant to the client website.
    2. Assign the most appropriate category from the 'Available Categories' list to the publisher. If none are suitable, assign "Other".

    Return ONLY a valid JSON object in this exact format, with no markdown formatting or backticks:
    {{
        "relevant": "Yes" or "No",
        "category": "Assigned Category"
    }}
    """
    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()

        # Clean formatting if the model returns markdown backticks
        if text_response.startswith('```json'):
            text_response = text_response[7:-3]
        elif text_response.startswith('```'):
            text_response = text_response[3:-3]

        data = json.loads(text_response.strip())
        return data
    except Exception as e:
        return {"relevant": "Error", "category": "Error"}

# --- User Interface ---

st.title("Publisher Relevance and Categorisation Tool")
st.write("Upload a CSV of publisher URLs and provide a client website. The programme will crawl one level deep to determine topical relevance and assign categories.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    api_key = st.text_input("Gemini API Key", type="password")
    client_url = st.text_input("Client Website URL", placeholder="https://www.example-client.com")
    uploaded_file = st.file_uploader("Upload Publisher URLs (CSV)", type=['csv'])

with col2:
    default_categories = "National News\nRegional News\nBlog\nE-commerce\nReview Site"
    categories_input = st.text_area("Categories (one per line)", value=default_categories, height=200)

if st.button("Analyse Publishers"):
    if not api_key or not client_url or not uploaded_file:
        st.write("**Error:** Please provide the API key, client URL, and upload a CSV file.")
    else:
        categories = [c.strip() for c in categories_input.split('\n') if c.strip()]
        if "Other" not in categories:
            categories.append("Other")

        df_uploaded = pd.read_csv(uploaded_file)

        # Auto-detect URL column
        url_col = None
        for col in df_uploaded.columns:
            if 'url' in col.lower() or 'link' in col.lower():
                url_col = col
                break
        if not url_col:
            url_col = df_uploaded.columns[0]

        # APPLIED FIX: Sanitise the publisher URLs
        raw_urls = df_uploaded[url_col].dropna().astype(str).tolist()
        publisher_urls = [ensure_scheme(url) for url in raw_urls]

        # APPLIED FIX: Sanitise the client URL just in case
        client_url_clean = ensure_scheme(client_url)

        st.write("**Status:** Commencing crawl of client website...")

        with tempfile.TemporaryDirectory() as tmpdirname:
            client_jl = os.path.join(tmpdirname, 'client.jl')
            pub_jl = os.path.join(tmpdirname, 'publishers.jl')

            # Crawl client (using the cleaned URL)
            run_isolated_crawl([client_url_clean], client_jl)
            client_summary = extract_domain_summaries(client_jl, [client_url_clean]).get(client_url_clean, "")

            st.write("**Status:** Commencing crawl of publisher websites (this may take a moment)...")

            # Crawl publishers
            run_isolated_crawl(publisher_urls, pub_jl)
            pub_summaries = extract_domain_summaries(pub_jl, publisher_urls)

            st.write("**Status:** Crawls complete. Analysing behaviour and relevance via Gemini...")

            results = []
            progress_bar = st.progress(0)
            total_pubs = len(publisher_urls)

            for idx, pub_url in enumerate(publisher_urls):
                pub_sum = pub_summaries.get(pub_url, "")
                res = analyse_with_gemini(client_summary, pub_sum, categories, api_key)

                results.append({
                    "Publisher URL": pub_url,
                    "Relevant": res.get("relevant", "Error"),
                    "Category": res.get("category", "Error")
                })

                progress_bar.progress((idx + 1) / total_pubs)

            st.write("**Status:** Analysis complete.")

            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)

            csv_output = df_results.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Results as CSV",
                data=csv_output,
                file_name="categorisation_results.csv",
                mime="text/csv",
            )

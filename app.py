import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import urllib.parse
import json
import time
import google.generativeai as genai

st.set_page_config(page_title="Publisher Relevance and Categorisation Tool", layout="wide")

def ensure_scheme(url):
    """Ensures the URL has an http or https scheme."""
    url = str(url).strip()
    if not url.startswith(('http://', 'https://')):
        return 'https://' + url
    return url

def get_page_data(url):
    """Fetches a single page and returns its title, meta description, and parsed HTML."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
    }
    try:
        response = requests.get(url, headers=headers, timeout=(3, 5), allow_redirects=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')

        title = soup.title.string.strip() if soup.title and soup.title.string else ""

        meta_desc = ""
        meta_tag = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_tag and 'content' in meta_tag.attrs:
            meta_desc = meta_tag['content'].strip()

        return {"title": title, "meta_desc": meta_desc, "soup": soup}
    except requests.exceptions.RequestException:
        return None

def extract_internal_links(soup, base_url, limit=5):
    """Extracts a limited number of internal links from the parsed HTML."""
    internal_links = set()
    base_domain = urllib.parse.urlparse(base_url).netloc

    for a_tag in soup.find_all('a', href=True):
        if len(internal_links) >= limit:
            break

        href = a_tag['href']
        full_url = urllib.parse.urljoin(base_url, href)
        parsed_url = urllib.parse.urlparse(full_url)

        # Keep only HTTP/HTTPS, matching domain, and remove anchor tags
        if parsed_url.scheme in ['http', 'https'] and parsed_url.netloc == base_domain:
            clean_url = urllib.parse.urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, '', '', ''))
            # Avoid crawling the homepage again or obvious media files
            if clean_url != base_url and not clean_url.endswith(('.png', '.jpg', '.pdf', '.mp4')):
                internal_links.add(clean_url)

    return list(internal_links)

def crawl_domain_for_summary(start_url, max_internal_pages=5):
    """Crawls the homepage and a few internal links to build a text summary for Gemini."""
    titles = set()
    metas = set()

    # 1. Fetch Homepage
    homepage_data = get_page_data(start_url)

    if not homepage_data:
        return "No data retrieved. Server refused connection or timed out."

    if homepage_data["title"]: titles.add(homepage_data["title"])
    if homepage_data["meta_desc"]: metas.add(homepage_data["meta_desc"])

    # 2. Extract and fetch internal links (1 link down)
    internal_links = extract_internal_links(homepage_data["soup"], start_url, limit=max_internal_pages)

    for link in internal_links:
        page_data = get_page_data(link)
        if page_data:
            if page_data["title"]: titles.add(page_data["title"])
            if page_data["meta_desc"]: metas.add(page_data["meta_desc"])

    # 3. Format the output for the LLM
    summary_text = "Page Titles Found:\n" + "\n".join(titles) + "\n\n"
    summary_text += "Meta Descriptions Found:\n" + "\n".join(metas)

    # Truncate to ensure we do not exceed token limits
    return summary_text[:4000]

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

        if text_response.startswith('```json'):
            text_response = text_response[7:-3]
        elif text_response.startswith('```'):
            text_response = text_response[3:-3]

        data = json.loads(text_response.strip())
        return data
    except Exception:
        return {"relevant": "Error", "category": "Error"}

# --- User Interface ---

st.title("Publisher Relevance and Categorisation Tool")
st.write("Upload a CSV of publisher URLs and provide a client website. The programme will extract data from the homepage and internal links to determine topical relevance.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    api_key = st.text_input("Gemini API Key", type="password")
    client_url = st.text_input("Client Website URL", placeholder="https://www.example-client.com")
    uploaded_file = st.file_uploader("Upload Publisher URLs (CSV)", type=['csv'])

with col2:
    default_categories = "National News\nRegional News\nBlog\nE-commerce\nReview Site"
    categories_input = st.text_area("Categories (one per line)", value=default_categories, height=200)
    pages_to_crawl = st.slider("Internal pages to crawl per domain", min_value=1, max_value=10, value=3, help="Higher numbers provide more context to the AI but make the tool run slower.")

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

        raw_urls = df_uploaded[url_col].dropna().astype(str).tolist()
        publisher_urls = [ensure_scheme(url) for url in raw_urls]
        client_url_clean = ensure_scheme(client_url)

        st.write("**Status:** Crawling client website...")
        client_summary = crawl_domain_for_summary(client_url_clean, max_internal_pages=pages_to_crawl)

        st.write("**Status:** Commencing publisher analysis...")

        results = []
        progress_bar = st.progress(0)

        # APPLIED FIX: Create an empty container for real-time text updates
        status_text = st.empty()

        total_pubs = len(publisher_urls)

        for idx, pub_url in enumerate(publisher_urls):
            # APPLIED FIX: Update the UI so you know exactly what is happening
            status_text.markdown(f"**Processing ({idx + 1}/{total_pubs}):** `{pub_url}`")

            pub_sum = crawl_domain_for_summary(pub_url, max_internal_pages=pages_to_crawl)
            res = analyse_with_gemini(client_summary, pub_sum, categories, api_key)

            results.append({
                "Publisher URL": pub_url,
                "Relevant": res.get("relevant", "Error"),
                "Category": res.get("category", "Error")
            })

            progress_bar.progress((idx + 1) / total_pubs)

            # APPLIED FIX: Pause for 2 seconds to avoid Gemini rate limits
            time.sleep(2)

        status_text.markdown("**Status:** Analysis complete.")

        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)

        csv_output = df_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Results as CSV",
            data=csv_output,
            file_name="categorisation_results.csv",
            mime="text/csv",
        )

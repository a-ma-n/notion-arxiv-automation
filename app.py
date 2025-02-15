import streamlit as st
import arxiv
import requests
import os
import re
import datetime
from keybert import KeyBERT
from notion_client import Client
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve default values from environment
DEFAULT_NOTION_TOKEN = os.getenv("NOTION_TOKEN", "")
DEFAULT_NOTION_PAGE_ID = os.getenv("NOTION_PAGE_ID", "")
DEFAULT_NOTION_DATABASE_ID = os.getenv("NOTION_DATABASE_ID", "")

# Sidebar: Display and allow override for Notion configuration
st.sidebar.title("Environment Setup")
notion_api_key_input = st.sidebar.text_input("Notion API Key", value=DEFAULT_NOTION_TOKEN)
notion_page_id_input = st.sidebar.text_input("Notion Page ID", value=DEFAULT_NOTION_PAGE_ID)
notion_database_id_input = st.sidebar.text_input("Notion Database ID (optional)", value=DEFAULT_NOTION_DATABASE_ID)

# Create the Notion client using the key from the UI
notion = Client(auth=notion_api_key_input)

# Force UTF-8 for stdout/stderr
os.environ["PYTHONIOENCODING"] = "utf-8"

# Initialize KeyBERT
kw_model = KeyBERT()

# Sidebar: Model selection for Ollama
available_models = ["deepseek-r1:1.5b", "llama2:7b", "qwen2.5:latest", "llama3.2:latest"]
selected_model = st.sidebar.selectbox("Select Ollama Model", available_models)
llm = OllamaLLM(model=selected_model)

# Sidebar: Configure summary prompt template
default_summary_prompt = """
Summarize this research paper in a structured way:

1️⃣ Core Contributions (1-2 sentences)
2️⃣ Key Techniques (2-3 sentences)
3️⃣ Results (2-3 sentences)
4️⃣ Implications (1-2 sentences)

Make it {style} in tone.

{text}
""".strip()


summary_prompt_template = st.sidebar.text_area("Summary Prompt Template", value=default_summary_prompt)

# Sidebar: Configure styles for summaries
style_normal_input = st.sidebar.text_input("Enter the style you would like the normal summary to be written in:", value="normal")
style_thrilling_input = st.sidebar.text_input("Enter the style you would like the thrilling summary to be written in:", value="exciting manga-style")

# Sidebar: Configure related terms (comma separated)
related_terms_input = st.sidebar.text_input("Enter related terms (comma separated):", 
                                              value="bci, brain-computer interface, eeg, neural network, deep learning, llm")
user_related_terms = {term.strip().lower() for term in related_terms_input.split(",") if term.strip()}

# Sidebar: Configure bonus multiplier for scoring
bonus_multiplier = st.sidebar.slider("Bonus Multiplier", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
# New checkbox: Enable/Disable bonus multiplier
apply_bonus_multiplier = st.sidebar.checkbox("Apply Bonus Multiplier", value=True)



with st.sidebar.expander("How Bonus Multiplier Works"):
    st.markdown("""
    The bonus multiplier is applied to increase a paper's score based on its rank if it is relevant.
    
    **Calculation:**  
    For a paper, the bonus is calculated as:
    
    \[
    \text{Bonus} = (\text{Total Papers} - \text{Paper Rank}) \times \text{Bonus Multiplier}
    \]
    
    If the paper is considered relevant (its abstract contains one of the defined related terms), the full bonus is applied.
    
    **Note:** If "Apply Bonus Multiplier" is unticked, no bonus is applied.
    """)

# --- Helper Functions ---

def get_available_ollama_models():
    try:
        # Run the command "ollama list" and capture output
        result = subprocess.run("ollama list", shell=True, capture_output=True, text=True)
        lines = result.stdout.strip().splitlines()
        models = []
        # Assume first line is header, so parse subsequent lines
        if len(lines) > 1:
            for line in lines[1:]:
                parts = line.split()
                if parts:
                    models.append(parts[0])
        return models
    except Exception as e:
        st.write("Error fetching available models:", e)
        return []

# Sidebar: Choose sort criterion for ArXiv search
sort_options = {
    "SubmittedDate": arxiv.SortCriterion.SubmittedDate,
    "Relevance": arxiv.SortCriterion.Relevance
}
selected_sort_key = st.sidebar.selectbox("Sort By", list(sort_options.keys()), index=0)
selected_sort = sort_options[selected_sort_key]

def get_existing_urls():
    """
    If using a Notion database, query it to get a set of URLs that are already stored.
    Assumes that the "URL" property in your database is of type URL.
    """
    urls = set()
    if notion_database_id_input:
        data = notion.databases.query(database_id=notion_database_id_input)
        for page in data.get("results", []):
            url = page.get("properties", {}).get("URL", {}).get("url")
            if url:
                urls.add(url)
    return urls

def fetch_papers(query="brain-computer interface", max_results=5, fetch_limit=50):
    search = arxiv.Search(
        query=query,
        max_results=fetch_limit,
        sort_by=selected_sort
    )
    results = list(search.results())
    if notion_database_id_input:
        existing_urls = get_existing_urls()
        results = [paper for paper in results if paper.entry_id not in existing_urls]
    return results[:max_results]

def split_text_into_chunks(text, max_length=2000):
    return [text[i:i+max_length] for i in range(0, len(text), max_length)]

def create_toggle_block(heading, text):
    """Creates a toggle block with the given heading and text split into paragraph children."""
    toggle_children = []
    for chunk in split_text_into_chunks(text):
        toggle_children.append({
            "object": "block",
            "type": "paragraph",
            "paragraph": {"rich_text": [{"type": "text", "text": {"content": chunk}}]}
        })
    return [{
        "object": "block",
        "type": "toggle",
        "toggle": {
            "rich_text": [{"type": "text", "text": {"content": heading}}],
            "children": toggle_children
        }
    }]


def get_citation_count(arxiv_id):
    url = f"https://api.semanticscholar.org/v1/paper/arXiv:{arxiv_id}"
    try:
        response = requests.get(url).json()
        citation = response.get("citationCount", 0)
        st.write(f"Fetched citation count for {arxiv_id}: {citation}")
        return citation
    except Exception as e:
        st.write(f"Error fetching citation count for {arxiv_id}: {e}")
        return 0
def is_bci_related(abstract, related_terms):
    keywords = kw_model.extract_keywords(abstract, keyphrase_ngram_range=(1,2), stop_words="english")
    return any(keyword.lower() in related_terms for keyword, _ in keywords)

def compute_paper_score(arxiv_id, abstract, rank, total, related_terms):
    citation_score = get_citation_count(arxiv_id)
    
    # Determine bonus based on relevance
    if apply_bonus_multiplier:
        if is_bci_related(abstract, related_terms):
            # Ensure a minimum bonus of, say, 1 (adjust as needed) if citation_score is 0
            bonus = max(1, (total - rank) * bonus_multiplier)
        else:
            bonus = max(0.5, (total - rank) * bonus_multiplier * 0.5)
    else:
        bonus = 0

    total_score = citation_score + bonus
    st.write(f"Citation Score for {arxiv_id}: {citation_score}")
    st.write(f"Bonus for {arxiv_id}: {bonus}")
    st.write(f"Total Score for {arxiv_id}: {total_score}")
    return total_score

def summarize_text(text, style="normal"):
    prompt = summary_prompt_template.format(style=style, text=text)
    try:
        response = llm(prompt)
        st.write("DEBUG: Generated response:", response)
        cleaned_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
        return cleaned_response.strip()
    except Exception as e:
        st.write(f"Error during summarization: {e}")
        return ""

import datetime

def send_to_notion(title, url, pub_date, summary_normal, summary_thrilling, score):
    children = [{
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": f"URL: {url}"}}]
        }
    }]
    # Use the style values in the headings for the toggle blocks
    children.extend(create_toggle_block(f"Summary", summary_normal))
    children.extend(create_toggle_block(f"Summary Style: {style_thrilling_input}", summary_thrilling))
    children.append({
        "object": "block",
        "type": "paragraph",
        "paragraph": {
            "rich_text": [{"type": "text", "text": {"content": f"Score: {score}"}}]
        }
    })
    
    parent = {"page_id": notion_page_id_input}
    properties = {"title": {"title": [{"text": {"content": title}}]}}
    
    if notion_database_id_input:
        parent = {"database_id": notion_database_id_input}
        properties["URL"] = {"url": url}
        properties["Score"] = {"rich_text": [{"text": {"content": str(score)}}]}
        properties["Publication Date"] = {"date": {"start": pub_date}}
        current_dt = datetime.datetime.now().strftime("%Y-%m-%d")
        properties["Fetched on"] = {"date": {"start": current_dt}}
    
    response = notion.pages.create(
        parent=parent,
        properties=properties,
        children=children
    )
    return response.get("id")

# --- Main function ---
def main(query, max_results, fetch_limit):
    st.write("Fetching papers...")
    papers = fetch_papers(query=query, max_results=max_results, fetch_limit=fetch_limit)
    total = len(papers)
    st.write(f"Found {total} papers.")
    for rank, paper in enumerate(papers):
        arxiv_id = paper.entry_id.split("/")[-1]
        abstract = paper.summary
        title = paper.title
        url = paper.entry_id
        pub_date = paper.published.strftime("%Y-%m-%d") if hasattr(paper, "published") else ""
        score = compute_paper_score(arxiv_id, abstract, rank, total, user_related_terms)
        st.write(f"Summarizing: {title} (Score: {score})")
        summary_normal = summarize_text(abstract, style=style_normal_input)
        summary_thrilling = summarize_text(abstract, style=style_thrilling_input)
        page_id = send_to_notion(title, url, pub_date, summary_normal, summary_thrilling, score)
        if page_id:
            st.write(f"Created Notion page with ID: {page_id}")
        else:
            st.write("Skipping duplicate paper.")

# --- Streamlit UI ---
st.title("ArXiv Paper Summarizer and Notion Automation")

query_input = st.text_input("Enter search query:", value="brain-computer interface AI")
max_results_input = st.number_input("Max results:", min_value=1, value=5, step=1)
fetch_limit_input = st.number_input("Fetch limit:", min_value=1, value=50, step=1)

if st.button("Run"):
    main(query_input, max_results_input, fetch_limit_input)
    st.success("Done! Papers sent to Notion.")
import os
import re
import datetime
import arxiv
import streamlit as st
from keybert import KeyBERT
from langchain_ollama.llms import OllamaLLM
from dotenv import load_dotenv
import subprocess
import requests


# Load environment variables
load_dotenv()

st.sidebar.title("Environment Setup")

# Force UTF-8 for stdout/stderr
os.environ["PYTHONIOENCODING"] = "utf-8"

# Initialize models (moved after streamlit setup)
@st.cache_resource
def initialize_models():
    try:
        import torch
        from sentence_transformers import SentenceTransformer
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            st.warning("⚠️ Running on CPU. For better performance, install NVIDIA drivers and CUDA toolkit.")
        else:
            st.success("🚀 Running on GPU")
        
        # Initialize the base model with the correct device
        model = KeyBERT(model='all-MiniLM-L6-v2')
        return model
    except Exception as e:
        st.error(f"Error initializing models: {str(e)}")
        return KeyBERT(model='all-MiniLM-L6-v2')

kw_model = initialize_models()

# Sidebar: Model selection for Ollama 
available_models = [os.getenv("LLM_MODEL")]
selected_model = st.sidebar.selectbox("Select Ollama Model", available_models)
llm = OllamaLLM(model=selected_model)

# Sidebar: Configure summary prompt template
default_summary_prompt = os.getenv("SUMMARY_PROMPT_TEMPLATE")


summary_prompt_template = st.sidebar.text_area("Summary Prompt Template", value=default_summary_prompt)

# Sidebar: Configure styles for summaries
style_normal_input = st.sidebar.text_input("Enter the style you would like the normal summary to be written in:", value="normal")
style_thrilling_input = st.sidebar.text_input("Enter the style you would like the thrilling summary to be written in:", value="exciting manga-style")

# Sidebar: Configure related terms (comma separated)
related_terms_input = st.sidebar.text_input("Enter related terms (comma separated):", 
                                              value=os.getenv("ARXIV_QUERY"))
user_related_terms = {term.strip().lower() for term in related_terms_input.split(",") if term.strip()}

# Sidebar: Configure bonus multiplier for scoring
bonus_multiplier = st.sidebar.slider("Bonus Multiplier", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
# New checkbox: Enable/Disable bonus multiplier
apply_bonus_multiplier = st.sidebar.checkbox("Apply Bonus Multiplier", value=True)



with st.sidebar.expander("How Bonus Multiplier Works"):
    st.markdown(r"""
    The bonus multiplier is applied to increase a paper's score based on its rank if it is relevant.
    
    **Calculation:**  
    For a paper, the bonus is calculated as:
    
    \[  # noqa: W605
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
    urls = set()
    try:
        with open("visited_url.txt", "r") as f:
            urls = set(line.strip() for line in f)
    except FileNotFoundError:
        urls = set()

    return urls

def fetch_papers(query=os.getenv("ARXIV_QUERY", "brain-computer interface"), max_results=5, fetch_limit=50):
    # Create arxiv client
    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=fetch_limit,
        sort_by=selected_sort
    )
 
    # Use client.results() instead of search.results()
    results = list(client.results(search))
    existing_urls = get_existing_urls()
    
    # Filter out papers that we've already processed
    filtered_results = [
        paper for paper in results 
        if paper.entry_id not in existing_urls
    ][:max_results]
    
    return filtered_results

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

# --- Main function ---
def main(query, max_results, fetch_limit):
    # Create papers directory if it doesn't exist
    os.makedirs("papers", exist_ok=True)
    
    # Fetch and process papers
    papers = fetch_papers(query, max_results, fetch_limit)
    
    if not papers:
        st.warning("No new papers found or all papers have been processed already.")
        return

    for paper in papers:
        arxiv_id = paper.entry_id.split("/")[-1]
        authors = ", ".join(str(author.name) for author in paper.authors)  # Fixed author extraction
        abstract = paper.summary
    
        title = paper.title
        url = paper.entry_id
        pub_date = paper.published.strftime("%Y-%m-%d") if hasattr(paper, "published") else ""
        score = compute_paper_score(arxiv_id, abstract, papers.index(paper), len(papers), user_related_terms)
        st.write(f"Summarizing: {title} (Score: {score})")
        summary_normal = summarize_text(abstract, style=style_normal_input).encode('utf-8').decode('ascii', 'ignore')
        summary_thrilling = summarize_text(abstract, style=style_thrilling_input).encode('utf-8').decode('ascii', 'ignore')
        with open(f"papers/{arxiv_id}.md", "w") as f:
            f.write(f"# {title}\n")
            f.write(f"Authors: {authors}\n")
            f.write(f"URL: {url}\n")
            f.write(f"Publication Date: {pub_date}\n")
            f.write(f"Score: {score}\n")
            f.write("\n")
            f.write("## Summary (Normal)\n")
            f.write(summary_normal + "\n")
            f.write("\n")
            f.write(f"## Summary (Style: {style_thrilling_input})\n")
            f.write(summary_thrilling + "\n")


        with open("visited_url.txt", "a") as f:
            f.write(url + "\n")

# --- Streamlit UI ---
st.title("ArXiv Paper Summarizer")

query_input = st.text_input("Enter search query:", value=os.getenv("ARXIV_QUERY"))
max_results_input = st.number_input("Max results:", min_value=1, value=5, step=1)
fetch_limit_input = st.number_input("Fetch limit:", min_value=1, value=50, step=1)

if st.button("Run"):
    main(query_input, max_results_input, fetch_limit_input)
    st.success("Done! Papers summarized and saved to the 'papers' directory.")
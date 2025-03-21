# Notion Research Paper Automation

This project fetches research papers from ArXiv, generates summaries using a local LLM via Ollama, scores the papers based on relevance, and uploads the results to Notion. The project is built with Streamlit and can be packaged as a standalone executable.

## Features: 
- Fetch Papers from ArXiv: Retrieve papers using a custom query and sort them by your chosen criterion. 
- Dynamic Summarization: Generate structured summaries with configurable prompt templates and writing styles using Ollama. 
- GPU-Accelerated Keyword Extraction: Utilizes CUDA-enabled GPU (when available) for faster keyword processing with KeyBERT.
- Paper Scoring: Compute a bonus score based on paper relevance and rank. 
- Notion Integration: Automatically create pages (or database entries) in Notion with details like URL, Publication Date, Fetched on date, and Score. 
- User Configuration: Configure settings such as Notion credentials, summary styles, related terms, and bonus multipliers via a Streamlit sidebar.

## Notion Integration Setup:

### Create a Notion Integration:
Go to Notion Developers (https://www.notion.so/my-integrations) and click "New integration".
Enter a name for your integration, select your workspace, and grant it the required permissions.
Copy your Internal Integration Token.
Share a Page or Database with Your Integration:
Open the target page or database in Notion.
Click the "Share" button at the top-right corner.
Invite your integration by its name.
If using a database, ensure it has the following properties: o URL (Type: URL) o Score (Type: Rich Text) o Publication Date (Type: Date) o Fetched on (Type: Date)

## Setup Instructions:

[Notion Database](media/notion-table.png)

[Setup](./media/env-setup.webm)

[Demo](./media/demo.webm)


Clone the Repository Example: git clone https://github.com/yourusername/notion-research-paper-automation.git cd notion-research-paper-automation

Create and Activate a Virtual Environment 

On Windows:
```bash
 python -m venv notion-venv .\notion-venv\Scripts\activate 
 ```

On macOS/Linux: 
```bash
python3 -m venv notion-venv source notion-venv/bin/activate
```

Install Required Dependencies Ensure you have a requirements.txt file in your repository.
Then run: 
```bash
pip install -r requirements.txt
```

Configure Environment Variables Create a file named .env in the project root and add your credentials and settings: 
```bash
# Notion API Configuration
NOTION_API_KEY=your_notion_api_key_here
NOTION_DATABASE_ID=your_notion_database_id_here

# ArXiv API Configuration
ARXIV_QUERY="AI Machine learning"  # Default search query
MAX_RESULTS=10  # Maximum number of papers to fetch

# Update Frequency
UPDATE_INTERVAL_HOURS=24  # How often to check for new papers

# GPU Configuration (optional)
CUDA_VISIBLE_DEVICES=0
TORCH_DEVICE=cuda
```

(Ensure your .gitignore file excludes the .env file to protect sensitive information.)

## System Requirements

### Basic Requirements
- Python 3.8 or higher
- pip package manager
- Virtual environment (recommended)

### For GPU Acceleration (Optional)
To enable GPU acceleration for faster keyword extraction:
1. NVIDIA GPU with CUDA support
2. NVIDIA GPU Drivers
3. CUDA Toolkit (recommended version 11.8)
4. cuDNN library

The application will automatically use GPU if available, otherwise will fall back to CPU mode.

Run the Streamlit App Launch the app using: 

```bash

streamlit run app.py 
```

This command will open your default browser to the Streamlit interface. Use the sidebar to configure settings (Notion credentials, summary styles, related terms, bonus multiplier, sort criterion, etc.) and click "Run" to process the papers and send them to Notion.

Build a Standalone Executable (EXE) To package the project as an executable named "notion-research-paper-automation": a. Install PyInstaller: 

```bash

pip install pyinstaller 

```

b. Build the Executable: '

```bash

pyinstaller --onefile --noconsole --name notion-research-paper-automation app.py 
```

The executable will be created in the "dist" directory. Double-click the executable to launch the Streamlit app in your default browser.

## License: This project is licensed under the MIT License.
You can save the above content in a plain text file named "README.txt" (or "README" with no extension) for use in your repository.


# Pull the llama3 
 ollama pull llama3.3     

 ollama pull deepseek-r1:1.5b

 ollama pull qwen2.5:latest

 ollama pull llama3.2:latest
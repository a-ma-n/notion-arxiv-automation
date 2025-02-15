# Notion Research Paper Automation

This project fetches research papers from ArXiv, generates summaries using a local LLM via Ollama, scores the papers based on relevance, and uploads the results to Notion. The project is built with Streamlit and can be packaged as a standalone executable.

## Features: 
- Fetch Papers from ArXiv: Retrieve papers using a custom query and sort them by your chosen criterion. 
- Dynamic Summarization: Generate structured summaries with configurable prompt templates and writing styles using Ollama. 
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

[Notion Database](.media/notion-table.png)
[Setup](./env-setup.webm)
[Demo](./demo.webm)


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

Configure Environment Variables Create a file named .env in the project root and add your Notion credentials in the following format: 
```bash

NOTION_TOKEN=your_notion_integration_token 
NOTION_PAGE_ID=your_notion_page_id 
NOTION_DATABASE_ID=your_notion_database_id #(optional, if using a database) 
```

(Ensure your .gitignore file excludes the .env file to protect sensitive information.)

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
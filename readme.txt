Cloning the Repository and Setting Up the Environment:
To get started, follow these steps to clone the repository, set up a virtual environment, and
install the dependencies on your platform.
Step 1: Clone the Repository
Run the following command in your terminal or command prompt to clone the repository:
git clone https://github.com/r-brema/support-and-faq-assistant-open-ai-chatbot.git
cd support-and-faq-assistant-open-ai-chatbot
Step 1: move to backend folder
Step 2: Create a Virtual Environment
Linux/Mac:
Run:
python3 -m venv .venv
source .venv/bin/activate
Windows:
Run:
python -m venv .venv
.venv\Scripts\activate

Step 3: Install Requirements
With the virtual environment activated, install the required packages:
 pip install -r requirements.txt
 
Step 4: Verify Setup
Run the following command to ensure all dependencies are installed correctly:
 python -m pip list


# 3) Ingest once
python -m app.ingestion.json_ingest

# 4)start the Serve API (unchanged)
uvicorn app.main:app --reload --port 8000

 http://127.0.0.1:8000

PowerShell command to delete ./chroma_db:
 ri -r -fo .\chroma_db






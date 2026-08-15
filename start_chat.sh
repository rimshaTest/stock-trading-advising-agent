python -m venv .venv
source .venv/Scripts/activate  # On Windows
pip install -r requirements.txt
ollama pull llama3.1:latest
python main.py
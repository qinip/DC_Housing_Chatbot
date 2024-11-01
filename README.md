# DC Housing Policy Chatbot (Demo)

A document QA chatbot demo built with LangChain and ChromaDB for analyzing DC housing policy documents.

## Credit
This project is adapted from [this tutorial](https://github.com/aidev9/tuts/tree/main/langchain-rag-pdf/tutorial-1) and Claude.ai's [Contextual Retrieval Cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings).

## Environment Setup

### System Requirements
- Tested on MacOS (Apple Silicon)
- Python 3.12.4

### Installation
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. (Optional) For local embedding/LLM:
- Install [Ollama](https://ollama.com/docs/installation) (Mac users can use Homebrew):
```bash
brew install ollama
ollama pull mxbai-embed-large
ollama pull llama3.2
```

3. API Setup:
- Copy `.env.example` to `.env`
- Add your API keys for:
  - OpenAI (embeddings)
  - Choice of LLM providers:
    - OpenAI
    - Anthropic
    - Groq
    - OpenRouter

Note on LLM Providers:
- Groq: Known for its ultra-fast inference speeds, especially for Claude and Mixtral models. Visit [Groq](https://console.groq.com) to get API key and see available models.
- OpenRouter: Acts as a middleware service that provides access to various LLMs (including GPT-4, Claude, Mixtral, etc.) through a unified API. Check [OpenRouter](https://openrouter.ai/docs) for available models and pricing.

## Project Structure
```
.
├── data/           # PDF documents
├── db/             # ChromaDB vector database
├── models.py       # Model configurations
├── ingest.py      # Document processing script
├── chat.py        # Chat interface
└── test_qa.md     # Test questions and answers
```

## Usage
1. Start document processing:
```bash
python ingest.py
```
- Monitors `./data` folder for new PDFs
- Processed files are marked with "_" prefix
- Drag-n-drop knowledge base updates

2. Start chat interface:
```bash
python chat.py
```
- Choose between terminal or Gradio web interface
- Terminal: Type 'q' to quit
- Web UI: Use Ctrl+C to stop server

Note on Gradio:
[Gradio](https://www.gradio.app) is an open-source Python library that makes it easy to create customizable web interfaces for ML models. 
- By default, the UI is accessible only locally
- To create a public link, set `share=True` in `launch()`:
  ```python
  demo.launch(show_api=False, share=True)
  ```
- Gradio will generate a temporary public URL (valid for 72 hours)
- This allows others to access your chatbot through the internet

## Key Parameters

### Document Processing
```python
chunk_size = 1000   # Characters per chunk
chunk_overlap = 200 # Overlap between chunks
```
Adjust based on your documents for optimal QA performance.

### Retrieval Strategy
In `chat.py`:
- Change `USE_HYBRID` to choose between simple vector retrieval or hybrid (vector + BM25) retrieval
- Adjustable parameters:
  - `k` value (default: 10) - number of retrieved documents
  - Vector/BM25 weights in hybrid retrieval (default: 0.8/0.2)

## Evaluation
See `test_qa.md` for three types of test questions designed by Claude 2.5 Sonnet:
1. Factual questions (single-source)
2. Synthetic questions (multi-source)
3. Analytical questions (policy recommendations)

## Used Documents 

Document Title | Year | Page Count
--- | :---: | :---:
[A 'Perfect storm' of problems pulls D.C. toward full-blown housing crisis (Washington Post)](https://www.washingtonpost.com/dc-md-va/2024/03/28/dc-housing-crisis-affordable-housing/) | 2024 | 8
[Single-Family Zoning in the District of Columbia](https://dchousing.dc.gov/sites/default/files/2020-09/Single-Family_Zoning_in_the_District_of_Columbia.pdf) | 2020 | 24
[Affordable Housing Policies in the Washington D.C. Metropolitan Area (LMU Honors Thesis)](https://digitalcommons.lmu.edu/cgi/viewcontent.cgi?article=1624&context=honors-thesis) | 2023 | 33
[DC Housing Survey Report: A Supplement to the Assessment of the Need for Large Units](https://dmped.dc.gov/sites/default/files/dc/sites/dmped/publication/attachments/Formatted%20DC%20Housing%20Survey%20Report_FINAL%206-24_1.pdf) | 2019 | 19
[An Assessment of the Need for Large Units in the District of Columbia](https://dmped.dc.gov/sites/default/files/dc/sites/dmped/publication/attachments/Formatted%20FSU%20Study_FINAL%206-24_1.pdf) | 2019 | 67
[Families sue D.C. for ending housing aid in unprecedented case](https://www.washingtonpost.com/dc-md-va/2024/10/25/dc-rapid-rehousing-lawsuit/) | 2024 | 5
[Housing Equity Report: Creating Goals for Areas of Our City](https://planning.dc.gov/sites/default/files/dc/sites/housingdc/publication/attachments/Housing%20Equity%20Report.pdf) | 2019 | 20
[Housing Insecurity in the District of Columbia](https://www.urban.org/sites/default/files/2023-11/Housing%20Insecurity%20in%20the%20District%20of%20Columbia_0.pdf) | 2023 | 101

### Built with

- [LangChain](https://python.langchain.com/)
- [ChromaDB](https://www.trychroma.com/)
- [Gradio](https://www.gradio.app/)
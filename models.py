import os
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_groq import ChatGroq

class Models:
    def __init__(self):
        # ollama pull mxbai-embed-large
        self.embeddings_ollama = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # ollama pull llama3.2
        self.model_ollama = ChatOllama(
            model="llama3.2",
            temperature=0,
        )

        # OpenAI embeddings
        self.embeddings_openai = OpenAIEmbeddings(
            model="text-embedding-3-large",
            dimensions=1024,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )

        # OpenAI chat model
        # self.model_openai = ChatOpenAI(
        #     model="gpt-4o",
        #     temperature=0,
        #     max_tokens=None,
        #     timeout=None,
        #     max_retries=2,
        #     api_key=os.environ.get("OPENAI_API_KEY"),
        # )

        # Anthropic Claude
        # self.model_claude = ChatAnthropic(
        #     model="claude-3-haiku-20240307",
        #     temperature=0,
        #     anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY"),
        # )

        # Groq
        self.model_groq = ChatGroq(
            temperature=0,
            model_name="llama-3.2-90b-vision-preview",
            api_key=os.environ.get("GROQ_API_KEY"),
        )

        # OpenRouter
        self.model_openrouter = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            model= "anthropic/claude-3.5-sonnet:beta", #"openai/gpt-4o-mini-2024-07-18",  
            temperature=0,
            openai_api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
from setuptools import setup, find_packages

setup(
    name="human2",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "transformers",
        "gymnasium",
        "stable-baselines3",
        "PyGithub",
        "markdown",
        "beautifulsoup4",
        "langchain",  # Core LangChain functionality
        "langchain-community",  # Community components
        "langchain-huggingface",  # HuggingFace integrations
        "langchain-chroma",  # Vector store
        "chromadb",
        "python-dotenv",
        "radon",  # Code metrics
        "pylint",  # Code analysis
        "pdfplumber",  # PDF processing
        "reportlab",  # PDF testing
        "unstructured",  # Document processing
    ],
    python_requires=">=3.8",
) 
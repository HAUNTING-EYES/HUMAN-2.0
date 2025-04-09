import os
import logging
from pathlib import Path
from typing import List
from github_collector import GitHubDataCollector
from doc_collector import DocumentationCollector

def setup_logging():
    """Set up logging configuration"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('data_collection.log'),
            logging.StreamHandler()
        ]
    )

def get_github_repos() -> List[str]:
    """Get list of GitHub repositories to collect from"""
    return [
        'https://github.com/Significant-Gravitas/AutoGPT',
        'https://github.com/yoheinakajima/babyagi',
        'https://github.com/langchain-ai/langchain',
        'https://github.com/openai/openai-python',
        'https://github.com/ChromaDB/ChromaDB',
        'https://github.com/pytorch/pytorch',
        'https://github.com/tensorflow/tensorflow',
        'https://github.com/scikit-learn/scikit-learn',
        'https://github.com/pandas-dev/pandas',
        'https://github.com/numpy/numpy'
    ]

def get_documentation_urls() -> List[str]:
    """Get list of documentation URLs to collect from"""
    return [
        'https://docs.python.org/3/',
        'https://pytorch.org/docs/stable/index.html',
        'https://www.tensorflow.org/api_docs',
        'https://scikit-learn.org/stable/documentation.html',
        'https://pandas.pydata.org/docs/',
        'https://numpy.org/doc/'
    ]

def main():
    """Main function to run data collection"""
    # Set up logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Get base directory
    base_dir = Path(__file__).parent.parent.parent
    
    try:
        # Initialize collectors
        github_token = os.getenv('GITHUB_TOKEN')
        github_collector = GitHubDataCollector(base_dir, github_token)
        doc_collector = DocumentationCollector(base_dir)
        
        # Collect from GitHub
        logger.info("Starting GitHub data collection...")
        github_repos = get_github_repos()
        github_collector.collect_from_repositories(github_repos)
        
        # Collect documentation
        logger.info("Starting documentation collection...")
        doc_urls = get_documentation_urls()
        doc_collector.collect_from_urls(doc_urls)
        
        # Collect local documentation
        logger.info("Collecting local documentation...")
        local_docs = list(base_dir.glob('**/*.md'))
        doc_collector.collect_from_files([str(doc) for doc in local_docs])
        
        logger.info("Data collection completed successfully")
        
    except Exception as e:
        logger.error(f"Error during data collection: {str(e)}")
        raise

if __name__ == '__main__':
    main() 
import os
import json
import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from src.components.web_learning import WebLearningSystem

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def collect_data(
    github_repos: Optional[List[str]] = None,
    doc_urls: Optional[List[str]] = None,
    base_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Collect data from GitHub repositories and documentation sources.
    
    Args:
        github_repos: List of GitHub repository URLs
        doc_urls: List of documentation URLs
        base_dir: Base directory for data storage
        
    Returns:
        Dictionary containing collection results
    """
    logger.info("Starting data collection process...")
    results = {
        'success': True,
        'github_repos': [],
        'documentation': [],
        'code_patterns': [],
        'examples': []
    }
    
    try:
        # Initialize web learning system
        system = WebLearningSystem(base_dir=base_dir)
        
        # Set GitHub token if available
        github_token = os.environ.get('GITHUB_TOKEN')
        if github_token:
            system.set_github_token(github_token)
            logger.info("GitHub token set successfully")
        
        # Collect data from GitHub repositories
        if github_repos:
            logger.info(f"Collecting data from {len(github_repos)} GitHub repositories...")
            for repo_url in github_repos:
                logger.info(f"Processing repository: {repo_url}")
                try:
                    repo_data = system.process_url(repo_url)
                    if repo_data['success']:
                        results['github_repos'].append({
                            'url': repo_url,
                            'summary': repo_data['summary'],
                            'categories': repo_data['categories']
                        })
                        logger.info(f"Successfully collected data from {repo_url}")
                    else:
                        logger.error(f"Failed to process repository {repo_url}: {repo_data['error']}")
                except Exception as e:
                    logger.error(f"Error processing repository {repo_url}: {str(e)}")
        
        # Collect data from documentation sources
        if doc_urls:
            logger.info(f"Collecting data from {len(doc_urls)} documentation sources...")
            for doc_url in doc_urls:
                logger.info(f"Processing documentation: {doc_url}")
                try:
                    doc_data = system.process_url(doc_url)
                    if doc_data['success']:
                        results['documentation'].append({
                            'url': doc_url,
                            'summary': doc_data['summary'],
                            'categories': doc_data['categories']
                        })
                        logger.info(f"Successfully collected data from {doc_url}")
                    else:
                        logger.error(f"Failed to process documentation {doc_url}: {doc_data['error']}")
                except Exception as e:
                    logger.error(f"Error processing documentation {doc_url}: {str(e)}")
        
        # Extract code patterns
        code_patterns = system.extract_code_patterns()
        results['code_patterns'] = code_patterns
        
        # Generate examples
        examples = system.generate_examples()
        results['examples'] = examples
        
        # Clean up
        system.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in data collection: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

if __name__ == '__main__':
    # Example usage
    github_repos = [
        'https://github.com/Significant-Gravitas/AutoGPT',
        'https://github.com/langchain-ai/langchain',
        'https://github.com/pytorch/pytorch',
        'https://github.com/tensorflow/tensorflow',
        'https://github.com/openai/gpt-3',
        'https://github.com/microsoft/TypeScript',
        'https://github.com/facebook/react',
        'https://github.com/kubernetes/kubernetes'
    ]
    
    doc_urls = [
        'https://docs.python.org/3/',
        'https://pytorch.org/docs/stable/',
        'https://www.tensorflow.org/guide',
        'https://reactjs.org/docs/getting-started.html',
        'https://kubernetes.io/docs/home/',
        'https://www.typescriptlang.org/docs/'
    ]
    
    collect_data(github_repos=github_repos, doc_urls=doc_urls) 
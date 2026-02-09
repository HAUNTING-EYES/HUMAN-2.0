"""
HUMAN 2.0 - LLM Module
Provides local and cloud LLM clients with intelligent routing.
"""

from .local_llm_client import LocalLLMClient
from .cloud_llm_client import CloudLLMClient
from .router import LLMRouter

__all__ = ['LocalLLMClient', 'CloudLLMClient', 'LLMRouter']

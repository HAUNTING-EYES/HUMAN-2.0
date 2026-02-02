"""
Firecrawl Knowledge Integration for HUMAN 2.0
Real-time web knowledge gathering for consciousness systems
"""

import requests
import json
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
from pathlib import Path
import os
from enum import Enum

@dataclass
class KnowledgeItem:
    """Structured knowledge item from web scraping"""
    url: str
    title: str
    content: str
    metadata: Dict[str, Any]
    timestamp: float
    domain: str
    confidence: float = 0.8

class FirecrawlKnowledgeGatherer:
    """
    Firecrawl integration for HUMAN 2.0's knowledge gathering
    Enhances consciousness systems with real-time web knowledge
    """
    
    def __init__(self, api_key: str = "fc-716c07d3e060432da2069a08b98a9bf1"):
        """
        Initialize Firecrawl knowledge gatherer
        
        Args:
            api_key: Firecrawl API key (can be set via environment)
            base_url: Firecrawl API base URL
        """
        self.api_key = api_key
        self.base_url = "https://api.firecrawl.dev/v0"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Knowledge storage
        self.knowledge_cache: Dict[str, KnowledgeItem] = {}
        self.search_history: List[Dict[str, Any]] = []
        
        # Integration with consciousness systems
        self.curiosity_engine = None
        self.reflection_engine = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
    def _get_api_key(self) -> str:
        """Get API key from environment or config"""
        api_key = os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            self.logger.warning("No Firecrawl API key found. Set FIRECRAWL_API_KEY environment variable.")
            return "demo-key"  # For testing
        return api_key
    
    def scrape_url(self, url: str, formats: List[str] = None) -> Optional[KnowledgeItem]:
        """
        Scrape a single URL and convert to knowledge item
        
        Args:
            url: URL to scrape
            formats: Output formats (markdown, html, links, etc.)
            
        Returns:
            KnowledgeItem if successful, None otherwise
        """
        if formats is None:
            formats = ["markdown", "html"]
            
        try:
            payload = {
                "url": url,
                "formats": formats,
                "onlyMainContent": True,
                "timeout": 30000
            }
            
            response = requests.post(
                f"{self.base_url}/scrape",
                headers=self.headers,
                json=payload,
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    return self._process_scrape_result(data["data"], url)
            else:
                self.logger.error(f"Scraping failed for {url}: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {e}")
            
        return None
    
    def search_and_scrape(self, query: str, limit: int = 5) -> List[KnowledgeItem]:
        """
        Search the web and scrape results
        
        Args:
            query: Search query
            limit: Maximum number of results
            
        Returns:
            List of knowledge items
        """
        try:
            payload = {
                "query": query,
                "limit": limit,
                "formats": ["markdown"],
                "onlyMainContent": True
            }
            
            response = requests.post(
                f"{self.base_url}/search",
                headers=self.headers,
                json=payload,
                timeout=120
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    knowledge_items = []
                    for result in data.get("data", []):
                        item = self._process_search_result(result, query)
                        if item:
                            knowledge_items.append(item)
                            self.knowledge_cache[item.url] = item
                    
                    # Record search in history
                    self.search_history.append({
                        "query": query,
                        "timestamp": time.time(),
                        "results_count": len(knowledge_items),
                        "urls": [item.url for item in knowledge_items]
                    })
                    
                    return knowledge_items
            else:
                self.logger.error(f"Search failed for '{query}': {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error searching for '{query}': {e}")
            
        return []
    
    def batch_scrape(self, urls: List[str]) -> List[KnowledgeItem]:
        """
        Scrape multiple URLs in batch
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of knowledge items
        """
        try:
            payload = {
                "urls": urls,
                "formats": ["markdown"],
                "onlyMainContent": True
            }
            
            response = requests.post(
                f"{self.base_url}/batch/scrape",
                headers=self.headers,
                json=payload,
                timeout=300
            )
            
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    # Get job ID and poll for results
                    job_id = data.get("id")
                    return self._poll_batch_results(job_id)
            else:
                self.logger.error(f"Batch scrape failed: {response.status_code}")
                
        except Exception as e:
            self.logger.error(f"Error in batch scrape: {e}")
            
        return []
    
    def enhance_curiosity_with_web_knowledge(self, curiosity_engine):
        """
        Enhance curiosity engine with web knowledge gathering
        
        Args:
            curiosity_engine: CuriosityEngine instance
        """
        self.curiosity_engine = curiosity_engine
        
        # Generate questions from curiosity engine
        questions = curiosity_engine.generate_curiosity()
        
        enhanced_knowledge = []
        for question in questions[:3]:  # Limit to top 3 questions
            # Convert question to search query
            search_query = self._question_to_search_query(question.content)
            
            # Search and gather knowledge
            knowledge_items = self.search_and_scrape(search_query, limit=3)
            
            # Add knowledge to curiosity engine
            for item in knowledge_items:
                curiosity_engine.update_knowledge({
                    "concept": item.title,
                    "definition": item.content[:500],  # First 500 chars
                    "source": item.url,
                    "related_concepts": self._extract_concepts(item.content),
                    "confidence": item.confidence,
                    "domain": item.domain
                })
            
            enhanced_knowledge.extend(knowledge_items)
        
        return enhanced_knowledge
    
    def enhance_reflection_with_web_context(self, reflection_engine, experience: Dict[str, Any]):
        """
        Enhance reflection with relevant web context
        
        Args:
            reflection_engine: ReflectionEngine instance
            experience: Experience to reflect on
        """
        self.reflection_engine = reflection_engine
        
        # Extract key concepts from experience
        key_concepts = self._extract_experience_concepts(experience)
        
        # Gather relevant web knowledge
        contextual_knowledge = []
        for concept in key_concepts[:2]:  # Limit to top 2 concepts
            knowledge_items = self.search_and_scrape(concept, limit=2)
            contextual_knowledge.extend(knowledge_items)
        
        # Add context to reflection
        enhanced_experience = experience.copy()
        enhanced_experience["web_context"] = [
            {
                "source": item.url,
                "title": item.title,
                "relevance": item.confidence,
                "content_preview": item.content[:200]
            }
            for item in contextual_knowledge
        ]
        
        # Process enhanced experience
        reflection_engine.process_experience(enhanced_experience)
        
        return contextual_knowledge
    
    def continuous_knowledge_monitoring(self, topics: List[str], interval_hours: int = 24):
        """
        Continuously monitor web for new knowledge on specified topics
        
        Args:
            topics: List of topics to monitor
            interval_hours: How often to check (in hours)
        """
        monitoring_data = {
            "topics": topics,
            "last_check": time.time(),
            "interval": interval_hours * 3600,  # Convert to seconds
            "new_knowledge_count": 0
        }
        
        # This would run in a background thread in a real implementation
        for topic in topics:
            knowledge_items = self.search_and_scrape(f"latest {topic} developments", limit=3)
            
            # Filter for truly new content
            new_items = [
                item for item in knowledge_items 
                if item.url not in self.knowledge_cache
            ]
            
            monitoring_data["new_knowledge_count"] += len(new_items)
            
            # Update cache
            for item in new_items:
                self.knowledge_cache[item.url] = item
        
        return monitoring_data
    
    def _process_scrape_result(self, data: Dict[str, Any], url: str) -> KnowledgeItem:
        """Process scraping result into knowledge item"""
        return KnowledgeItem(
            url=url,
            title=data.get("metadata", {}).get("title", "Unknown"),
            content=data.get("markdown", ""),
            metadata=data.get("metadata", {}),
            timestamp=time.time(),
            domain=self._extract_domain(url),
            confidence=0.8
        )
    
    def _process_search_result(self, result: Dict[str, Any], query: str) -> Optional[KnowledgeItem]:
        """Process search result into knowledge item"""
        if not result.get("markdown"):
            return None
            
        return KnowledgeItem(
            url=result.get("url", ""),
            title=result.get("metadata", {}).get("title", "Unknown"),
            content=result.get("markdown", ""),
            metadata=result.get("metadata", {}),
            timestamp=time.time(),
            domain=self._extract_domain(result.get("url", "")),
            confidence=0.9  # Higher confidence for search results
        )
    
    def _poll_batch_results(self, job_id: str, max_wait: int = 300) -> List[KnowledgeItem]:
        """Poll batch scraping results"""
        start_time = time.time()
        
        while time.time() - start_time < max_wait:
            try:
                response = requests.get(
                    f"{self.base_url}/batch/scrape/{job_id}",
                    headers=self.headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("status") == "completed":
                        knowledge_items = []
                        for result in data.get("data", []):
                            item = self._process_scrape_result(result, result.get("url", ""))
                            knowledge_items.append(item)
                        return knowledge_items
                    elif data.get("status") == "failed":
                        self.logger.error(f"Batch job {job_id} failed")
                        break
                
                # Wait before polling again
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error polling batch results: {e}")
                break
        
        return []
    
    def _question_to_search_query(self, question: str) -> str:
        """Convert curiosity question to search query"""
        # Remove question words and extract key terms
        query = question.lower()
        question_words = ["what", "why", "how", "when", "where", "who", "which"]
        
        for word in question_words:
            query = query.replace(word, "")
        
        # Clean up and extract meaningful terms
        query = " ".join(query.split()[:5])  # Take first 5 words
        return query.strip()
    
    def _extract_concepts(self, content: str) -> List[str]:
        """Extract key concepts from content"""
        # Simple concept extraction (could be enhanced with NLP)
        words = content.lower().split()
        
        # Filter for meaningful terms (longer than 3 chars, not common words)
        common_words = {"the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out", "day", "get", "has", "him", "his", "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy", "did", "its", "let", "put", "say", "she", "too", "use"}
        
        concepts = [
            word.strip(".,!?;:") 
            for word in words 
            if len(word) > 3 and word not in common_words
        ]
        
        # Return top 10 most frequent concepts
        from collections import Counter
        concept_counts = Counter(concepts)
        return [concept for concept, count in concept_counts.most_common(10)]
    
    def _extract_experience_concepts(self, experience: Dict[str, Any]) -> List[str]:
        """Extract key concepts from experience"""
        concepts = []
        
        # Extract from content
        content = str(experience.get("content", ""))
        concepts.extend(self._extract_concepts(content))
        
        # Extract from type
        exp_type = experience.get("type", "")
        if exp_type:
            concepts.append(exp_type)
        
        return concepts[:5]  # Top 5 concepts
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc
        except:
            return "unknown"
    
    def get_knowledge_summary(self) -> Dict[str, Any]:
        """Get summary of gathered knowledge"""
        return {
            "total_items": len(self.knowledge_cache),
            "domains": list(set(item.domain for item in self.knowledge_cache.values())),
            "recent_searches": len(self.search_history),
            "last_search": self.search_history[-1] if self.search_history else None,
            "cache_size_mb": sum(len(item.content) for item in self.knowledge_cache.values()) / (1024 * 1024)
        }

# Example usage and integration
if __name__ == "__main__":
    # Initialize knowledge gatherer
    kg = FirecrawlKnowledgeGatherer()
    
    # Test basic scraping
    knowledge = kg.scrape_url("https://en.wikipedia.org/wiki/Artificial_intelligence")
    if knowledge:
        print(f"Scraped: {knowledge.title}")
        print(f"Content length: {len(knowledge.content)} chars")
    
    # Test search functionality
    search_results = kg.search_and_scrape("machine learning consciousness", limit=3)
    print(f"Found {len(search_results)} search results")
    
    # Get summary
    summary = kg.get_knowledge_summary()
    print(f"Knowledge summary: {summary}") 
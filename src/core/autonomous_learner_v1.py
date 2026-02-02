#!/usr/bin/env python3
"""
HUMAN 2.0 Autonomous Learner V1
Real implementation that:
1. Uses curiosity engine to generate questions
2. Searches web for answers
3. Extracts and stores knowledge
4. Implements learned concepts as code
5. Validates implementations
"""

import os
import sys
import json
import time
import logging
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
from dotenv import load_dotenv

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from consciousness.curiosity import CuriosityEngine, KnowledgeDomain, Question


# Custom JSON encoder for Enum and other non-serializable types
class EnumEncoder(json.JSONEncoder):
    """JSON encoder that handles Enum types"""
    def default(self, obj):
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)
# from components.web_learning import WebLearningSystem
# from components.firecrawl_knowledge import FirecrawlKnowledgeGatherer
from core.self_improvement_v1 import LLMCodeImprover

# Load environment
load_dotenv()


class WebSearcher:
    """Search the web for knowledge"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.firecrawl_api_key = os.getenv('FIRECRAWL_API_KEY')
        self.together_api_key = os.getenv('TOGETHER_API_KEY')

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search web and return results"""

        self.logger.info(f"Searching web for: {query}")

        # Use DuckDuckGo API (free, no key needed)
        try:
            url = "https://api.duckduckgo.com/"
            params = {
                'q': query,
                'format': 'json',
                'no_html': 1,
                'skip_disambig': 1
            }

            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            results = []

            # Get abstract
            if data.get('AbstractText'):
                results.append({
                    'title': data.get('Heading', 'Summary'),
                    'url': data.get('AbstractURL', ''),
                    'snippet': data.get('AbstractText', ''),
                    'source': 'duckduckgo_abstract'
                })

            # Get related topics
            for topic in data.get('RelatedTopics', [])[:max_results]:
                if isinstance(topic, dict) and 'Text' in topic:
                    results.append({
                        'title': topic.get('Text', '')[:100],
                        'url': topic.get('FirstURL', ''),
                        'snippet': topic.get('Text', ''),
                        'source': 'duckduckgo_related'
                    })

            self.logger.info(f"Found {len(results)} results")
            return results

        except Exception as e:
            self.logger.error(f"Web search failed: {e}")
            return []

    def fetch_content(self, url: str) -> Optional[str]:
        """Fetch content from URL"""

        try:
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            response.raise_for_status()
            return response.text

        except Exception as e:
            self.logger.error(f"Failed to fetch {url}: {e}")
            return None


class KnowledgeExtractor:
    """Extract and structure knowledge from web content"""

    def __init__(self):
        self.llm = LLMCodeImprover()
        self.logger = logging.getLogger(__name__)

    def extract_knowledge(self, content: str, query: str) -> Dict[str, Any]:
        """Extract structured knowledge from content"""

        prompt = f"""Extract key knowledge from this content related to: {query}

Content:
{content[:3000]}  # Limit content size

Return a JSON with:
{{
    "summary": "brief summary of key points",
    "key_concepts": ["list of main concepts"],
    "code_examples": ["relevant code snippets if any"],
    "related_topics": ["related topics to explore"],
    "implementation_ideas": ["how this could be implemented"],
    "confidence": 0.0-1.0
}}
"""

        try:
            result = self.llm._call_llm(prompt, json_mode=True)
            return result

        except Exception as e:
            self.logger.error(f"Knowledge extraction failed: {e}")
            return {
                'summary': '',
                'key_concepts': [],
                'code_examples': [],
                'related_topics': [],
                'implementation_ideas': [],
                'confidence': 0.0
            }


class KnowledgeImplementer:
    """Implement learned knowledge as code"""

    def __init__(self):
        self.llm = LLMCodeImprover()
        self.logger = logging.getLogger(__name__)

    def generate_implementation(self, knowledge: Dict[str, Any]) -> Optional[str]:
        """Generate code implementing the learned knowledge"""

        if not knowledge.get('implementation_ideas'):
            return None

        prompt = f"""You are implementing new knowledge learned about: {knowledge.get('summary', '')}

Key Concepts:
{chr(10).join(f"- {c}" for c in knowledge.get('key_concepts', []))}

Implementation Ideas:
{chr(10).join(f"- {i}" for i in knowledge.get('implementation_ideas', []))}

Generate a Python class or module that implements this knowledge.
The code should be:
1. Production-ready
2. Well-documented
3. Include docstrings
4. Follow best practices
5. Include example usage

Return ONLY the Python code."""

        try:
            code = self.llm._call_llm(prompt, json_mode=False)
            return code

        except Exception as e:
            self.logger.error(f"Implementation generation failed: {e}")
            return None


class AutonomousLearnerV1:
    """
    Autonomous learning system that:
    - Generates curiosity-driven questions
    - Searches web for answers
    - Extracts knowledge
    - Implements learned concepts
    - Validates and integrates
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

        # Core components
        self.curiosity = CuriosityEngine()
        self.searcher = WebSearcher()
        self.extractor = KnowledgeExtractor()
        self.implementer = KnowledgeImplementer()

        # State tracking
        self.learned_knowledge = []
        self.implemented_code = []
        self.learning_cycle = 0

        # Storage
        self.knowledge_dir = Path('learned_data/autonomous')
        self.knowledge_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("AutonomousLearnerV1 initialized")

    def autonomous_learning_cycle(self, max_questions: int = 3):
        """Run one cycle of autonomous learning"""

        self.logger.info(f"=== Learning Cycle {self.learning_cycle} ===")

        # 1. Generate questions from curiosity
        questions = self.curiosity.generate_curiosity()

        if not questions:
            self.logger.info("No new questions generated - creating seed questions")
            questions = self._generate_seed_questions()

        self.logger.info(f"Generated {len(questions)} questions")

        # 2. Learn from top N questions
        learned_this_cycle = []

        for question in questions[:max_questions]:
            knowledge = self._learn_from_question(question)

            if knowledge and knowledge.get('confidence', 0) > 0.5:
                learned_this_cycle.append(knowledge)

        # 3. Implement learned knowledge
        for knowledge in learned_this_cycle:
            implementation = self._implement_knowledge(knowledge)

            if implementation:
                self.implemented_code.append({
                    'knowledge': knowledge,
                    'code': implementation,
                    'timestamp': datetime.now().isoformat(),
                    'cycle': self.learning_cycle
                })

        # 4. Save progress
        self._save_learning_state()

        # 5. Generate report
        report = self._generate_report(learned_this_cycle)

        self.learning_cycle += 1

        return report

    def _learn_from_question(self, question: Question) -> Optional[Dict[str, Any]]:
        """Learn from a single question"""

        self.logger.info(f"Learning: {question.content}")

        # 1. Search web
        search_results = self.searcher.search(question.content)

        if not search_results:
            self.logger.warning(f"No search results for: {question.content}")
            return None

        # 2. Extract knowledge from results
        all_knowledge = []

        for result in search_results[:3]:  # Top 3 results
            # Fetch content
            content = result.get('snippet', '')

            if not content:
                continue

            # Extract knowledge
            knowledge = self.extractor.extract_knowledge(content, question.content)
            knowledge['source_url'] = result.get('url', '')
            knowledge['source_title'] = result.get('title', '')

            all_knowledge.append(knowledge)

        if not all_knowledge:
            return None

        # 3. Combine knowledge
        combined = self._combine_knowledge(all_knowledge)
        combined['original_question'] = question.content
        combined['domain'] = question.domain.value

        # 4. Update curiosity engine
        self.curiosity.update_knowledge({
            'concept': question.content,
            'confidence': combined.get('confidence', 0.5),
            'related_concepts': combined.get('related_topics', [])
        })

        # 5. Save knowledge
        self.learned_knowledge.append(combined)

        return combined

    def _combine_knowledge(self, knowledge_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple knowledge sources"""

        combined = {
            'summary': '',
            'key_concepts': [],
            'code_examples': [],
            'related_topics': [],
            'implementation_ideas': [],
            'confidence': 0.0,
            'sources': len(knowledge_list)
        }

        # Merge all fields
        for knowledge in knowledge_list:
            if knowledge.get('summary'):
                combined['summary'] += knowledge['summary'] + ' '

            combined['key_concepts'].extend(knowledge.get('key_concepts', []))
            combined['code_examples'].extend(knowledge.get('code_examples', []))
            combined['related_topics'].extend(knowledge.get('related_topics', []))
            combined['implementation_ideas'].extend(knowledge.get('implementation_ideas', []))

            combined['confidence'] += knowledge.get('confidence', 0.0)

        # Average confidence
        if knowledge_list:
            combined['confidence'] /= len(knowledge_list)

        # Deduplicate
        combined['key_concepts'] = list(set(combined['key_concepts']))
        combined['related_topics'] = list(set(combined['related_topics']))
        combined['implementation_ideas'] = list(set(combined['implementation_ideas']))

        return combined

    def _implement_knowledge(self, knowledge: Dict[str, Any]) -> Optional[str]:
        """Implement learned knowledge as code"""

        self.logger.info(f"Implementing: {knowledge.get('original_question', 'knowledge')}")

        # Generate implementation
        code = self.implementer.generate_implementation(knowledge)

        if not code:
            return None

        # Save implementation
        filename = self._generate_filename(knowledge)
        filepath = self.knowledge_dir / filename

        with open(filepath, 'w') as f:
            f.write(f"# Auto-generated from learned knowledge\n")
            f.write(f"# Question: {knowledge.get('original_question', '')}\n")
            f.write(f"# Cycle: {self.learning_cycle}\n")
            f.write(f"# Confidence: {knowledge.get('confidence', 0.0):.2f}\n\n")
            f.write(code)

        self.logger.info(f"Saved implementation to: {filepath}")

        return code

    def _generate_filename(self, knowledge: Dict[str, Any]) -> str:
        """Generate filename for implementation"""

        question = knowledge.get('original_question', 'knowledge')

        # Clean filename
        filename = question.lower()
        filename = filename.replace(' ', '_')
        filename = ''.join(c for c in filename if c.isalnum() or c == '_')
        filename = filename[:50]  # Limit length

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        return f"{filename}_{timestamp}.py"

    def _generate_seed_questions(self) -> List[Question]:
        """Generate initial seed questions"""

        seed_topics = [
            ("How can I implement better code optimization algorithms?", KnowledgeDomain.COGNITIVE),
            ("What are best practices for Python code generation?", KnowledgeDomain.COGNITIVE),
            ("How does meta-learning work in AI systems?", KnowledgeDomain.ABSTRACT),
            ("What is self-modifying code and how to implement it safely?", KnowledgeDomain.COGNITIVE),
            ("How do AI systems learn from web resources?", KnowledgeDomain.COGNITIVE)
        ]

        questions = []
        for i, (content, domain) in enumerate(seed_topics):
            question = Question(
                id=f"seed_{i}",
                domain=domain,
                content=content,
                importance=0.8,
                urgency=0.6,
                complexity=0.7,
                related_nodes=[],
                status="unanswered"
            )
            questions.append(question)

        return questions

    def _save_learning_state(self):
        """Save current learning state"""

        state_file = self.knowledge_dir / 'learning_state.json'

        state = {
            'learning_cycle': self.learning_cycle,
            'total_knowledge': len(self.learned_knowledge),
            'total_implementations': len(self.implemented_code),
            'last_update': datetime.now().isoformat(),
            'recent_knowledge': self.learned_knowledge[-10:],
            'curiosity_state': self.curiosity.get_curiosity_state()
        }

        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, cls=EnumEncoder)

    def _generate_report(self, learned_this_cycle: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate learning report"""

        report = {
            'cycle': self.learning_cycle,
            'timestamp': datetime.now().isoformat(),
            'learned_this_cycle': len(learned_this_cycle),
            'total_learned': len(self.learned_knowledge),
            'total_implemented': len(self.implemented_code),
            'knowledge_details': learned_this_cycle
        }

        # Save report
        report_file = Path('reports') / f'learning_cycle_{self.learning_cycle}.json'
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, cls=EnumEncoder)

        self.logger.info(f"Learning report saved to: {report_file}")

        return report


def main():
    """Test autonomous learner"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Autonomous Learner V1")
    print("=" * 70)

    # Initialize
    learner = AutonomousLearnerV1()

    # Run learning cycle
    print("\nStarting autonomous learning cycle...")
    report = learner.autonomous_learning_cycle(max_questions=2)

    print("\nResults:")
    print(f"Cycle: {report['cycle']}")
    print(f"Learned: {report['learned_this_cycle']} new concepts")
    print(f"Total Knowledge: {report['total_learned']}")
    print(f"Implementations: {report['total_implemented']}")

    print("\nAutonomous learning cycle complete!")


if __name__ == "__main__":
    main()

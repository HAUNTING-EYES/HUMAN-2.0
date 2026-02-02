#!/usr/bin/env python3
"""
HUMAN 2.0 Code Embedder
Unified ChromaDB strategy for semantic code search and improvement tracking.

Unifies 3 existing ChromaDB instances into strategic collections:
1. "codebase" - All Python files for similarity search
2. "improvements" - History of improvements for meta-learning
3. "external_knowledge" - Patterns learned from GitHub/web
"""

import ast
import logging
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import chromadb
from chromadb.config import Settings
from langchain_huggingface import HuggingFaceEmbeddings


class CodeEmbedder:
    """
    Unified code embedding and semantic search using ChromaDB.

    Provides:
    - Semantic search for similar code
    - Improvement history tracking
    - External knowledge storage
    """

    def __init__(self, chroma_dir: str = "data/unified_chroma_db"):
        """
        Initialize code embedder with unified ChromaDB.

        Args:
            chroma_dir: Directory for ChromaDB persistence
        """
        self.logger = logging.getLogger(__name__)
        self.chroma_dir = Path(chroma_dir)
        self.chroma_dir.mkdir(parents=True, exist_ok=True)

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.chroma_dir),
            settings=Settings(anonymized_telemetry=False)
        )

        # Initialize embeddings model
        self.logger.info("Initializing HuggingFace embeddings...")
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu', 'batch_size': 32}
        )

        # Get or create collections
        self.codebase_collection = self.client.get_or_create_collection(
            name="codebase",
            metadata={"description": "All Python files in src/ for similarity search"}
        )

        self.improvements_collection = self.client.get_or_create_collection(
            name="improvements",
            metadata={"description": "History of code improvements for meta-learning"}
        )

        self.external_knowledge_collection = self.client.get_or_create_collection(
            name="external_knowledge",
            metadata={"description": "Patterns learned from GitHub and web"}
        )

        self.logger.info(f"CodeEmbedder initialized with ChromaDB at: {self.chroma_dir}")
        self.logger.info(f"Collections: {self.client.list_collections()}")

    def embed_codebase(self, target_dirs: List[str], root_dir: str = '.'):
        """
        Embed all Python files in target directories.

        Args:
            target_dirs: Directories to scan (relative to root_dir)
            root_dir: Root directory of project
        """
        self.logger.info(f"Embedding codebase from: {target_dirs}")

        root_path = Path(root_dir).resolve()
        python_files = []

        # Find all Python files
        for target_dir in target_dirs:
            target_path = root_path / target_dir

            if not target_path.exists():
                self.logger.warning(f"Directory not found: {target_path}")
                continue

            for py_file in target_path.rglob('*.py'):
                # Skip tests, __pycache__, etc.
                if any(skip in str(py_file) for skip in ['test_', '__pycache__', '.pytest_cache', 'archived_']):
                    continue

                python_files.append(str(py_file))

        self.logger.info(f"Found {len(python_files)} Python files to embed")

        # Embed each file
        embedded_count = 0
        for file_path in python_files:
            try:
                self._embed_file(file_path, root_path)
                embedded_count += 1

                if embedded_count % 10 == 0:
                    self.logger.info(f"Embedded {embedded_count}/{len(python_files)} files")

            except Exception as e:
                self.logger.error(f"Error embedding {file_path}: {e}")

        self.logger.info(f"Embedding complete: {embedded_count}/{len(python_files)} files")

    def _embed_file(self, file_path: str, root_dir: Path):
        """Embed a single file into codebase collection."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                code = f.read()

            # Extract metadata using AST
            metadata = self._extract_metadata(code, file_path, root_dir)

            # Generate embedding
            embedding = self.embeddings.embed_query(code)

            # Generate unique ID
            file_id = self._generate_file_id(file_path)

            # Store in ChromaDB
            self.codebase_collection.upsert(
                ids=[file_id],
                embeddings=[embedding],
                documents=[code],
                metadatas=[metadata]
            )

        except Exception as e:
            self.logger.error(f"Error embedding file {file_path}: {e}")
            raise

    def _extract_metadata(self, code: str, file_path: str, root_dir: Path) -> Dict[str, Any]:
        """Extract metadata from code using AST."""
        metadata = {
            'file_path': file_path,
            'relative_path': str(Path(file_path).relative_to(root_dir)),
            'last_modified': datetime.now().isoformat(),
            'num_lines': len(code.splitlines()),
        }

        try:
            tree = ast.parse(code)

            # Extract functions and classes
            functions = []
            classes = []
            imports = []

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append(node.name)
                elif isinstance(node, ast.ClassDef):
                    classes.append(node.name)
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            metadata['functions'] = ','.join(functions[:20])  # Limit to 20
            metadata['classes'] = ','.join(classes[:20])
            metadata['imports'] = ','.join(imports[:20])
            metadata['num_functions'] = len(functions)
            metadata['num_classes'] = len(classes)

        except Exception as e:
            self.logger.warning(f"Could not parse AST for {file_path}: {e}")

        return metadata

    def find_similar_code(self, code: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """
        Find similar code in codebase using semantic search.

        Args:
            code: Code snippet to search for
            n_results: Number of results to return

        Returns:
            List of similar code snippets with metadata
        """
        try:
            # Generate embedding for query
            query_embedding = self.embeddings.embed_query(code)

            # Query ChromaDB
            results = self.codebase_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            # Format results
            similar_code = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    similar_code.append({
                        'id': results['ids'][0][i],
                        'code': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i],
                        'distance': results['distances'][0][i] if 'distances' in results else None
                    })

            return similar_code

        except Exception as e:
            self.logger.error(f"Error finding similar code: {e}")
            return []

    def get_improvement_context(self, file_path: str, code: str) -> Dict[str, Any]:
        """
        Get comprehensive context for improving a file.

        Args:
            file_path: Path to file being improved
            code: Current code content

        Returns:
            Dict with similar code, past improvements, external patterns
        """
        context = {
            'similar_code': [],
            'past_improvements': [],
            'external_patterns': []
        }

        try:
            # Find similar code (reduced from 5 to 2 for performance)
            context['similar_code'] = self.find_similar_code(code, n_results=2)

            # Find past improvements for this file
            context['past_improvements'] = self._get_file_improvements(file_path)

            # Find relevant external knowledge (reduced from 3 to 2)
            context['external_patterns'] = self._get_external_knowledge(code, n_results=2)

        except Exception as e:
            self.logger.error(f"Error getting improvement context: {e}")

        return context

    def _get_file_improvements(self, file_path: str) -> List[Dict[str, Any]]:
        """Get past improvements for a specific file."""
        try:
            # Query improvements collection
            results = self.improvements_collection.get(
                where={"file_path": file_path}
            )

            improvements = []
            if results and results['ids']:
                for i in range(len(results['ids'])):
                    improvements.append({
                        'id': results['ids'][i],
                        'metadata': results['metadatas'][i],
                        'timestamp': results['metadatas'][i].get('timestamp')
                    })

            return improvements

        except Exception as e:
            self.logger.error(f"Error getting file improvements: {e}")
            return []

    def _get_external_knowledge(self, code: str, n_results: int = 3) -> List[Dict[str, Any]]:
        """Get relevant external knowledge patterns."""
        try:
            # Generate embedding for code
            query_embedding = self.embeddings.embed_query(code)

            # Query external knowledge
            results = self.external_knowledge_collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results
            )

            knowledge = []
            if results and results['ids'] and results['ids'][0]:
                for i in range(len(results['ids'][0])):
                    knowledge.append({
                        'id': results['ids'][0][i],
                        'pattern': results['documents'][0][i],
                        'metadata': results['metadatas'][0][i]
                    })

            return knowledge

        except Exception as e:
            self.logger.error(f"Error getting external knowledge: {e}")
            return []

    def record_improvement(self, file_path: str, before_code: str, after_code: str,
                          success: bool, metrics: Dict[str, Any], strategy: str = "default"):
        """
        Record improvement outcome for meta-learning.

        Args:
            file_path: Path to improved file
            before_code: Code before improvement
            after_code: Code after improvement
            success: Whether improvement was successful
            metrics: Metrics after improvement
            strategy: Strategy used for improvement
        """
        try:
            # Create combined document (before + after for context)
            document = f"BEFORE:\n{before_code}\n\nAFTER:\n{after_code}"

            # Generate embedding of combined document
            embedding = self.embeddings.embed_query(document)

            # Generate unique ID
            improvement_id = str(uuid.uuid4())

            # Metadata
            metadata = {
                'file_path': file_path,
                'strategy': strategy,
                'success': str(success),  # ChromaDB requires string
                'timestamp': datetime.now().isoformat(),
                'before_metrics': str(metrics.get('before_metrics', {})),
                'after_metrics': str(metrics.get('after_metrics', {})),
                'complexity': str(metrics.get('complexity', 0.0)),
                'maintainability': str(metrics.get('maintainability', 0.0))
            }

            # Store in improvements collection
            self.improvements_collection.add(
                ids=[improvement_id],
                embeddings=[embedding],
                documents=[document],
                metadatas=[metadata]
            )

            self.logger.info(f"Recorded improvement for {file_path} (success={success})")

        except Exception as e:
            self.logger.error(f"Error recording improvement: {e}")

    def store_external_knowledge(self, source: str, patterns: List[str],
                                 topic: str, related_question: str = None):
        """
        Store knowledge learned from external sources.

        Args:
            source: Source URL (GitHub repo, web page, etc.)
            patterns: List of code patterns or best practices
            topic: Topic or domain of knowledge
            related_question: Curiosity question that led to this learning
        """
        try:
            for i, pattern in enumerate(patterns):
                # Generate embedding
                embedding = self.embeddings.embed_query(pattern)

                # Generate unique ID
                pattern_id = hashlib.md5(f"{source}_{i}".encode()).hexdigest()

                # Metadata
                metadata = {
                    'source': source,
                    'topic': topic,
                    'timestamp': datetime.now().isoformat(),
                    'pattern_index': str(i),
                    'total_patterns': str(len(patterns))
                }

                if related_question:
                    metadata['related_question'] = related_question

                # Store in external knowledge collection
                self.external_knowledge_collection.upsert(
                    ids=[pattern_id],
                    embeddings=[embedding],
                    documents=[pattern],
                    metadatas=[metadata]
                )

            self.logger.info(f"Stored {len(patterns)} patterns from {source}")

        except Exception as e:
            self.logger.error(f"Error storing external knowledge: {e}")

    def migrate_existing_chromadb(self):
        """
        Migrate data from existing ChromaDB instances to unified structure.

        Existing instances:
        - chroma_db/ (root)
        - data/learning/chroma_db/
        - learned_data/chroma_db/
        """
        self.logger.info("Migrating existing ChromaDB instances...")

        old_instances = [
            'chroma_db',
            'data/learning/chroma_db',
            'learned_data/chroma_db'
        ]

        for old_path in old_instances:
            if not Path(old_path).exists():
                continue

            try:
                self.logger.info(f"Migrating from: {old_path}")

                # Open old instance
                old_client = chromadb.PersistentClient(path=old_path)

                # Get all collections
                collections = old_client.list_collections()

                for collection in collections:
                    self.logger.info(f"  Migrating collection: {collection.name}")

                    # Get all data from old collection
                    data = collection.get()

                    if not data or not data['ids']:
                        continue

                    # Determine target collection based on name
                    if 'learning' in collection.name or 'web' in collection.name:
                        target = self.external_knowledge_collection
                    else:
                        target = self.codebase_collection

                    # Migrate data
                    target.add(
                        ids=data['ids'],
                        embeddings=data['embeddings'] if 'embeddings' in data else None,
                        documents=data['documents'] if 'documents' in data else None,
                        metadatas=data['metadatas'] if 'metadatas' in data else None
                    )

                    self.logger.info(f"    Migrated {len(data['ids'])} items")

            except Exception as e:
                self.logger.error(f"Error migrating from {old_path}: {e}")

        self.logger.info("Migration complete")

    def _generate_file_id(self, file_path: str) -> str:
        """Generate unique ID for file based on path."""
        return hashlib.md5(file_path.encode()).hexdigest()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about embeddings."""
        return {
            'codebase_files': self.codebase_collection.count(),
            'improvements_recorded': self.improvements_collection.count(),
            'external_patterns': self.external_knowledge_collection.count(),
            'chroma_dir': str(self.chroma_dir)
        }


def main():
    """Test code embedder."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Code Embedder Test")
    print("=" * 70)

    # Initialize
    embedder = CodeEmbedder()

    # Print stats
    stats = embedder.get_stats()
    print(f"\nCurrent Stats:")
    print(f"  Codebase files: {stats['codebase_files']}")
    print(f"  Improvements recorded: {stats['improvements_recorded']}")
    print(f"  External patterns: {stats['external_patterns']}")

    # Embed codebase
    print("\nEmbedding codebase from src/...")
    embedder.embed_codebase(['src'])

    # Test similarity search
    print("\nTesting similarity search...")
    test_code = "def analyze_code(self, code: str):"
    similar = embedder.find_similar_code(test_code, n_results=3)

    print(f"Found {len(similar)} similar code snippets:")
    for i, result in enumerate(similar, 1):
        print(f"\n{i}. {result['metadata'].get('relative_path', 'unknown')}")
        print(f"   Functions: {result['metadata'].get('functions', 'none')}")

    # Print updated stats
    stats = embedder.get_stats()
    print(f"\nUpdated Stats:")
    print(f"  Codebase files: {stats['codebase_files']}")


if __name__ == "__main__":
    main()

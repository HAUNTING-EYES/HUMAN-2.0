#!/usr/bin/env python3
"""
HUMAN 2.0 - Setup and Verification Script
Prepares the system for autonomous operation:
1. Loads environment variables from .env
2. Builds dependency graph
3. Verifies API keys
4. Shows system status
"""

import os
import sys
from pathlib import Path

# Load .env file
print("\n" + "="*70)
print("  HUMAN 2.0 - SYSTEM SETUP & VERIFICATION")
print("="*70)
print()

# Try to load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Loaded .env file")
except ImportError:
    print("[!] python-dotenv not installed")
    print("    Installing now...")
    os.system("pip install python-dotenv -q")
    from dotenv import load_dotenv
    load_dotenv()
    print("[OK] Loaded .env file")

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Verify API keys
print("\n--- API KEYS ---")
anthropic_key = os.getenv('ANTHROPIC_API_KEY')
together_key = os.getenv('TOGETHER_API_KEY')
github_token = os.getenv('GITHUB_TOKEN')

if anthropic_key:
    print(f"[OK] ANTHROPIC_API_KEY: {anthropic_key[:10]}...{anthropic_key[-4:]}")
else:
    print("[X] ANTHROPIC_API_KEY: NOT FOUND")

if together_key:
    print(f"[OK] TOGETHER_API_KEY: {together_key[:10]}...{together_key[-4:]}")
else:
    print("[!] TOGETHER_API_KEY: NOT FOUND (optional)")

if github_token:
    print(f"[OK] GITHUB_TOKEN: {github_token[:10]}...{github_token[-4:]}")
else:
    print("[!] GITHUB_TOKEN: NOT FOUND (optional)")

# Build dependency graph
print("\n--- BUILDING DEPENDENCY GRAPH ---")
print("[*] This may take 10-20 seconds...")

try:
    from src.core.shared_resources import SharedResources

    resources = SharedResources()

    print("[*] Analyzing Python files in src/...")
    graph = resources.dependency_analyzer.build_graph(['src'])

    print(f"[OK] Dependency graph built!")
    print(f"     Nodes (files): {graph.number_of_nodes()}")
    print(f"     Edges (imports): {graph.number_of_edges()}")

    # Save the graph
    resources.save_all()
    print(f"[OK] Resources saved to disk")

except Exception as e:
    print(f"[X] Error building dependency graph: {e}")
    import traceback
    traceback.print_exc()

# Check ChromaDB
print("\n--- CHROMADB STATUS ---")
try:
    codebase_count = resources.code_embedder.codebase_collection.count()
    improvements_count = resources.code_embedder.improvements_collection.count()
    external_count = resources.code_embedder.external_knowledge_collection.count()

    print(f"[OK] Codebase: {codebase_count} files indexed")
    print(f"[OK] Improvements: {improvements_count} records")
    print(f"[OK] External Knowledge: {external_count} entries")
except Exception as e:
    print(f"[!] ChromaDB status unavailable: {e}")

# Check Knowledge Graph
print("\n--- KNOWLEDGE GRAPH STATUS ---")
try:
    print(f"[OK] Knowledge Nodes: {len(resources.knowledge_graph)}")
    print(f"[OK] Patterns: {len(resources.pattern_library)}")
    print(f"[OK] Cached Responses: {len(resources.model_cache)}")
except Exception as e:
    print(f"[!] Knowledge graph status unavailable: {e}")

# Test API connection (optional)
print("\n--- TESTING API CONNECTION ---")
if anthropic_key:
    print("[*] Testing Claude API connection...")
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)

        # Test with a simple message
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=50,
            messages=[
                {"role": "user", "content": "Say 'API test successful' in 3 words"}
            ]
        )

        response = message.content[0].text
        print(f"[OK] Claude API working! Response: '{response}'")

    except Exception as e:
        print(f"[X] Claude API test failed: {e}")
else:
    print("[!] Skipping API test (no key)")

# Final status
print("\n" + "="*70)
print("  SYSTEM STATUS")
print("="*70)

ready = True
issues = []

if not anthropic_key:
    ready = False
    issues.append("ANTHROPIC_API_KEY not set")

if graph.number_of_nodes() == 0:
    ready = False
    issues.append("Dependency graph is empty")

if ready:
    print("\n  [OK] SYSTEM READY FOR AUTONOMOUS OPERATION!")
    print()
    print("  You can now run:")
    print("    python run_autonomous_demo.py")
    print()
    print("  Or for continuous 24/7 operation:")
    print("    python run_autonomous_demo.py --continuous")
else:
    print("\n  [!] SYSTEM NOT READY")
    print()
    print("  Issues:")
    for issue in issues:
        print(f"    - {issue}")

print("\n" + "="*70)
print()

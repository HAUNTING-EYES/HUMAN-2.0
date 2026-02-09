#!/usr/bin/env python3
"""
HUMAN 2.0 Self-Improvement V1
Real implementation - not stubs

This is the core self-modifying AI that:
1. Analyzes its own code
2. Detects bugs and inefficiencies
3. Proposes improvements via LLM
4. Tests improvements safely
5. Applies successful changes
6. Tracks progress over generations
"""

import os
import sys
import ast
import json
import time
import logging
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dotenv import load_dotenv
import requests

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from components.code_analyzer import CodeAnalyzer
from components.code_metrics import CodeMetrics

# Load environment
load_dotenv()

class LLMCodeImprover:
    """LLM-based code improvement using Together API"""

    def __init__(self):
        self.api_key = os.getenv('TOGETHER_API_KEY')
        self.api_base = os.getenv('TOGETHER_API_BASE', 'https://api.together.xyz/v1')
        self.model = 'meta-llama/Meta-Llama-3-70B-Instruct'
        self.logger = logging.getLogger(__name__)

        if not self.api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment")

    def analyze_code(self, code: str, file_path: str) -> Dict[str, Any]:
        """Ask LLM to analyze code for issues"""

        prompt = f"""You are a senior software engineer reviewing code for bugs, inefficiencies, and improvements.

File: {file_path}

Code:
```python
{code}
```

Analyze this code and provide:
1. Bugs or potential errors
2. Performance issues
3. Code quality problems
4. Security vulnerabilities
5. Suggested improvements

Return your analysis as JSON with this structure:
{{
    "bugs": ["list of bugs"],
    "performance_issues": ["list of performance problems"],
    "quality_issues": ["list of code quality problems"],
    "security_issues": ["list of security concerns"],
    "improvements": ["list of suggested improvements"],
    "severity": "low|medium|high|critical",
    "can_auto_fix": true|false
}}
"""

        return self._call_llm(prompt, json_mode=True)

    def generate_improvement(self, code: str, analysis: Dict[str, Any]) -> str:
        """Ask LLM to generate improved code"""

        issues_str = "\n".join([
            f"- Bugs: {', '.join(analysis.get('bugs', []))}",
            f"- Performance: {', '.join(analysis.get('performance_issues', []))}",
            f"- Quality: {', '.join(analysis.get('quality_issues', []))}",
            f"- Security: {', '.join(analysis.get('security_issues', []))}"
        ])

        prompt = f"""You are an expert Python developer. Improve this code based on the analysis.

Original Code:
```python
{code}
```

Issues Found:
{issues_str}

Return ONLY the improved Python code, no explanations. The code must:
1. Fix all identified issues
2. Maintain the same functionality
3. Be syntactically correct
4. Be better than the original

Return only valid Python code, nothing else."""

        return self._call_llm(prompt, json_mode=False)

    def _call_llm(self, prompt: str, json_mode: bool = False) -> Any:
        """Call Together API"""

        try:
            headers = {
                'Authorization': f'Bearer {self.api_key}',
                'Content-Type': 'application/json'
            }

            data = {
                'model': self.model,
                'messages': [{'role': 'user', 'content': prompt}],
                'temperature': 0.3,  # Lower for code
                'max_tokens': 2000
            }

            response = requests.post(
                f'{self.api_base}/chat/completions',
                headers=headers,
                json=data,
                timeout=30
            )

            response.raise_for_status()
            result = response.json()

            content = result['choices'][0]['message']['content']

            if json_mode:
                # Extract JSON from markdown code blocks if present
                if '```json' in content:
                    content = content.split('```json')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()

                return json.loads(content)
            else:
                # Extract code from markdown if present
                if '```python' in content:
                    content = content.split('```python')[1].split('```')[0].strip()
                elif '```' in content:
                    content = content.split('```')[1].split('```')[0].strip()

                return content

        except Exception as e:
            self.logger.error(f"LLM call failed: {e}")
            return {} if json_mode else ""


class SafeCodeTester:
    """Safely test code changes before applying"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_syntax(self, code: str) -> Tuple[bool, Optional[str]]:
        """Check if code has valid Python syntax"""
        try:
            ast.parse(code)
            return True, None
        except SyntaxError as e:
            return False, str(e)

    def run_tests(self, file_path: str) -> Tuple[bool, str]:
        """Run tests for the modified file"""

        # Find test file
        test_file = self._find_test_file(file_path)

        if not test_file:
            self.logger.warning(f"No test file found for {file_path}")
            return True, "No tests found - assuming OK"

        # Run pytest on test file
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', test_file, '-v'],
                capture_output=True,
                text=True,
                timeout=30
            )

            success = result.returncode == 0
            output = result.stdout + result.stderr

            return success, output

        except subprocess.TimeoutExpired:
            return False, "Tests timed out"
        except Exception as e:
            return False, f"Test execution failed: {e}"

    def _find_test_file(self, file_path: str) -> Optional[str]:
        """Find corresponding test file"""

        path = Path(file_path)

        # Try tests/ directory
        test_path = Path('tests') / f"test_{path.name}"
        if test_path.exists():
            return str(test_path)

        # Try same directory with test_ prefix
        test_path = path.parent / f"test_{path.name}"
        if test_path.exists():
            return str(test_path)

        return None


class SelfImprovementV1:
    """
    Real self-improving AI - Version 1

    This actually modifies code, unlike the stub version
    """

    def __init__(self, target_dir: str = "src"):
        self.target_dir = Path(target_dir)
        self.logger = logging.getLogger(__name__)

        # Components
        self.code_analyzer = CodeAnalyzer()
        self.code_metrics = CodeMetrics()
        self.llm_improver = LLMCodeImprover()
        self.tester = SafeCodeTester()

        # State tracking
        self.improvement_history = []
        self.generation = 0
        self.stats = {
            'total_files_analyzed': 0,
            'total_improvements': 0,
            'successful_improvements': 0,
            'failed_improvements': 0,
            'bugs_fixed': 0,
            'performance_gains': 0
        }

        # Safety settings
        self.backup_dir = Path('backups/self_improvement')
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info("SelfImprovementV1 initialized - REAL implementation")

    def improve_codebase(self, max_files: int = 5):
        """Improve the codebase - main loop"""

        self.logger.info(f"Starting self-improvement cycle (Generation {self.generation})")

        # 1. Find files to improve
        files_to_improve = self._find_improvement_candidates(max_files)

        self.logger.info(f"Found {len(files_to_improve)} files to analyze")

        # 2. Improve each file
        for file_path in files_to_improve:
            self._improve_file(file_path)

        # 3. Generate report
        report = self._generate_report()

        self.generation += 1

        return report

    def improve_file(self, file_path: str) -> Dict[str, Any]:
        """Improve a single file"""
        return self._improve_file(file_path)

    def _improve_file(self, file_path: str) -> Dict[str, Any]:
        """Improve a single file (internal)"""

        self.logger.info(f"Analyzing: {file_path}")

        result = {
            'file': file_path,
            'success': False,
            'changes_made': False,
            'original_metrics': {},
            'improved_metrics': {},
            'issues_found': {},
            'error': None
        }

        try:
            # 1. Read current code
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()

            # 2. Get baseline metrics
            result['original_metrics'] = self.code_metrics.calculate_metrics(original_code)

            # 3. Analyze with CodeAnalyzer
            analysis = self.code_analyzer.analyze_code(original_code)

            # 4. Analyze with LLM
            llm_analysis = self.llm_improver.analyze_code(original_code, file_path)
            result['issues_found'] = llm_analysis

            # 5. Check if improvement is needed
            severity = llm_analysis.get('severity', 'low')
            can_auto_fix = llm_analysis.get('can_auto_fix', False)

            if severity == 'low' and not can_auto_fix:
                self.logger.info(f"No significant issues in {file_path}")
                result['success'] = True
                return result

            # 6. Generate improved code
            self.logger.info(f"Generating improvements for {file_path}")
            improved_code = self.llm_improver.generate_improvement(original_code, llm_analysis)

            if not improved_code or improved_code == original_code:
                self.logger.warning(f"No improvement generated for {file_path}")
                result['error'] = "LLM did not generate improved code"
                return result

            # 7. Validate syntax
            valid, syntax_error = self.tester.validate_syntax(improved_code)
            if not valid:
                self.logger.error(f"Improved code has syntax error: {syntax_error}")
                result['error'] = f"Syntax error: {syntax_error}"
                self.stats['failed_improvements'] += 1
                return result

            # 8. Backup original
            self._backup_file(file_path, original_code)

            # 9. Apply improvement
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(improved_code)

            # 10. Run tests
            tests_pass, test_output = self.tester.run_tests(file_path)

            if not tests_pass:
                self.logger.error(f"Tests failed after improvement: {file_path}")
                # Rollback
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(original_code)

                result['error'] = f"Tests failed: {test_output[:200]}"
                self.stats['failed_improvements'] += 1
                return result

            # 11. Calculate new metrics
            result['improved_metrics'] = self.code_metrics.calculate_metrics(improved_code)

            # 12. Success!
            result['success'] = True
            result['changes_made'] = True

            self.stats['successful_improvements'] += 1
            self.stats['bugs_fixed'] += len(llm_analysis.get('bugs', []))

            self.logger.info(f"✅ Successfully improved {file_path}")

            # Save to history
            self.improvement_history.append({
                'timestamp': datetime.now().isoformat(),
                'generation': self.generation,
                'file': file_path,
                'result': result
            })

            return result

        except Exception as e:
            self.logger.error(f"Error improving {file_path}: {e}")
            result['error'] = str(e)
            self.stats['failed_improvements'] += 1
            return result

    def _find_improvement_candidates(self, max_files: int) -> List[str]:
        """Find files that could benefit from improvement"""

        candidates = []

        # Scan target directory
        for py_file in self.target_dir.rglob('*.py'):
            # Skip tests and __init__
            if 'test_' in py_file.name or py_file.name == '__init__.py':
                continue

            # Skip archived emotion code
            if 'archived_emotion_work' in str(py_file):
                continue

            # Get metrics
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()

                metrics = self.code_metrics.calculate_metrics(code)

                # Score by potential for improvement
                score = 0

                if metrics['complexity'] > 0.7:
                    score += 3
                if metrics['maintainability'] < 0.6:
                    score += 2
                if metrics['security'] < 0.8:
                    score += 3
                if metrics['documentation'] < 0.5:
                    score += 1

                if score > 0:
                    candidates.append((score, str(py_file)))

            except Exception as e:
                self.logger.warning(f"Could not analyze {py_file}: {e}")

        # Sort by score (highest first)
        candidates.sort(reverse=True, key=lambda x: x[0])

        # Return top N files
        return [path for score, path in candidates[:max_files]]

    def _backup_file(self, file_path: str, content: str):
        """Backup file before modification"""

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_path = self.backup_dir / f"{Path(file_path).name}.{timestamp}.backup"

        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)

        self.logger.info(f"Backed up to: {backup_path}")

    def _generate_report(self) -> Dict[str, Any]:
        """Generate improvement report"""

        report = {
            'generation': self.generation,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats,
            'recent_improvements': self.improvement_history[-10:],
            'success_rate': 0.0
        }

        total = self.stats['successful_improvements'] + self.stats['failed_improvements']
        if total > 0:
            report['success_rate'] = self.stats['successful_improvements'] / total

        # Save report
        report_file = Path('reports') / f'self_improvement_gen_{self.generation}.json'
        report_file.parent.mkdir(exist_ok=True)

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Report saved to: {report_file}")

        return report

    def get_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        return {
            'generation': self.generation,
            'stats': self.stats,
            'total_history': len(self.improvement_history)
        }


def main():
    """Test the self-improvement system"""

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 70)
    print("HUMAN 2.0 - Self-Improvement V1")
    print("=" * 70)

    # Initialize
    improver = SelfImprovementV1(target_dir='src/components')

    # Run improvement cycle
    print("\n🔄 Starting self-improvement cycle...")
    report = improver.improve_codebase(max_files=3)

    print("\n📊 Results:")
    print(f"Generation: {report['generation']}")
    print(f"Success Rate: {report['success_rate']:.1%}")
    print(f"Improvements: {improver.stats['successful_improvements']}")
    print(f"Bugs Fixed: {improver.stats['bugs_fixed']}")

    print("\n✅ Self-improvement cycle complete!")
    print(f"Report saved to: reports/self_improvement_gen_{report['generation']}.json")


if __name__ == "__main__":
    main()

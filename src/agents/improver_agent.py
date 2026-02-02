#!/usr/bin/env python3
"""
HUMAN 2.0 - Improver Agent
Generates and applies code improvements using Claude 3.5 Sonnet.

Responsibilities:
- Generate code improvements based on analysis
- Apply refactoring strategies
- Fix bugs and optimize performance
- Validate improvements before applying
- Use pattern library for successful strategies
"""

import ast
import os
import logging
import anthropic
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from agents.base_agent import BaseAgent, AgentStatus
from core.event_bus import EventBus, Event, EventTypes, EventPriority
from core.shared_resources import SharedResources, Pattern


@dataclass
class Improvement:
    """Proposed code improvement"""
    file_path: str
    improvement_type: str  # "refactoring", "bug_fix", "performance", "testing"
    description: str
    original_code: str
    improved_code: str
    rationale: str
    estimated_impact: float  # 0-1
    pattern_id: Optional[str] = None
    timestamp: datetime = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


class ImproverAgent(BaseAgent):
    """
    Agent responsible for generating and applying code improvements.

    Subscribes to:
    - code_analyzed: Generate improvements based on analysis
    - strategy_optimized: Update improvement strategies

    Publishes:
    - improvement_proposed: Improvement ready for review
    - improvement_applied: Improvement successfully applied
    - improvement_failed: Improvement failed
    """

    def __init__(self, name: str, event_bus: EventBus, resources: SharedResources):
        """
        Initialize Improver Agent.

        Args:
            name: Agent name
            event_bus: Event bus for communication
            resources: Shared resources
        """
        super().__init__(name, event_bus)
        self.resources = resources

        # Initialize Claude API
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        if not self.api_key:
            self.logger.warning("ANTHROPIC_API_KEY not set - improvements will fail")
            self.client = None
        else:
            self.client = anthropic.Anthropic(api_key=self.api_key)

        # Configuration
        self.config = {
            'model': 'claude-sonnet-4-5-20250929',
            'max_tokens': 4000,
            'timeout': 300,
            'temperature': 0.7,
            'min_priority_for_auto_apply': 0.5  # Auto-apply improvements above this priority
        }

        self.logger.info(f"ImproverAgent initialized with model: {self.config['model']}")

    def register_event_handlers(self):
        """Register event handlers"""
        self.event_bus.subscribe(EventTypes.CODE_ANALYZED, self.on_code_analyzed, self.name)
        self.event_bus.subscribe(EventTypes.STRATEGY_OPTIMIZED, self.on_strategy_optimized, self.name)
        self.logger.info(f"Subscribed to: {EventTypes.CODE_ANALYZED}, {EventTypes.STRATEGY_OPTIMIZED}")

    async def on_code_analyzed(self, event: Event):
        """Handle code analyzed event"""
        analysis = event.data.get('analysis')
        file_path = event.data.get('file_path')

        self.logger.info(f"Received analysis for: {file_path}")

        # Only improve if priority is high enough or there are specific issues
        priority = analysis.get('priority_score', 0)
        if priority < 0.3 and not analysis.get('code_smells'):
            self.logger.debug(f"Skipping {file_path} - low priority ({priority:.2f})")
            return

        # Create improvement task
        task = {
            'type': 'improve',
            'file_path': file_path,
            'analysis': analysis
        }
        result = await self.execute_task(task)

    async def on_strategy_optimized(self, event: Event):
        """Handle strategy optimized event"""
        strategy = event.data.get('strategy')
        self.logger.info(f"Strategy optimized: {strategy}")
        # Update config with new strategy parameters
        if 'parameters' in event.data:
            self.config.update(event.data['parameters'])

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main processing: Generate and apply code improvement.

        Args:
            task: Task with file_path and analysis

        Returns:
            Improvement result
        """
        file_path = task['file_path']
        analysis = task['analysis']

        self.logger.info(f"Generating improvement for: {file_path}")

        # Read original code
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
        except Exception as e:
            raise ValueError(f"Cannot read file {file_path}: {e}")

        # Check model cache first
        cache_key = f"{file_path}_{analysis.get('timestamp')}"
        cached_improvement = self.resources.get_cached_response(cache_key)
        if cached_improvement:
            self.logger.info("Using cached improvement")
            improved_code = cached_improvement
        else:
            # Generate improvement using Claude
            improved_code = await self._generate_improvement(
                file_path=file_path,
                original_code=original_code,
                analysis=analysis
            )

            # Cache the result
            self.resources.cache_response(cache_key, improved_code)

        # Validate improvement with retry for syntax errors
        if not self._validate_improvement(original_code, improved_code):
            # Check if it's specifically a syntax error
            syntax_valid, syntax_error = self._check_syntax(improved_code)
            if not syntax_valid:
                self.logger.warning(f"Syntax error detected, attempting fix: {syntax_error}")
                # Retry with simpler prompt to fix syntax
                improved_code = await self._fix_syntax_errors(improved_code, syntax_error)

                # Validate again after fix attempt
                if not self._validate_improvement(original_code, improved_code):
                    raise ValueError("Generated improvement failed validation even after syntax fix attempt")
            else:
                raise ValueError("Generated improvement failed validation")

        # Create improvement object
        improvement = Improvement(
            file_path=file_path,
            improvement_type=self._determine_improvement_type(analysis),
            description=self._create_description(analysis),
            original_code=original_code,
            improved_code=improved_code,
            rationale=self._create_rationale(analysis),
            estimated_impact=analysis.get('priority_score', 0.5)
        )

        # Publish improvement proposed event
        await self.publish_event(
            EventTypes.IMPROVEMENT_PROPOSED,
            {
                'improvement': asdict(improvement),
                'analysis': analysis
            },
            EventPriority.HIGH if improvement.estimated_impact > 0.7 else EventPriority.NORMAL
        )

        # Auto-apply if priority is high enough
        if improvement.estimated_impact >= self.config['min_priority_for_auto_apply']:
            success = await self._apply_improvement(improvement)

            if success:
                await self.publish_event(
                    EventTypes.IMPROVEMENT_APPLIED,
                    {
                        'improvement': asdict(improvement),
                        'file_path': file_path
                    },
                    EventPriority.NORMAL
                )
                return {
                    'success': True,
                    'improvement': asdict(improvement),
                    'applied': True
                }
            else:
                await self.publish_event(
                    EventTypes.IMPROVEMENT_FAILED,
                    {
                        'improvement': asdict(improvement),
                        'reason': 'Application failed'
                    },
                    EventPriority.HIGH
                )
                return {
                    'success': False,
                    'improvement': asdict(improvement),
                    'applied': False,
                    'error': 'Application failed'
                }
        else:
            # Just propose, don't apply
            return {
                'success': True,
                'improvement': asdict(improvement),
                'applied': False,
                'reason': 'Priority too low for auto-apply'
            }

    async def _generate_improvement_multi_step(self, file_path: str, original_code: str,
                                              analysis: Dict[str, Any]) -> str:
        """
        PHASE 2: Multi-step reasoning for complex improvements.
        Chains multiple LLM calls for deeper analysis.

        Steps:
        1. Understand the code deeply
        2. Generate multiple approaches
        3. Evaluate trade-offs
        4. Implement best approach
        5. Refine

        Args:
            file_path: Path to file
            original_code: Original code
            analysis: Analysis results

        Returns:
            Improved code
        """
        if not self.client:
            # Fallback to single-step
            return await self._generate_improvement(file_path, original_code, analysis)

        try:
            # Step 1: Deep understanding
            understanding = self.client.messages.create(
                model=self.config['model'],
                max_tokens=1500,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"""Analyze this code deeply:

FILE: {file_path}

CODE:
```python
{original_code[:2000]}
```

Provide:
1. What does this code do? (purpose)
2. What are the main problems? (issues)
3. What are possible improvement strategies? (3-5 approaches)

Be specific and concise."""
                }]
            )

            understanding_text = understanding.content[0].text

            # Step 2: Evaluate approaches
            evaluation = self.client.messages.create(
                model=self.config['model'],
                max_tokens=1000,
                temperature=0.3,
                messages=[{
                    "role": "user",
                    "content": f"""Based on this analysis:

{understanding_text}

Which improvement approach is best? Consider:
- Impact on code quality
- Risk of breaking changes
- Maintainability
- Simplicity

Choose ONE approach and explain why."""
                }]
            )

            chosen_approach = evaluation.content[0].text

            # Step 3: Implement
            implementation = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens'],
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": f"""Implement this improvement:

CHOSEN APPROACH:
{chosen_approach}

ORIGINAL CODE:
```python
{original_code}
```

Return ONLY the improved Python code, no explanations."""
                }]
            )

            improved_code = implementation.content[0].text

            # Extract code
            if "```python" in improved_code:
                improved_code = improved_code.split("```python")[1].split("```")[0].strip()
            elif "```" in improved_code:
                improved_code = improved_code.split("```")[1].split("```")[0].strip()

            self.logger.info(f"Multi-step improvement complete for {file_path}")
            return improved_code

        except Exception as e:
            self.logger.error(f"Multi-step reasoning failed: {e}, falling back to single-step")
            return await self._generate_improvement(file_path, original_code, analysis)

    async def _generate_improvement(self, file_path: str, original_code: str,
                                   analysis: Dict[str, Any]) -> str:
        """
        Generate code improvement using Claude 3.5 Sonnet (single-step).

        Args:
            file_path: Path to file
            original_code: Original code
            analysis: Analysis results

        Returns:
            Improved code
        """
        # Build context from similar code
        similar_code = self.resources.search_similar_code(
            query=f"Code from {file_path}",
            n_results=2
        )

        # Get relevant patterns from pattern library
        improvement_type = self._determine_improvement_type(analysis)
        patterns = self.resources.search_patterns(
            category=improvement_type,
            min_success_rate=0.6
        )

        # Build improvement prompt
        prompt = self._build_improvement_prompt(
            file_path=file_path,
            original_code=original_code,
            analysis=analysis,
            similar_code=similar_code,
            patterns=patterns[:2]  # Top 2 patterns
        )

        # Call Claude API
        try:
            if not self.client:
                raise ValueError("Claude API client not initialized - ANTHROPIC_API_KEY not set")

            self.logger.info(f"Calling Claude API for {file_path}")

            message = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens'],
                temperature=self.config['temperature'],
                timeout=self.config['timeout'],
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            improved_code = message.content[0].text

            # Extract code from markdown if present
            if "```python" in improved_code:
                improved_code = improved_code.split("```python")[1].split("```")[0].strip()
            elif "```" in improved_code:
                improved_code = improved_code.split("```")[1].split("```")[0].strip()

            return improved_code

        except Exception as e:
            self.logger.error(f"Claude API error: {e}")
            raise ValueError(f"Failed to generate improvement: {e}")

    async def _generate_improvements_batch(self, files_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate improvements for multiple files in a single API call.
        PHASE 1 ENHANCEMENT: Batched API calls for higher throughput.

        Args:
            files_data: List of dicts with keys: file_path, original_code, analysis

        Returns:
            Dict mapping file_path to improved_code
        """
        if not files_data:
            return {}

        if len(files_data) == 1:
            # Single file - use regular method
            file_data = files_data[0]
            improved_code = await self._generate_improvement(
                file_data['file_path'],
                file_data['original_code'],
                file_data['analysis']
            )
            return {file_data['file_path']: improved_code}

        # Build batch prompt
        prompt = self._build_batch_improvement_prompt(files_data)

        try:
            if not self.client:
                raise ValueError("Claude API client not initialized")

            self.logger.info(f"Calling Claude API for {len(files_data)} files in batch")

            message = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens'] * min(len(files_data), 3),  # Scale tokens
                temperature=self.config['temperature'],
                timeout=self.config['timeout'] * 2,  # More time for batch
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            response_text = message.content[0].text

            # Parse response to extract improvements for each file
            improvements = self._parse_batch_response(response_text, files_data)

            self.logger.info(f"Batch API call successful: {len(improvements)} files improved")
            return improvements

        except Exception as e:
            self.logger.error(f"Batch Claude API error: {e}, falling back to individual calls")
            # Fallback to individual calls
            improvements = {}
            for file_data in files_data:
                try:
                    improved_code = await self._generate_improvement(
                        file_data['file_path'],
                        file_data['original_code'],
                        file_data['analysis']
                    )
                    improvements[file_data['file_path']] = improved_code
                except Exception as e2:
                    self.logger.error(f"Failed to improve {file_data['file_path']}: {e2}")
            return improvements

    def _build_batch_improvement_prompt(self, files_data: List[Dict[str, Any]]) -> str:
        """Build a batch improvement prompt for multiple files"""

        prompt = f"""You are an expert Python code improver. Your task is to improve {len(files_data)} files.

For each file, I will provide:
- File path
- Current code
- Analysis results
- Issues identified

Please improve all files and return the improved code for each file in this format:

=== FILE: <file_path> ===
<improved code here>
=== END FILE ===

---

"""

        for i, file_data in enumerate(files_data, 1):
            file_path = file_data['file_path']
            original_code = file_data['original_code']
            analysis = file_data['analysis']

            prompt += f"""
FILE {i}/{len(files_data)}: {file_path}

ANALYSIS:
- Complexity: {analysis.get('complexity', 0)}
- Maintainability: {analysis.get('maintainability_index', 0):.1f}/100
- Priority Score: {analysis.get('priority_score', 0):.2f}

ISSUES:
"""
            code_smells = analysis.get('code_smells', [])
            if code_smells:
                for smell in code_smells[:3]:
                    prompt += f"- {smell['smell_type']}: {smell['description']}\n"

            prompt += f"""
ORIGINAL CODE:
```python
{original_code}
```

---

"""

        prompt += """
REQUIREMENTS:
- Improve all files
- Fix identified issues
- Reduce complexity
- Preserve functionality
- Return improvements in the specified format (=== FILE: ... ===)
- Ensure code is syntactically correct
- DO NOT add docstrings unless already present

Begin improvements:
"""

        return prompt

    def _parse_batch_response(self, response_text: str, files_data: List[Dict[str, Any]]) -> Dict[str, str]:
        """Parse batched response to extract improvements for each file"""

        improvements = {}

        # Split by file markers
        file_sections = response_text.split('=== FILE:')

        for section in file_sections[1:]:  # Skip first empty split
            try:
                # Extract file path
                header, rest = section.split('===', 1)
                file_path = header.strip()

                # Extract code
                if '=== END FILE ===' in rest:
                    code = rest.split('=== END FILE ===')[0].strip()
                else:
                    code = rest.strip()

                # Clean up code blocks
                if '```python' in code:
                    code = code.split('```python')[1].split('```')[0].strip()
                elif '```' in code:
                    code = code.split('```')[1].split('```')[0].strip()

                # Find matching file path (may not be exact match)
                matched_file = None
                for file_data in files_data:
                    if file_data['file_path'] in file_path or file_path in file_data['file_path']:
                        matched_file = file_data['file_path']
                        break

                if matched_file:
                    improvements[matched_file] = code
                    self.logger.info(f"Extracted improvement for {matched_file}")

            except Exception as e:
                self.logger.error(f"Failed to parse batch response section: {e}")

        return improvements

    def _build_improvement_prompt(self, file_path: str, original_code: str,
                                 analysis: Dict[str, Any],
                                 similar_code: List[Dict[str, Any]],
                                 patterns: List[Pattern]) -> str:
        """Build improvement prompt for Claude"""

        prompt = f"""You are an expert Python code improver. Your task is to improve the following code.

FILE: {file_path}

ANALYSIS RESULTS:
- Lines of Code: {analysis.get('lines_of_code', 0)}
- Complexity: {analysis.get('complexity', 0)}
- Maintainability: {analysis.get('maintainability_index', 0):.1f}/100
- Priority Score: {analysis.get('priority_score', 0):.2f}

ISSUES IDENTIFIED:
"""

        # Add code smells
        code_smells = analysis.get('code_smells', [])
        if code_smells:
            prompt += "\nCode Smells:\n"
            for smell in code_smells[:5]:  # Top 5
                prompt += f"- {smell['smell_type']}: {smell['description']} (Line {smell['location']})\n"
                prompt += f"  Suggestion: {smell['suggestion']}\n"

        # Add refactoring opportunities
        refactoring = analysis.get('refactoring_opportunities', [])
        if refactoring:
            prompt += "\nRefactoring Opportunities:\n"
            for opp in refactoring[:3]:  # Top 3
                prompt += f"- {opp}\n"

        # Add patterns
        if patterns:
            prompt += "\nSUCCESSFUL PATTERNS TO CONSIDER:\n"
            for pattern in patterns:
                prompt += f"\n{pattern.name} (Success Rate: {pattern.success_rate:.1%}):\n"
                prompt += f"{pattern.description}\n"

        # Add similar code context
        if similar_code:
            prompt += "\nSIMILAR CODE IN CODEBASE:\n"
            for idx, similar in enumerate(similar_code[:2], 1):
                prompt += f"\nExample {idx} from {similar.get('metadata', {}).get('file_path', 'unknown')}:\n"
                prompt += f"```python\n{similar.get('document', '')[:500]}...\n```\n"

        prompt += f"""

ORIGINAL CODE:
```python
{original_code}
```

TASK:
Improve this code by:
1. Fixing identified code smells
2. Reducing complexity where possible
3. Improving maintainability
4. Following best practices
5. Maintaining or improving functionality

REQUIREMENTS:
- Return ONLY the improved Python code, no explanations
- Preserve all functionality
- Maintain the same module structure and API
- Ensure all imports are correct
- Make sure the code is syntactically correct
- DO NOT add docstrings or type hints unless they were already present
- Focus on the specific issues identified above

Return the complete improved code:
"""

        return prompt

    def _validate_improvement(self, original_code: str, improved_code: str) -> bool:
        """
        Validate that improved code is valid and safe.

        Args:
            original_code: Original code
            improved_code: Improved code

        Returns:
            True if valid
        """
        # 1. Syntax check
        try:
            ast.parse(improved_code)
        except SyntaxError as e:
            self.logger.error(f"Improved code has syntax error: {e}")
            return False

        # 2. Not empty
        if not improved_code.strip():
            self.logger.error("Improved code is empty")
            return False

        # 3. Similar length (shouldn't be drastically different)
        # RELAXED: Allow 20% to 500% of original length
        original_lines = len(original_code.split('\n'))
        improved_lines = len(improved_code.split('\n'))

        if improved_lines < original_lines * 0.2:
            self.logger.error(f"Improved code too short ({improved_lines} vs {original_lines} lines)")
            return False

        if improved_lines > original_lines * 5:
            self.logger.error(f"Improved code too long ({improved_lines} vs {original_lines} lines)")
            return False

        # 4. Contains key elements from original (basic check)
        # Extract function/class names from original
        try:
            original_tree = ast.parse(original_code)
            improved_tree = ast.parse(improved_code)

            original_names = {
                node.name for node in ast.walk(original_tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }
            improved_names = {
                node.name for node in ast.walk(improved_tree)
                if isinstance(node, (ast.FunctionDef, ast.ClassDef))
            }

            # Should have most of the same names
            # RELAXED: Allow 50% instead of 70% (more aggressive refactoring allowed)
            if original_names and len(improved_names & original_names) / len(original_names) < 0.5:
                self.logger.warning(f"Many functions/classes missing in improved code")
                # Don't fail, just warn

        except:
            pass  # Skip this check if parsing fails

        return True

    def _check_syntax(self, code: str) -> tuple:
        """
        Check if code has valid Python syntax.

        Returns:
            Tuple of (is_valid: bool, error_message: str)
        """
        try:
            ast.parse(code)
            return True, ""
        except SyntaxError as e:
            error_msg = f"Line {e.lineno}: {e.msg}"
            return False, error_msg

    async def _fix_syntax_errors(self, broken_code: str, syntax_error: str) -> str:
        """
        Attempt to fix syntax errors with a simple, focused prompt.
        This uses fewer tokens and is more likely to succeed.

        Args:
            broken_code: Code with syntax errors
            syntax_error: The specific syntax error message

        Returns:
            Fixed code (or original if fix fails)
        """
        if not self.client:
            return broken_code

        simple_prompt = f"""Fix ONLY the syntax error in this Python code. Do NOT refactor or change logic.

SYNTAX ERROR: {syntax_error}

CODE:
```python
{broken_code}
```

Return ONLY the corrected Python code with the syntax error fixed. No explanations."""

        try:
            self.logger.info("Attempting syntax fix with simplified prompt...")
            message = self.client.messages.create(
                model=self.config['model'],
                max_tokens=self.config['max_tokens'],
                temperature=0.1,  # Lower temperature for more deterministic fix
                timeout=60,  # Shorter timeout for simple fix
                messages=[{"role": "user", "content": simple_prompt}]
            )

            fixed_code = message.content[0].text

            # Extract code from markdown if present
            if "```python" in fixed_code:
                fixed_code = fixed_code.split("```python")[1].split("```")[0].strip()
            elif "```" in fixed_code:
                fixed_code = fixed_code.split("```")[1].split("```")[0].strip()

            # Verify the fix worked
            is_valid, _ = self._check_syntax(fixed_code)
            if is_valid:
                self.logger.info("Syntax fix successful!")
                return fixed_code
            else:
                self.logger.warning("Syntax fix attempt failed, returning original")
                return broken_code

        except Exception as e:
            self.logger.error(f"Syntax fix API call failed: {e}")
            return broken_code

    async def _apply_improvement(self, improvement: Improvement) -> bool:
        """
        Apply improvement to file.

        Args:
            improvement: Improvement to apply

        Returns:
            True if successful
        """
        try:
            # Backup original file
            backup_path = Path(improvement.file_path).with_suffix('.py.backup')
            with open(improvement.file_path, 'r', encoding='utf-8') as f:
                original = f.read()
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original)

            # Write improved code
            with open(improvement.file_path, 'w', encoding='utf-8') as f:
                f.write(improvement.improved_code)

            self.logger.info(f"Applied improvement to {improvement.file_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply improvement: {e}")
            # Restore from backup if it exists
            if backup_path.exists():
                with open(backup_path, 'r', encoding='utf-8') as f:
                    original = f.read()
                with open(improvement.file_path, 'w', encoding='utf-8') as f:
                    f.write(original)
            return False

    def _determine_improvement_type(self, analysis: Dict[str, Any]) -> str:
        """Determine improvement type from analysis"""
        if analysis.get('complexity', 0) > 15:
            return "refactoring"
        if analysis.get('code_smells'):
            return "refactoring"
        if "performance" in str(analysis.get('refactoring_opportunities', [])).lower():
            return "performance"
        if "test" in str(analysis.get('refactoring_opportunities', [])).lower():
            return "testing"
        return "refactoring"

    def _create_description(self, analysis: Dict[str, Any]) -> str:
        """Create improvement description"""
        issues = []
        if analysis.get('complexity', 0) > 10:
            issues.append(f"High complexity ({analysis['complexity']})")
        if analysis.get('maintainability_index', 100) < 65:
            issues.append(f"Low maintainability ({analysis['maintainability_index']:.1f})")
        if analysis.get('code_smells'):
            issues.append(f"{len(analysis['code_smells'])} code smells")

        return "Improve: " + ", ".join(issues) if issues else "General code improvement"

    def _create_rationale(self, analysis: Dict[str, Any]) -> str:
        """Create improvement rationale"""
        rationale = "This improvement addresses:\n"

        if analysis.get('code_smells'):
            rationale += f"- {len(analysis['code_smells'])} code smell(s)\n"

        if analysis.get('complexity', 0) > 10:
            rationale += f"- High complexity ({analysis['complexity']})\n"

        if analysis.get('maintainability_index', 100) < 65:
            rationale += f"- Low maintainability ({analysis['maintainability_index']:.1f}/100)\n"

        for opp in analysis.get('refactoring_opportunities', [])[:3]:
            rationale += f"- {opp}\n"

        return rationale

    def validate_output(self, output: Any) -> bool:
        """Validate improvement output"""
        if not isinstance(output, dict):
            return False

        if not output.get('success') and 'error' not in output:
            return False

        if output.get('success') and 'improvement' not in output:
            return False

        return True


if __name__ == "__main__":
    # Test improver agent
    import asyncio
    logging.basicConfig(level=logging.INFO)

    from core.event_bus import EventBus, create_event
    from core.shared_resources import SharedResources

    async def test_improver():
        """Test improver agent"""
        # Setup
        bus = EventBus()
        resources = SharedResources()

        # Build dependency graph
        resources.dependency_analyzer.build_graph(['src'])

        agent = ImproverAgent("improver", bus, resources)

        # Create mock analysis
        analysis = {
            'file_path': 'src/agents/base_agent.py',
            'timestamp': datetime.now().isoformat(),
            'lines_of_code': 200,
            'complexity': 12,
            'maintainability_index': 60,
            'priority_score': 0.6,
            'code_smells': [
                {
                    'smell_type': 'long_function',
                    'severity': 'medium',
                    'location': 'line 116',
                    'description': 'Function execute_task has 50 lines',
                    'suggestion': 'Break down into smaller functions'
                }
            ],
            'refactoring_opportunities': [
                'High complexity (12) - Consider refactoring'
            ]
        }

        # Test improvement generation (will fail if no API key)
        if os.getenv('ANTHROPIC_API_KEY'):
            task = {
                'type': 'improve',
                'file_path': 'src/agents/base_agent.py',
                'analysis': analysis
            }

            try:
                result = await agent.execute_task(task)
                print(f"\nImprovement Result:")
                print(f"  Success: {result.get('success')}")
                print(f"  Applied: {result.get('applied')}")
                if result.get('success'):
                    improvement = result['improvement']
                    print(f"  Type: {improvement['improvement_type']}")
                    print(f"  Impact: {improvement['estimated_impact']:.2f}")
                    print(f"  Description: {improvement['description']}")
            except Exception as e:
                print(f"Error: {e}")
        else:
            print("ANTHROPIC_API_KEY not set - skipping API test")

        # Test event subscription
        await bus.publish(create_event(
            EventTypes.CODE_ANALYZED,
            {'file_path': 'src/agents/base_agent.py', 'analysis': analysis},
            'analyzer'
        ))

        await asyncio.sleep(0.5)

        # Get status
        status = agent.get_status()
        print(f"\nAgent Status:")
        print(f"  Name: {status['name']}")
        print(f"  Status: {status['status']}")
        print(f"  Tasks Processed: {status['metrics']['tasks_processed']}")

    asyncio.run(test_improver())

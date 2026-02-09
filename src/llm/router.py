"""
HUMAN 2.0 - LLM Router
Intelligently routes requests between local and cloud LLMs.

Hybrid Architecture:
- Local (FREE): Ollama + DeepSeek-Coder 6.7B for simple tasks
- Cloud (Paid): Claude API for complex tasks

Routing Logic:
- Simple tasks (complexity < 10, lines < 200) -> Local LLM (FREE)
- Complex tasks (complexity >= 10 or lines >= 200) -> Claude API

Features:
- Automatic fallback if local unavailable
- Cost optimization (80%+ local = FREE)
- Quality-based routing
"""

import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from .local_llm_client import LocalLLMClient
from .cloud_llm_client import CloudLLMClient, CloudProvider


class RoutingDecision(Enum):
    """Where to route the request."""
    LOCAL = "local"           # Ollama + DeepSeek (FREE)
    CLOUD = "cloud"           # Claude API (paid)


class TaskType(Enum):
    """Type of task for routing."""
    FULL_REWRITE = "full_rewrite"      # Regenerate entire file
    TARGETED_EDIT = "targeted_edit"    # Fix specific issues only
    SYNTAX_FIX = "syntax_fix"          # Fix syntax errors only


@dataclass
class RoutingConfig:
    """Configuration for routing decisions."""
    # Thresholds for local routing (below these -> FREE local LLM)
    local_max_complexity: int = 10
    local_max_lines: int = 200

    # For targeted edits, local can handle more complex files
    local_targeted_max_complexity: int = 50
    local_targeted_max_lines: int = 800

    # Force Claude API for certain patterns
    force_cloud_patterns: list = None

    # Prefer local even if slower (saves money)
    prefer_local: bool = True

    # Use targeted edits for files with embedded content (HTML, SQL, etc.)
    use_targeted_for_embedded: bool = True

    def __post_init__(self):
        if self.force_cloud_patterns is None:
            self.force_cloud_patterns = [
                "refactor entire",
                "rewrite",
                "redesign",
                "complex algorithm",
                "architectural"
            ]


class LLMRouter:
    """
    Intelligent router for LLM requests.

    Automatically selects the best LLM based on:
    - Task complexity
    - Code size
    - Available resources
    - Cost constraints
    """

    def __init__(
        self,
        local_client: LocalLLMClient = None,
        cloud_client: CloudLLMClient = None,
        config: RoutingConfig = None
    ):
        """
        Initialize LLM Router.

        Args:
            local_client: Local LLM client (optional, created if not provided)
            cloud_client: Cloud LLM client (optional, created if not provided)
            config: Routing configuration
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or RoutingConfig()

        # Initialize clients
        self.local_client = local_client or LocalLLMClient()
        self.cloud_client = cloud_client or CloudLLMClient()

        # Stats tracking
        self.routing_stats = {
            RoutingDecision.LOCAL: 0,
            RoutingDecision.CLOUD: 0,
        }
        self.total_cost_saved = 0.0  # Estimated savings from local routing

        self.logger.info(
            f"LLMRouter initialized. Local available: {self.local_client.available}, "
            f"Cloud available: {self.cloud_client.available}"
        )

    def detect_embedded_content(self, code: str) -> bool:
        """
        Detect if code contains embedded content (HTML, SQL, etc.)
        that makes full regeneration risky.

        Args:
            code: Source code to analyze

        Returns:
            True if embedded content detected
        """
        indicators = [
            '"""<!DOCTYPE',    # Embedded HTML
            "'''<!DOCTYPE",
            '<html',
            '<div',
            '<style>',
            'render_template',
            'f"""<',           # f-string HTML
            "f'''<",
            'CREATE TABLE',    # Embedded SQL
            'INSERT INTO',
            'SELECT * FROM',
        ]

        code_lower = code.lower()
        for indicator in indicators:
            if indicator.lower() in code_lower:
                return True
        return False

    def decide_task_type(
        self,
        code: str,
        complexity: int = 0,
        lines_of_code: int = 0
    ) -> TaskType:
        """
        Decide the best task type for a file.

        Args:
            code: Source code
            complexity: Complexity score
            lines_of_code: Line count

        Returns:
            TaskType indicating how to process
        """
        # Files with embedded content -> targeted edits (safer)
        if self.config.use_targeted_for_embedded and self.detect_embedded_content(code):
            self.logger.info("Embedded content detected -> using targeted edits")
            return TaskType.TARGETED_EDIT

        # Very large or complex files -> targeted edits
        if lines_of_code > 500 or complexity > 30:
            self.logger.info(f"Large/complex file (lines={lines_of_code}, complexity={complexity}) -> targeted edits")
            return TaskType.TARGETED_EDIT

        # Standard files -> full rewrite is OK
        return TaskType.FULL_REWRITE

    def decide_routing(
        self,
        complexity: int = 0,
        lines_of_code: int = 0,
        task_description: str = "",
        force_local: bool = False,
        force_cloud: bool = False,
        task_type: TaskType = None
    ) -> RoutingDecision:
        """
        Decide where to route a request.

        Args:
            complexity: Code complexity score (cyclomatic complexity)
            lines_of_code: Number of lines in the code
            task_description: Description of the task
            force_local: Force local routing
            force_cloud: Force cloud routing
            task_type: Type of task (affects thresholds)

        Returns:
            RoutingDecision indicating where to route
        """
        # Force overrides
        if force_local and self.local_client.available:
            return RoutingDecision.LOCAL

        if force_cloud or not self.local_client.available:
            return RoutingDecision.CLOUD

        # Check for patterns that need cloud (complex tasks)
        task_lower = task_description.lower()
        for pattern in self.config.force_cloud_patterns:
            if pattern in task_lower:
                self.logger.debug(f"Pattern '{pattern}' requires Claude API")
                return RoutingDecision.CLOUD

        # Targeted edits can use local for larger files
        if task_type == TaskType.TARGETED_EDIT:
            if complexity <= self.config.local_targeted_max_complexity:
                if lines_of_code <= self.config.local_targeted_max_lines:
                    self.logger.info("Targeted edit -> routing to LOCAL (extended limits)")
                    return RoutingDecision.LOCAL

        # Full rewrite: standard complexity-based routing
        # Simple tasks -> Local (FREE)
        # Complex tasks -> Claude API (paid)
        if complexity <= self.config.local_max_complexity:
            if lines_of_code <= self.config.local_max_lines:
                return RoutingDecision.LOCAL

        return RoutingDecision.CLOUD

    async def generate(
        self,
        prompt: str,
        complexity: int = 0,
        lines_of_code: int = 0,
        task_description: str = "",
        max_tokens: int = 4000,
        temperature: float = 0.7,
        system: str = None,
        force_local: bool = False,
        force_cloud: bool = False
    ) -> Optional[str]:
        """
        Generate text using the appropriate LLM.

        Args:
            prompt: The prompt to complete
            complexity: Code complexity score
            lines_of_code: Number of lines in the code
            task_description: Description of the task
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system: System prompt
            force_local: Force local routing
            force_cloud: Force cloud routing

        Returns:
            Generated text or None if failed
        """
        # Decide routing
        decision = self.decide_routing(
            complexity=complexity,
            lines_of_code=lines_of_code,
            task_description=task_description,
            force_local=force_local,
            force_cloud=force_cloud
        )

        self.routing_stats[decision] += 1
        self.logger.info(f"Routing decision: {decision.value}")

        # Route to appropriate LLM
        if decision == RoutingDecision.LOCAL:
            result = self.local_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system
            )

            if result:
                # Estimate cost saved (Claude API would cost ~$0.01-0.02 per call)
                estimated_cloud_cost = (len(prompt) + len(result)) / 1000 * 0.01
                self.total_cost_saved += estimated_cloud_cost
                return result

            # Fallback to Claude API if local fails
            self.logger.warning("Local LLM failed, falling back to Claude API")
            decision = RoutingDecision.CLOUD

        # Cloud routing
        return await self.cloud_client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system
        )

    def _extract_function_chunk(self, code: str, issue: str) -> tuple:
        """
        Extract just the function that needs fixing (NOT entire classes).
        Capped at 40 lines max for fast local LLM processing.

        Args:
            code: Full source code
            issue: Issue description (often contains line number)

        Returns:
            Tuple of (chunk_code, start_line, end_line) or (None, 0, 0)
        """
        import ast
        import re

        lines = code.split('\n')

        # Try to extract line number from issue
        line_match = re.search(r'[Ll]ine\s*(\d+)', issue)
        target_line = int(line_match.group(1)) if line_match else None

        if not target_line:
            # No line number - just extract 40 lines from middle
            mid = len(lines) // 2
            start = max(0, mid - 20)
            end = min(len(lines), mid + 20)
            return '\n'.join(lines[start:end]), start + 1, end

        try:
            tree = ast.parse(code)

            # Find the SMALLEST function containing this line (NOT classes)
            best_func = None
            best_size = float('inf')

            for node in ast.walk(tree):
                # ONLY match functions - classes are too big
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if hasattr(node, 'lineno') and hasattr(node, 'end_lineno'):
                        if node.lineno <= target_line <= node.end_lineno:
                            size = node.end_lineno - node.lineno
                            if size < best_size:
                                best_func = node
                                best_size = size

            if best_func and best_size <= 40:
                # Good small function - use it
                start = max(0, best_func.lineno - 2)
                end = min(len(lines), best_func.end_lineno + 2)
                chunk = '\n'.join(lines[start:end])
                return chunk, start + 1, end

        except:
            pass

        # Fallback: extract 40 lines around the target (capped for speed)
        start = max(0, target_line - 20)
        end = min(len(lines), target_line + 20)
        chunk = '\n'.join(lines[start:end])
        return chunk, start + 1, end

    async def generate_targeted_edit(
        self,
        original_code: str,
        issues: list,
        file_path: str = "",
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate targeted edits for specific issues - sends only relevant chunks.
        Much faster and more reliable than sending the entire file.

        Args:
            original_code: Original source code
            issues: List of specific issues to fix (from analysis)
            file_path: Path for context
            max_tokens: Max tokens

        Returns:
            Improved code or None
        """
        lines = original_code.split('\n')
        total_lines = len(lines)

        # For small files, just process normally
        if total_lines <= 150:
            return await self._generate_full_file_edit(original_code, issues, file_path, max_tokens)

        # For large files, extract and fix individual chunks
        self.logger.info(f"Large file ({total_lines} lines) - using chunk-based editing")

        modified_lines = list(lines)  # Copy to modify
        fixes_applied = 0

        for issue in issues[:3]:  # Limit to 3 issues per cycle
            chunk, start_line, end_line = self._extract_function_chunk(original_code, issue)

            if not chunk:
                continue

            self.logger.info(f"Fixing chunk lines {start_line}-{end_line} for: {issue[:50]}...")

            # Create a focused prompt for just this chunk
            prompt = f"""Fix this specific issue in the code chunk below.

ISSUE: {issue}

CODE CHUNK (lines {start_line}-{end_line}):
```python
{chunk}
```

Return ONLY the fixed code chunk. Keep the same structure and indentation.
Do NOT add docstrings, comments, or type hints unless fixing the issue requires it."""

            # Route to local for small chunks
            decision = self.decide_routing(
                complexity=0,
                lines_of_code=len(chunk.split('\n')),
                task_description="chunk fix",
                task_type=TaskType.TARGETED_EDIT
            )

            self.routing_stats[decision] += 1

            if decision == RoutingDecision.LOCAL and self.local_client.available:
                result = self.local_client.generate(
                    prompt=prompt,
                    max_tokens=1000,  # Smaller for chunks
                    temperature=0.2,
                    timeout=60  # Shorter timeout for small chunks
                )
            else:
                result = await self.cloud_client.generate(
                    prompt=prompt,
                    max_tokens=1000,
                    temperature=0.2
                )

            if result:
                fixed_chunk = self._extract_code(result)

                # Validate chunk BEFORE merging
                if not fixed_chunk or len(fixed_chunk.strip()) < 10:
                    self.logger.warning(f"Chunk too short, skipping: {issue[:30]}")
                    continue

                # Check chunk structure matches original roughly
                original_chunk_lines = chunk.split('\n')
                fixed_chunk_lines = fixed_chunk.split('\n')

                # Reject if line count differs too much (>50% change)
                if abs(len(fixed_chunk_lines) - len(original_chunk_lines)) > len(original_chunk_lines) * 0.5:
                    self.logger.warning(f"Chunk size mismatch ({len(original_chunk_lines)} -> {len(fixed_chunk_lines)}), skipping")
                    continue

                # Try to validate syntax by creating a test merge
                test_lines = list(modified_lines)
                for i, line in enumerate(fixed_chunk_lines):
                    line_idx = start_line - 1 + i
                    if line_idx < len(test_lines):
                        test_lines[line_idx] = line
                    elif i < len(original_chunk_lines):
                        test_lines.append(line)

                # Validate merged result before committing
                try:
                    compile('\n'.join(test_lines), '<test_merge>', 'exec')
                except SyntaxError as e:
                    self.logger.warning(f"Chunk would create syntax error: {e}, skipping")
                    continue

                # Chunk validated - apply to modified_lines
                for i, line in enumerate(fixed_chunk_lines):
                    line_idx = start_line - 1 + i
                    if line_idx < len(modified_lines):
                        modified_lines[line_idx] = line
                    elif i < len(original_chunk_lines):
                        modified_lines.append(line)

                fixes_applied += 1
                self.logger.info(f"Applied fix for: {issue[:50]}")

        if fixes_applied > 0:
            result_code = '\n'.join(modified_lines)
            # Final syntax check
            try:
                compile(result_code, '<result>', 'exec')
                self.logger.info(f"Chunk-based editing complete: {fixes_applied} fixes applied")
                return result_code
            except SyntaxError as e:
                self.logger.warning(f"Final result has syntax error: {e}")
                return None

        return None

    async def _generate_full_file_edit(
        self,
        original_code: str,
        issues: list,
        file_path: str,
        max_tokens: int
    ) -> Optional[str]:
        """Generate edit for smaller files (< 150 lines)."""
        issues_text = "\n".join(f"- {issue}" for issue in issues[:5])

        prompt = f"""Fix ONLY these specific issues. Return the complete fixed code.

FILE: {file_path}

ISSUES:
{issues_text}

CODE:
```python
{original_code}
```

Return the complete fixed Python code:"""

        decision = self.decide_routing(
            complexity=0,
            lines_of_code=len(original_code.split('\n')),
            task_description="small file fix",
            task_type=TaskType.TARGETED_EDIT
        )

        self.routing_stats[decision] += 1
        self.logger.info(f"Small file edit routing: {decision.value}")

        if decision == RoutingDecision.LOCAL:
            result = self.local_client.generate(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2
            )
            if result:
                return self._extract_code(result)

        result = await self.cloud_client.generate(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2
        )
        return self._extract_code(result) if result else None

    def _extract_code(self, response: str) -> str:
        """Extract code from markdown response."""
        if not response:
            return ""
        if "```python" in response:
            return response.split("```python")[1].split("```")[0].strip()
        if "```" in response:
            return response.split("```")[1].split("```")[0].strip()
        return response.strip()

    async def generate_code(
        self,
        prompt: str,
        language: str = "python",
        complexity: int = 0,
        lines_of_code: int = 0,
        max_tokens: int = 2000
    ) -> Optional[str]:
        """
        Generate code with specialized handling.

        Args:
            prompt: Code generation prompt
            language: Programming language
            complexity: Code complexity
            lines_of_code: Lines of code
            max_tokens: Max tokens

        Returns:
            Generated code or None
        """
        system_prompt = f"""You are an expert {language} programmer.
Generate clean, well-documented code.
Return ONLY the code, no explanations.
Ensure the code is syntactically correct."""

        result = await self.generate(
            prompt=prompt,
            complexity=complexity,
            lines_of_code=lines_of_code,
            task_description=f"generate {language} code",
            max_tokens=max_tokens,
            temperature=0.3,
            system=system_prompt
        )

        if result:
            # Extract code from markdown blocks if present
            if f"```{language}" in result:
                result = result.split(f"```{language}")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()

        return result

    def get_stats(self) -> Dict[str, Any]:
        """Get routing statistics."""
        total_requests = sum(self.routing_stats.values())
        local_requests = self.routing_stats[RoutingDecision.LOCAL]

        return {
            "total_requests": total_requests,
            "routing_breakdown": {
                k.value: v for k, v in self.routing_stats.items()
            },
            "local_percentage": (
                local_requests / total_requests * 100
                if total_requests > 0 else 0
            ),
            "estimated_cost_saved": self.total_cost_saved,
            "local_available": self.local_client.available,
            "cloud_available": self.cloud_client.available,
            "local_stats": self.local_client.get_stats(),
            "cloud_stats": self.cloud_client.get_stats()
        }

    def print_stats(self):
        """Print formatted statistics."""
        stats = self.get_stats()

        print("\n" + "=" * 50)
        print("LLM Router Statistics")
        print("=" * 50)
        print(f"Total Requests: {stats['total_requests']}")
        print(f"Local Percentage: {stats['local_percentage']:.1f}%")
        print(f"Estimated Cost Saved: ${stats['estimated_cost_saved']:.2f}")
        print("\nRouting Breakdown:")
        for route, count in stats['routing_breakdown'].items():
            pct = count / stats['total_requests'] * 100 if stats['total_requests'] > 0 else 0
            print(f"  {route}: {count} ({pct:.1f}%)")
        print("=" * 50)


# Singleton instance for easy access
_default_router: Optional[LLMRouter] = None


def get_router() -> LLMRouter:
    """Get the default router instance."""
    global _default_router
    if _default_router is None:
        _default_router = LLMRouter()
    return _default_router


async def smart_generate(
    prompt: str,
    complexity: int = 0,
    lines_of_code: int = 0,
    **kwargs
) -> Optional[str]:
    """Convenience function for smart generation."""
    router = get_router()
    return await router.generate(
        prompt=prompt,
        complexity=complexity,
        lines_of_code=lines_of_code,
        **kwargs
    )


if __name__ == "__main__":
    import asyncio
    logging.basicConfig(level=logging.INFO)

    async def test():
        print("Testing LLM Router (Local + Claude API)...")
        router = LLMRouter()

        # Test routing decisions
        print("\nRouting Decision Tests:")

        # Simple task - should route to local (FREE)
        decision1 = router.decide_routing(complexity=5, lines_of_code=100)
        print(f"  Simple (complexity=5, lines=100): {decision1.value} {'(FREE)' if decision1 == RoutingDecision.LOCAL else '(PAID)'}")

        # Medium task - should route to Claude API
        decision2 = router.decide_routing(complexity=15, lines_of_code=300)
        print(f"  Medium (complexity=15, lines=300): {decision2.value}")

        # Complex task - should route to Claude API
        decision3 = router.decide_routing(complexity=25, lines_of_code=800)
        print(f"  Complex (complexity=25, lines=800): {decision3.value}")

        # Test generation if available
        if router.local_client.available or router.cloud_client.available:
            print("\nTesting generation...")
            result = await router.generate(
                prompt="Write a Python hello world function",
                complexity=2,
                lines_of_code=10
            )
            print(f"Result: {result[:100] if result else 'None'}...")

        router.print_stats()

    asyncio.run(test())

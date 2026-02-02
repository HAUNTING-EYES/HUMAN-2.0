"""
Documentation Agent - PHASE 4
Generates and updates documentation.
"""

import os
import anthropic
import logging
from typing import Dict, List, Any
from .base_agent import BaseAgent
from ..core.event_bus import Event, EventTypes, EventPriority


class DocumentationAgent(BaseAgent):
    """Agent specialized in documentation generation"""

    def __init__(self, event_bus, resources, config: Dict[str, Any] = None):
        super().__init__("DocumentationAgent", event_bus, resources, config)
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = anthropic.Anthropic(api_key=self.api_key) if self.api_key else None

    def register_event_handlers(self):
        """Register for documentation events"""
        self.event_bus.subscribe(EventTypes.IMPROVEMENT_APPLIED, self.on_improvement_applied, self.name)
        self.logger.info("Subscribed to: improvement_applied")

    async def on_improvement_applied(self, event: Event):
        """Update docs when code is improved"""
        file_path = event.data.get('file_path')
        self.logger.info(f"Checking if {file_path} needs documentation update")

    def generate_docstring(self, code: str, function_name: str) -> str:
        """Generate docstring for a function"""
        if not self.client:
            return '"""TODO: Add docstring"""'

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-5-20250929",
                max_tokens=500,
                messages=[{
                    "role": "user",
                    "content": f"""Generate a concise Python docstring for this function:

```python
{code}
```

Return ONLY the docstring in triple quotes."""
                }]
            )

            return message.content[0].text.strip()

        except Exception as e:
            self.logger.error(f"Failed to generate docstring: {e}")
            return '"""TODO: Add docstring"""'

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process documentation task"""
        self.logger.info(f"Processing documentation task: {task.get('type')}")
        return {'success': True}

    def validate_output(self, output: Any) -> bool:
        """Validate documentation"""
        return True

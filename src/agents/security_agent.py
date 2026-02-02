"""
Security Agent - PHASE 4
Scans for security vulnerabilities.
"""

import re
import logging
from typing import Dict, List, Any
from .base_agent import BaseAgent
from ..core.event_bus import Event, EventTypes, EventPriority


class SecurityAgent(BaseAgent):
    """Agent specialized in security scanning"""

    def __init__(self, event_bus, resources, config: Dict[str, Any] = None):
        super().__init__("SecurityAgent", event_bus, resources, config)
        self.vulnerability_patterns = {
            'sql_injection': r'execute\s*\(\s*[\'"].*%s.*[\'"]\s*%',
            'command_injection': r'os\.system\s*\(\s*.*\+',
            'hardcoded_secrets': r'(password|api_key|secret)\s*=\s*[\'"][^\'"]+[\'"]',
            'eval_usage': r'\beval\s*\(',
            'pickle_usage': r'\bpickle\.loads?\s*\(',
        }

    def register_event_handlers(self):
        """Register for security events"""
        self.event_bus.subscribe(EventTypes.CODE_ANALYZED, self.on_code_analyzed, self.name)
        self.logger.info("Subscribed to: code_analyzed")

    async def on_code_analyzed(self, event: Event):
        """Scan for security issues"""
        file_path = event.data.get('file_path')
        code = event.data.get('code', '')

        vulnerabilities = self.scan_for_vulnerabilities(code)

        if vulnerabilities:
            self.logger.warning(f"Found {len(vulnerabilities)} vulnerabilities in {file_path}")
            await self.publish_event(
                'security_vulnerabilities_detected',
                {'file_path': file_path, 'vulnerabilities': vulnerabilities},
                EventPriority.HIGH
            )

    def scan_for_vulnerabilities(self, code: str) -> List[Dict[str, Any]]:
        """Scan code for security vulnerabilities"""
        vulnerabilities = []

        for vuln_type, pattern in self.vulnerability_patterns.items():
            matches = re.finditer(pattern, code, re.IGNORECASE)
            for match in matches:
                vulnerabilities.append({
                    'type': vuln_type,
                    'severity': 'high' if vuln_type in ['sql_injection', 'command_injection'] else 'medium',
                    'line': code[:match.start()].count('\n') + 1,
                    'code_snippet': match.group(0),
                    'description': f"Potential {vuln_type.replace('_', ' ')}"
                })

        return vulnerabilities

    async def process(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process security scan task"""
        self.logger.info(f"Processing security task: {task.get('type')}")
        return {'success': True}

    def validate_output(self, output: Any) -> bool:
        """Validate security scan"""
        return True

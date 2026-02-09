import os
import ast
import copy
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import difflib
from src.components.code_analyzer import CodeAnalyzer
from src.components.code_metrics import CodeMetrics

class SelfModificationFramework:
    """Framework for safe self-modification of AI code."""
    
    def __init__(self, base_dir: str):
        """Initialize self-modification framework.
        
        Args:
            base_dir: Base directory containing AI code
        """
        self.base_dir = Path(base_dir)
        self.logger = logging.getLogger(__name__)
        self.code_analyzer = CodeAnalyzer()
        self.code_metrics = CodeMetrics()
        self.backup_dir = self.base_dir / ".backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
    def analyze_modification(self, file_path: str, new_code: str) -> Dict[str, Any]:
        """Analyze proposed code modification for safety and improvements.
        
        Args:
            file_path: Path to file to modify
            new_code: Proposed new code
            
        Returns:
            Analysis results including safety checks and metrics
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
            
        with open(file_path) as f:
            original_code = f.read()
            
        # Safety checks
        safety_checks = self._perform_safety_checks(original_code, new_code)
        if not safety_checks["is_safe"]:
            return {
                "is_safe": False,
                "safety_issues": safety_checks["issues"],
                "diff": self._generate_diff(original_code, new_code)
            }
            
        # Analyze improvements
        metrics_before = self.code_metrics.calculate_metrics(original_code)
        metrics_after = self.code_metrics.calculate_metrics(new_code)
        
        return {
            "is_safe": True,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "improvements": self._analyze_improvements(metrics_before, metrics_after),
            "diff": self._generate_diff(original_code, new_code)
        }
        
    def apply_modification(self, file_path: str, new_code: str) -> bool:
        """Safely apply code modification with backup.
        
        Args:
            file_path: Path to file to modify
            new_code: New code to apply
            
        Returns:
            True if modification was successful
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise ValueError(f"File not found: {file_path}")
            
        # Analyze modification first
        analysis = self.analyze_modification(file_path, new_code)
        if not analysis["is_safe"]:
            self.logger.error("Unsafe modification detected")
            return False
            
        # Create backup
        backup_path = self._create_backup(file_path)
        
        try:
            # Apply modification
            with open(file_path, "w") as f:
                f.write(new_code)
                
            # Verify modification
            if not self._verify_modification(file_path, new_code):
                self._restore_backup(backup_path, file_path)
                return False
                
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying modification: {str(e)}")
            self._restore_backup(backup_path, file_path)
            return False
            
    def _perform_safety_checks(self, original_code: str, new_code: str) -> Dict[str, Any]:
        """Perform safety checks on code modification."""
        issues = []
        
        # Check for dangerous imports
        dangerous_imports = {"os.system", "subprocess", "eval", "exec"}
        original_imports = self._get_imports(original_code)
        new_imports = self._get_imports(new_code)
        
        for imp in new_imports - original_imports:
            if any(d in imp for d in dangerous_imports):
                issues.append(f"Dangerous import detected: {imp}")
                
        # Check for dangerous function calls
        dangerous_calls = {"eval(", "exec(", "os.system(", "subprocess.run("}
        for call in dangerous_calls:
            if call in new_code and call not in original_code:
                issues.append(f"Dangerous function call detected: {call}")
                
        # Check code structure
        try:
            ast.parse(new_code)
        except SyntaxError as e:
            issues.append(f"Syntax error in new code: {str(e)}")
            
        return {
            "is_safe": len(issues) == 0,
            "issues": issues
        }
        
    def _get_imports(self, code: str) -> set:
        """Extract imports from code."""
        imports = set()
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for name in node.names:
                        imports.add(name.name)
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    for name in node.names:
                        imports.add(f"{module}.{name.name}")
        except:
            pass
        return imports
        
    def _analyze_improvements(self, metrics_before: Dict[str, float], 
                            metrics_after: Dict[str, float]) -> Dict[str, float]:
        """Analyze improvements in code metrics."""
        return {
            key: metrics_after[key] - metrics_before[key]
            for key in metrics_before.keys()
        }
        
    def _generate_diff(self, original_code: str, new_code: str) -> str:
        """Generate readable diff between original and new code."""
        diff = difflib.unified_diff(
            original_code.splitlines(keepends=True),
            new_code.splitlines(keepends=True),
            fromfile="original",
            tofile="modified"
        )
        return "".join(diff)
        
    def _create_backup(self, file_path: Path) -> Path:
        """Create backup of file before modification."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"{file_path.name}.{timestamp}.bak"
        with open(file_path) as f:
            original_code = f.read()
        with open(backup_path, "w") as f:
            f.write(original_code)
        return backup_path
        
    def _restore_backup(self, backup_path: Path, file_path: Path) -> None:
        """Restore file from backup."""
        with open(backup_path) as f:
            backup_code = f.read()
        with open(file_path, "w") as f:
            f.write(backup_code)
            
    def _verify_modification(self, file_path: Path, expected_code: str) -> bool:
        """Verify file was modified correctly."""
        with open(file_path) as f:
            actual_code = f.read()
        return actual_code == expected_code 
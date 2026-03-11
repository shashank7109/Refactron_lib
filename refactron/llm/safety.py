"""Safety gate for validating LLM-generated code."""

import ast
from typing import List, Set

from refactron.llm.models import RefactoringSuggestion, SafetyCheckResult


class SafetyGate:
    """Validates code suggestions for safety and correctness."""

    def __init__(self, min_confidence: float = 0.7):
        self.min_confidence = min_confidence

    def validate(self, suggestion: RefactoringSuggestion) -> SafetyCheckResult:
        """Validate a refactoring suggestion.

        Args:
            suggestion: The suggestion to validate

        Returns:
            Safety check result
        """
        issues = []
        side_effects = []
        score = 1.0

        # 1. Syntax Check
        try:
            ast.parse(suggestion.proposed_code)
            syntax_valid = True
        except SyntaxError as e:
            syntax_valid = False
            issues.append(f"Syntax Error: {e}")
            score = 0.0

        # 2. Confidence Check
        if suggestion.confidence_score < self.min_confidence:
            issues.append(
                f"Low confidence score: {suggestion.confidence_score:.2f} < {self.min_confidence}"
            )
            score *= 0.8

        # 3. Basic Security/Risk Checks
        risk_score = self._assess_risk(suggestion.proposed_code)
        if risk_score > 0.8:
            issues.append("High risk code detected")
            score *= 0.5

        # 4. Dangerous Imports Check
        dangerous_imports = self._check_dangerous_imports(
            suggestion.proposed_code, suggestion.original_code
        )
        if dangerous_imports:
            issues.append(f"Dangerous imports detected: {', '.join(dangerous_imports)}")
            side_effects.extend([f"Import: {imp}" for imp in dangerous_imports])
            score *= 0.7

        return SafetyCheckResult(
            passed=(score > 0.8 and syntax_valid),
            score=score,
            issues=issues,
            syntax_valid=syntax_valid,
            side_effects=side_effects,
        )

    def _assess_risk(self, code: str) -> float:
        """Assess the risk of the code patch."""
        risk = 0.0

        # Keywords that suggest side effects
        risky_keywords = [
            "subprocess",
            "os.system",
            "shutil.rmtree",
            "open(",
            "requests.",
            "urllib",
            "eval(",
            "exec(",
        ]

        for keyword in risky_keywords:
            if keyword in code:
                risk += 0.3

        return min(risk, 1.0)

    def _check_dangerous_imports(self, proposed_code: str, original_code: str) -> List[str]:
        """Check for potentially dangerous imports that are NEW."""
        dangerous_modules = ["subprocess", "os", "shutil", "sys"]

        def get_imports(code: str) -> Set[str]:
            imports = set()
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for name in node.names:
                            if name.name in dangerous_modules:
                                imports.add(name.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module in dangerous_modules:
                            imports.add(node.module)
            except SyntaxError:
                pass
            return imports

        original_imports = get_imports(original_code)
        proposed_imports = get_imports(proposed_code)

        # Only flag imports that were added by the LLM
        return list(proposed_imports - original_imports)

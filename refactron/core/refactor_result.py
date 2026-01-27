"""Refactoring result representation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

from refactron.core.models import RefactoringOperation


@dataclass
class RefactorResult:
    """Result of refactoring operations."""

    operations: List[RefactoringOperation] = field(default_factory=list)
    applied: bool = False
    preview_mode: bool = True

    @property
    def total_operations(self) -> int:
        """Total number of refactoring operations."""
        return len(self.operations)

    @property
    def high_risk_operations(self) -> List[RefactoringOperation]:
        """Operations with risk score > 0.7."""
        return [op for op in self.operations if op.risk_score > 0.7]

    @property
    def safe_operations(self) -> List[RefactoringOperation]:
        """Operations with risk score <= 0.3."""
        return [op for op in self.operations if op.risk_score <= 0.3]

    def top_ranked_operations(self, top_n: int = 10) -> List[RefactoringOperation]:
        """Get top N ranked operations by ranking score."""
        ranked = [op for op in self.operations if "ranking_score" in op.metadata]
        ranked.sort(key=lambda op: op.metadata.get("ranking_score", 0.0), reverse=True)
        return ranked[:top_n]

    def get_ranking_score(self, operation: RefactoringOperation) -> float:
        """Get ranking score for an operation (0.0 if not ranked)."""
        value = operation.metadata.get("ranking_score", 0.0)
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    def operations_by_file(self, file_path: Path) -> List[RefactoringOperation]:
        """Get operations for a specific file."""
        return [op for op in self.operations if op.file_path == file_path]

    def operations_by_type(self, operation_type: str) -> List[RefactoringOperation]:
        """Get operations of a specific type."""
        return [op for op in self.operations if op.operation_type == operation_type]

    def show_diff(self) -> str:
        """Show a diff of all operations."""
        lines = []
        lines.append("=" * 80)
        lines.append("REFACTORING PREVIEW")
        lines.append("=" * 80)
        lines.append("")
        lines.append(f"Total Operations: {self.total_operations}")
        lines.append(f"Safe Operations: {len(self.safe_operations)}")
        lines.append(f"High Risk Operations: {len(self.high_risk_operations)}")
        lines.append("")

        for i, op in enumerate(self.operations, 1):
            lines.append("-" * 80)
            lines.append(f"Operation {i}: {op.operation_type}")
            lines.append(f"Location: {op.file_path}:{op.line_number}")
            lines.append(f"Risk Score: {op.risk_score:.2f}")
            ranking_score = self.get_ranking_score(op)
            if ranking_score > 0:
                lines.append(f"Ranking Score: {ranking_score:.3f}")
            lines.append(f"Description: {op.description}")

            if op.reasoning:
                lines.append(f"Reasoning: {op.reasoning}")

            lines.append("")
            lines.append("- OLD CODE:")
            for line in op.old_code.split("\n"):
                lines.append(f"  - {line}")

            lines.append("")
            lines.append("+ NEW CODE:")
            for line in op.new_code.split("\n"):
                lines.append(f"  + {line}")
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def apply(self) -> bool:
        """Apply the refactoring operations (placeholder)."""
        # This would actually apply the changes to files
        self.applied = True
        return True

    def summary(self) -> Dict[str, int]:
        """Get a summary of refactoring operations."""
        return {
            "total_operations": self.total_operations,
            "high_risk": len(self.high_risk_operations),
            "safe": len(self.safe_operations),
            "applied": self.applied,
        }

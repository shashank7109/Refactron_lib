"""Quality gate parsing and enforcement for CI/CD pipelines."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from refactron.core.analysis_result import AnalysisResult


@dataclass
class QualityGate:
    """Configuration for quality gate thresholds."""

    max_critical: int = 0
    max_errors: int = 10
    max_warnings: int = 50
    max_total: Optional[int] = None
    fail_on_critical: bool = True
    fail_on_errors: bool = False
    fail_on_warnings: bool = False
    min_success_rate: float = 0.95  # 95% of files must analyze successfully

    def check(self, result: AnalysisResult) -> Tuple[bool, str]:
        """Check if quality gate passes.

        Args:
            result: Analysis result to check

        Returns:
            Tuple of (passed, message)
        """
        summary = result.summary()
        failures: List[str] = []

        # Check critical issues
        if summary["critical"] > self.max_critical:
            failures.append(
                f"Critical issues exceed threshold: {summary['critical']} > {self.max_critical}"
            )

        # Check error-level issues
        if summary["errors"] > self.max_errors:
            failures.append(
                f"Error-level issues exceed threshold: {summary['errors']} > {self.max_errors}"
            )

        # Check warning-level issues
        if summary["warnings"] > self.max_warnings:
            msg = (
                f"Warning-level issues exceed threshold: "
                f"{summary['warnings']} > {self.max_warnings}"
            )
            failures.append(msg)

        # Check total issues
        if self.max_total and summary["total_issues"] > self.max_total:
            failures.append(
                f"Total issues exceed threshold: {summary['total_issues']} > {self.max_total}"
            )

        # Check success rate
        if summary["total_files"] > 0:
            success_rate = summary["files_analyzed"] / summary["total_files"]
            if success_rate < self.min_success_rate:
                msg = (
                    f"Analysis success rate too low: {success_rate:.1%} < "
                    f"{self.min_success_rate:.1%}"
                )
                failures.append(msg)

        # Check fail flags
        if self.fail_on_critical and summary["critical"] > 0:
            failures.append("Critical issues found (fail_on_critical=True)")

        if self.fail_on_errors and summary["errors"] > 0:
            failures.append("Error-level issues found (fail_on_errors=True)")

        if self.fail_on_warnings and summary["warnings"] > 0:
            failures.append("Warning-level issues found (fail_on_warnings=True)")

        if failures:
            return False, "; ".join(failures)

        return True, "Quality gate passed"

    def to_dict(self) -> Dict:
        """Convert quality gate to dictionary."""
        return {
            "max_critical": self.max_critical,
            "max_errors": self.max_errors,
            "max_warnings": self.max_warnings,
            "max_total": self.max_total,
            "fail_on_critical": self.fail_on_critical,
            "fail_on_errors": self.fail_on_errors,
            "fail_on_warnings": self.fail_on_warnings,
            "min_success_rate": self.min_success_rate,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "QualityGate":
        """Create quality gate from dictionary."""
        return cls(**data)


class QualityGateParser:
    """Parse CLI output and enforce quality gates."""

    @staticmethod
    def parse_json_output(json_path: Path) -> Dict:
        """Parse JSON output from refactron analyze --format json.

        Args:
            json_path: Path to JSON output file

        Returns:
            Parsed JSON dictionary

        Raises:
            ValueError: If JSON is invalid
            FileNotFoundError: If file doesn't exist
        """
        if not json_path.exists():
            raise FileNotFoundError(f"JSON output file not found: {json_path}")

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                result: Dict[str, Any] = json.load(f)
                return result
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in output file: {e}") from e

    @staticmethod
    def parse_text_output(text: str) -> Dict[str, int]:
        """Parse text output from refactron analyze command.

        Args:
            text: Text output from CLI

        Returns:
            Dictionary with issue counts
        """
        summary: Dict[str, int] = {
            "critical": 0,
            "errors": 0,
            "warnings": 0,
            "info": 0,
            "total": 0,
        }

        # Extract summary numbers from text output
        critical_match = re.search(r"Critical:\s*(\d+)", text, re.IGNORECASE)
        if critical_match:
            summary["critical"] = int(critical_match.group(1))

        error_match = re.search(r"Error[s]?:\s*(\d+)", text, re.IGNORECASE)
        if error_match:
            summary["errors"] = int(error_match.group(1))

        warning_match = re.search(r"Warning[s]?:\s*(\d+)", text, re.IGNORECASE)
        if warning_match:
            summary["warnings"] = int(warning_match.group(1))

        info_match = re.search(r"Info:\s*(\d+)", text, re.IGNORECASE)
        if info_match:
            summary["info"] = int(info_match.group(1))

        total_match = re.search(r"Total[:\s]+(\d+)", text, re.IGNORECASE)
        if total_match:
            summary["total"] = int(total_match.group(1))

        return summary

    @staticmethod
    def parse_exit_code(exit_code: int) -> Dict[str, int]:
        """Parse exit code from refactron analyze.

        Args:
            exit_code: Process exit code

        Returns:
            Dictionary indicating if build should fail
        """
        return {
            "exit_code": exit_code,
            "should_fail": exit_code != 0,
        }

    @staticmethod
    def enforce_gate(result: AnalysisResult, gate: QualityGate) -> Tuple[bool, str, int]:
        """Enforce quality gate on analysis result.

        Args:
            result: Analysis result
            gate: Quality gate configuration

        Returns:
            Tuple of (passed, message, exit_code)
        """
        passed, message = gate.check(result)

        if passed:
            return True, message, 0

        return False, message, 1

    @staticmethod
    def generate_summary(result: AnalysisResult) -> str:
        """Generate quality gate summary for CI/CD.

        Args:
            result: Analysis result

        Returns:
            Formatted summary string
        """
        summary = result.summary()
        lines = []

        lines.append("## 📊 Refactron Quality Gate Summary")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        lines.append(f"| Files Analyzed | {summary['files_analyzed']}/{summary['total_files']} |")
        lines.append(f"| Critical Issues | {summary['critical']} |")
        lines.append(f"| Error Issues | {summary['errors']} |")
        lines.append(f"| Warning Issues | {summary['warnings']} |")
        lines.append(f"| Total Issues | {summary['total_issues']} |")
        lines.append("")

        if summary["files_failed"] > 0:
            lines.append(f"⚠️ **{summary['files_failed']} file(s) failed analysis**")
            lines.append("")

        return "\n".join(lines)

#!/usr/bin/env python3
"""Analyze existing feedback and pattern data for ML model training.

This script scans all Refactron storage directories to assess:
- Volume of feedback records
- Quality of data (completeness)
- Distribution of actions and operation types
- Readiness for ML training
"""

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from refactron.patterns.storage import PatternStorage  # noqa: E402


def analyze_feedback() -> Optional[Dict[str, Any]]:
    """Analyze all available feedback data."""

    # Find all pattern storage directories
    root = Path(".")
    storage_dirs = list(root.glob("**/.refactron/patterns"))

    print(f"Found {len(storage_dirs)} storage directories\n")

    all_feedback = []
    all_patterns = {}

    # Aggregate data from all projects
    for storage_dir in storage_dirs:
        try:
            storage = PatternStorage(storage_dir)
            feedback = storage.load_feedback()
            patterns = storage.load_patterns()

            all_feedback.extend(feedback)
            all_patterns.update(patterns)

            print(f"📁 {storage_dir.parent.parent}")
            print(f"   Feedback: {len(feedback)}, Patterns: {len(patterns)}")
        except Exception as e:
            print(f"⚠️  Error loading {storage_dir}: {e}")

    if not all_feedback:
        print("\n❌ No feedback data found!")
        print("   Run some refactoring operations and provide feedback first.")
        return None

    print(f"\n{'='*60}")
    print("AGGREGATE STATISTICS")
    print(f"{'='*60}\n")

    print("📊 Total Records:")
    print(f"   Feedback: {len(all_feedback)}")
    print(f"   Patterns: {len(all_patterns)}")

    # Action distribution
    actions = Counter(f.action for f in all_feedback)
    print("\n✅ Action Distribution:")
    for action, count in actions.most_common():
        pct = count / len(all_feedback) * 100
        print(f"   {action:12s}: {count:4d} ({pct:5.1f}%)")

    # Operation types
    operation_types = Counter(f.operation_type for f in all_feedback)
    print("\n🔧 Operation Types:")
    for op_type, count in operation_types.most_common(5):
        pct = count / len(all_feedback) * 100
        print(f"   {op_type:20s}: {count:4d} ({pct:5.1f}%)")

    # Data quality
    with_patterns = sum(
        1 for f in all_feedback if hasattr(f, "code_pattern_hash") and f.code_pattern_hash
    )
    with_reason = sum(1 for f in all_feedback if hasattr(f, "reason") and f.reason)

    print("\n📋 Data Quality:")
    print(f"   With pattern hash: {with_patterns:4d} ({with_patterns/len(all_feedback)*100:5.1f}%)")
    print(f"   With reason:       {with_reason:4d} ({with_reason/len(all_feedback)*100:5.1f}%)")

    # ML readiness
    quality_score = with_patterns / len(all_feedback) if all_feedback else 0

    print("\n🎯 ML Readiness:")
    print(f"   Quality Score: {quality_score:.2%}")

    if len(all_feedback) < 50:
        print("   Status: ❌ INSUFFICIENT DATA")
        print(f"   Need: {50 - len(all_feedback)} more feedback records")
    elif quality_score < 0.7:
        print("   Status: ⚠️  LOW QUALITY")
        print("   Many records missing pattern hashes")
    else:
        print("   Status: ✅ READY FOR TRAINING")

    # Save detailed report
    report = {
        "summary": {
            "total_feedback": len(all_feedback),
            "total_patterns": len(all_patterns),
            "quality_score": quality_score,
            "ml_ready": len(all_feedback) >= 50 and quality_score >= 0.7,
        },
        "actions": dict(actions),
        "operation_types": dict(operation_types),
        "quality": {"with_pattern_hash": with_patterns, "with_reason": with_reason},
    }

    report_file = Path("feedback_analysis.json")
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n💾 Detailed report saved to: {report_file}")

    return report


if __name__ == "__main__":
    print("🔍 Refactron Feedback Data Analysis\n")
    report = analyze_feedback()

    if report and report["summary"]["ml_ready"]:
        print("\n✨ Ready to proceed with ML model training!")
    elif report:
        print("\n⏳ Collect more feedback data before training.")

    sys.exit(0 if report else 1)

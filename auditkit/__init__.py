"""AuditKIT: Regulatory Compliance & Evaluation Benchmark for Regulated Banking Models.

Provides automated evaluation suites for citation accuracy, red-team structuring
refusals, and hallucination suppression using SGLang.
"""

__version__ = "0.1.0"
__all__ = [
    "ComplianceBenchmark",
    "AuditEvaluator",
    "ComparativeEvaluator",
    "run_compliance_audit",
]

from .benchmark import ComplianceBenchmark
from .evaluator import AuditEvaluator, run_compliance_audit
from .compare import ComparativeEvaluator

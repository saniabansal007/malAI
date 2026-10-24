"""AuditKIT: Compliance & Evaluation Benchmark for Enchanted Emporium (Dobby).

Provides automated evaluation suites for citation accuracy ([MAG-...], [RET-...]),
Dark Arts and fraudulent policy bypass refusals, and performance benchmarking using SGLang.
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

"""
CuratorKIT: High-throughput Data Curation Engine for Regulated Domain Assistants.
Uses SGLang for high-speed policy Q&A synthesis, adversarial negative sampling,
and verifiable ground-truth generation.
"""

from curatorkit.extractor import PolicyClause, PolicyExtractor
from curatorkit.generator import SGLangCurator
from curatorkit.verifier import PolicyVerifier
from curatorkit.export import DatasetExporter

__all__ = [
    "PolicyClause",
    "PolicyExtractor",
    "SGLangCurator",
    "PolicyVerifier",
    "DatasetExporter",
]

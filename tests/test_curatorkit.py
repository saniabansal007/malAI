import os
import sys
import json
import pytest
from pathlib import Path

# Ensure localmind root directory is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from curatorkit.extractor import PolicyExtractor, PolicyClause
from curatorkit.verifier import PolicyVerifier, CuratedSample
from curatorkit.export import DatasetExporter
from curatorkit.generator import clean_json_content, build_generation_payload


def test_policy_extractor_finds_corpus():
    extractor = PolicyExtractor()
    clauses = extractor.extract_all()
    assert len(clauses) > 0, "PolicyExtractor should find clauses in curatorkit/corpus"
    for c in clauses:
        assert isinstance(c, PolicyClause)
        assert c.citation_handle.startswith("[POL-")
        assert len(c.text.strip()) > 0
        assert len(c.keywords) > 0


def test_clean_json_content():
    raw_markdown = '```json\n{"prompt": "test question", "answer": "test answer"}\n```'
    cleaned = clean_json_content(raw_markdown)
    data = json.loads(cleaned)
    assert data["prompt"] == "test question"
    assert data["answer"] == "test answer"


def test_policy_verifier_qa():
    verifier = PolicyVerifier()
    clause = PolicyClause(
        doc_id="POL-AML-TEST",
        section_number="Sec. 1.1",
        title="Test Clause",
        citation_handle="[POL-AML-TEST / Sec. 1.1]",
        text="A CTR must be filed within 15 days.",
        keywords=["CTR", "filing"]
    )

    valid_response = {
        "prompt": "When should CTR be filed?",
        "answer": "According to [POL-AML-TEST / Sec. 1.1], a CTR must be filed within 15 days."
    }
    sample = verifier.verify_sample("policy_qa", valid_response, clause)
    assert sample is not None
    assert sample.category == "policy_qa"
    assert sample.citation_handle == clause.citation_handle

    # Test invalid response (missing citation)
    invalid_response = {
        "prompt": "When should CTR be filed?",
        "answer": "A CTR must be filed within 15 days without citation."
    }
    invalid_sample = verifier.verify_sample("policy_qa", invalid_response, clause)
    assert invalid_sample is None


def test_dpo_pair_generation():
    verifier = PolicyVerifier()
    sample = CuratedSample(
        sample_id="test-001",
        category="policy_qa",
        clause_id="POL-TEST / Sec. 1",
        citation_handle="[POL-TEST / Sec. 1]",
        prompt="What is the rule?",
        chosen_response="Under [POL-TEST / Sec. 1], all transactions over $10,000 must be reported.",
        rejected_response="You do not need to report anything.",
        system_prompt="You are a compliance assistant."
    )
    pair = verifier.build_dpo_pair(sample)
    assert pair["prompt"] == sample.prompt
    assert pair["chosen"] == sample.chosen_response
    assert pair["rejected"] == sample.rejected_response


def test_dataset_exporter(tmp_path):
    exporter = DatasetExporter(output_dir=str(tmp_path))
    sample = CuratedSample(
        sample_id="test-001",
        category="policy_qa",
        clause_id="POL-TEST / Sec. 1",
        citation_handle="[POL-TEST / Sec. 1]",
        prompt="What is the rule?",
        chosen_response="Under [POL-TEST / Sec. 1], report transactions over $10,000.",
        rejected_response="No reporting is needed.",
        system_prompt="You are a compliance assistant."
    )
    summary = exporter.export([sample], val_ratio=0.0)
    assert summary["total_verified_samples"] == 1
    assert os.path.exists(tmp_path / "sft_train.jsonl")
    assert os.path.exists(tmp_path / "dpo_pairs.jsonl")
    assert os.path.exists(tmp_path / "curation_summary.json")

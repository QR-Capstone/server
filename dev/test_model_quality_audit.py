#!/usr/bin/env python3
"""Self-check the current model quality audit command."""
from __future__ import annotations

import tempfile
from pathlib import Path

import audit_model_quality as audit
import stamp_model_quality_metadata as stamp


def main() -> int:
    assert audit.main([]) == 0
    assert audit.main(["--min-accuracy", "1.01"]) == 1
    assert audit.main(["--max-p95-ms", "0.1"]) == 1
    with tempfile.TemporaryDirectory(prefix="quality_audit_rules_") as tmp_raw:
        rule = Path(tmp_raw) / "rules.py"
        rule.write_text("KNOWN = {'good.example'}\n", encoding="utf-8")
        payload = {"runtime_rule_files": [stamp.runtime_rule_record(rule)]}
        assert audit._runtime_rule_failures(payload, [rule]) == []
        rule.write_text("KNOWN = {'changed.example'}\n", encoding="utf-8")
        failures = audit._runtime_rule_failures(payload, [rule])
        assert any("sha256_mismatch" in failure for failure in failures), failures
    print("model quality audit selfcheck passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

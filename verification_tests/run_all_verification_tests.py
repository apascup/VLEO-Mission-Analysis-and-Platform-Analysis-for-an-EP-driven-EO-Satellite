"""
===============================================================================
Project:       VLEO Mission Analysis and Platform Analysis for an EP-driven,
               EO Satellite
Collaboration: In collaboration with ArianeGroup
Institution:   Cranfield University
Author:        Arnau Pascual
Year:          2026
===============================================================================
File:          run_all_verification_tests.py
Description:
    Master test suite runner executing Sections 4.5 through 4.8 and compiling verification matrices.
===============================================================================
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

# Setup paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIF_DIR = PROJECT_ROOT / "verification_tests"
RESULTS_DIR = VERIF_DIR / "results"

if str(VERIF_DIR) not in sys.path:
    sys.path.insert(0, str(VERIF_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from verification_config import (
    TestRecord,
    ensure_results_directories,
    init_orekit,
    write_markdown_summary,
)

# Import test group modules
from test_45_unit_verification import run_all_unit_tests
from test_46_orbital_analytical_verification import run_all_orbital_analytical_tests
from test_47_integrated_model_verification import run_all_integrated_tests
from test_48_environmental_model_validation import run_all_environmental_tests


def run_full_suite() -> int:
    """Run all verification and validation test groups and compile final report."""
    start_time = time.perf_counter()
    ensure_results_directories()

    print("\n" + "#" * 78)
    print("  PRELIMINARY VLEO MISSION ANALYSIS VERIFICATION & VALIDATION SUITE")
    print("  ECSS-Style Verification Matrix (Sections 4.5, 4.6, 4.7, 4.8)")
    print("#" * 78 + "\n")

    # Verify Orekit initialization
    if not init_orekit():
        print("[CRITICAL] Orekit failed to initialize. Aborting orbital simulations.")
        return 1

    all_records: List[TestRecord] = []

    # 1. Section 4.5: Unit Verification
    try:
        records_45 = run_all_unit_tests()
        all_records.extend(records_45)
    except Exception as e:
        print(f"[ERROR] Section 4.5 failed with exception: {e}")

    # 2. Section 4.6: Orbital Analytical Verification
    try:
        records_46 = run_all_orbital_analytical_tests()
        all_records.extend(records_46)
    except Exception as e:
        print(f"[ERROR] Section 4.6 failed with exception: {e}")

    # 3. Section 4.7: Integrated Model Verification
    try:
        records_47 = run_all_integrated_tests()
        all_records.extend(records_47)
    except Exception as e:
        print(f"[ERROR] Section 4.7 failed with exception: {e}")

    # 4. Section 4.8: Environmental Model Validation
    try:
        records_48 = run_all_environmental_tests()
        all_records.extend(records_48)
    except Exception as e:
        print(f"[ERROR] Section 4.8 failed with exception: {e}")

    # Generate Markdown Summary
    summary_path = RESULTS_DIR / "verification_summary.md"
    write_markdown_summary(summary_path, all_records)

    elapsed = time.perf_counter() - start_time
    total = len(all_records)
    passed = sum(1 for r in all_records if r.status == "PASS")
    failed = sum(1 for r in all_records if r.status == "FAIL")
    warnings = sum(1 for r in all_records if r.status == "WARNING")

    print("\n" + "=" * 78)
    print("  VERIFICATION & VALIDATION SUITE EXECUTION SUMMARY")
    print("=" * 78)
    print(f"  Total tests executed: {total}")
    print(f"  Passed:               {passed}")
    print(f"  Failed:               {failed}")
    print(f"  Warnings / Reviews:   {warnings}")
    print(f"  Execution time:       {elapsed:.1f} s")
    print(f"  Results directory:    {RESULTS_DIR.resolve()}")
    print(f"  Markdown summary:     {summary_path.resolve()}")
    print("=" * 78 + "\n")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit_code = run_full_suite()
    sys.exit(exit_code)

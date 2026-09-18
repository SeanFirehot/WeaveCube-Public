#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterable

VERSION = "v37.527"
EXPECTED_CORPUS_SHA256 = "77bd96782994e5ae41e98ebab013de42354178bb9589e043dc12c4e416fee88e"
EXPECTED_D6_SHA256 = "5b985def30317e416a03eb8bde4c039bc7ea1c4d7c34683dff4ff8cc24055dd0"

SEAL_REL = Path("reports/presentation/full24_v37_525/SEALED_FULL24_CORPUS_v37_525.json")
D6_REL = Path("reports/pdcc_cache/p1_tail_nosuffix_v36_6i13_d6.pkl")
DRIVER_REL = Path("scripts/audit_adaptive_random_coverage_v37_153.py")

CASE_COUNT = 24
SCRAMBLE_LEN = 25

class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams
    def write(self, s: str) -> int:
        for st in self.streams:
            st.write(s)
            st.flush()
        return len(s)
    def flush(self) -> None:
        for st in self.streams:
            st.flush()

def sha256_file(path: Path, chunk: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()

def canonical_corpus_hash(seal: dict[str, Any]) -> str:
    core = {
        "schema": seal["schema"],
        "source_directive": seal["source_directive"],
        "generator": seal["generator"],
        "cases": seal["cases"],
    }
    canonical = json.dumps(core, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def load_and_validate_seal(path: Path) -> tuple[dict[str, Any], list[tuple[str, ...]]]:
    if not path.is_file():
        raise SystemExit(f"BLOCKED: missing sealed corpus: {path}")
    seal = json.loads(path.read_text(encoding="utf-8"))

    observed = canonical_corpus_hash(seal)
    recorded = str(seal.get("corpus_sha256", ""))
    if observed != recorded:
        raise SystemExit(
            "BLOCKED: sealed corpus internal SHA mismatch\n"
            f"recorded={recorded}\nobserved={observed}"
        )
    if observed != EXPECTED_CORPUS_SHA256:
        raise SystemExit(
            "BLOCKED: sealed corpus is not the previously frozen corpus\n"
            f"expected={EXPECTED_CORPUS_SHA256}\nobserved={observed}"
        )

    cases = seal.get("cases", [])
    if len(cases) != CASE_COUNT:
        raise SystemExit(f"BLOCKED: expected {CASE_COUNT} cases, found {len(cases)}")

    words: list[tuple[str, ...]] = []
    for i, row in enumerate(cases, 1):
        moves = tuple(str(x) for x in row.get("scramble_moves", []))
        if len(moves) != SCRAMBLE_LEN:
            raise SystemExit(
                f"BLOCKED: case {i} expected {SCRAMBLE_LEN} moves, found {len(moves)}"
            )
        words.append(moves)
    return seal, words

def recursive_scrambles(obj: Any, path: str = "$") -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{path}.{k}"
            if "scrambl" in str(k).lower():
                if isinstance(v, str):
                    rows.append((p, v.strip()))
                elif isinstance(v, (list, tuple)) and v and all(isinstance(x, str) for x in v):
                    # A single tokenized scramble if it looks like moves.
                    if len(v) == SCRAMBLE_LEN and all(" " not in x for x in v):
                        rows.append((p, " ".join(v)))
                    else:
                        for j, x in enumerate(v):
                            rows.append((f"{p}[{j}]", x.strip()))
            rows.extend(recursive_scrambles(v, p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            rows.extend(recursive_scrambles(v, f"{path}[{i}]"))
    return rows

def dedupe_in_order(values: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out

def classify_summary(analysis: Any) -> dict[str, Any]:
    """
    Best-effort extraction only. Raw v37.153 JSON remains authority.
    We deliberately avoid inventing field names if absent.
    """
    out: dict[str, Any] = {}
    if not isinstance(analysis, dict):
        return out
    for key in (
        "solved", "solved_count", "fresh_count", "fallback_count",
        "resource_inconclusive", "complete_no_dr_through_max",
        "replay_failures", "decision", "coverage", "case_summaries",
    ):
        if key in analysis:
            val = analysis[key]
            if isinstance(val, list):
                out[key] = {"count": len(val)}
            else:
                out[key] = val
    return out

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=r"C:\python\CubeLab")
    ap.add_argument("--output-dir", default="reports/presentation/full24_v37_527")
    args = ap.parse_args()

    root = Path(args.root).resolve()
    outdir = root / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Reuse the already sealed corpus. NO generation and NO CubeLab import yet.
    # ------------------------------------------------------------------
    seal_path = root / SEAL_REL
    seal, sealed_words = load_and_validate_seal(seal_path)
    sealed_strings = [" ".join(w) for w in sealed_words]

    print("# CubeLab SEALED FULL-24 operational runner v37.527")
    print(f"root                    : {root}")
    print(f"seal                    : {seal_path}")
    print(f"corpus sha256           : {seal['corpus_sha256']}")
    print(f"cases                   : {len(sealed_words)}")
    print(f"moves/case              : {SCRAMBLE_LEN}")
    print("corpus regenerated      : False")
    print("solver imported yet     : False")

    # ------------------------------------------------------------------
    # 2. Validate recovered D6 before importing solver.
    # ------------------------------------------------------------------
    d6 = root / D6_REL
    if not d6.is_file():
        raise SystemExit(f"BLOCKED: missing D6: {d6}")
    d6_sha = sha256_file(d6)
    print(f"D6 sha256               : {d6_sha}")
    print(f"D6 valid                : {d6_sha == EXPECTED_D6_SHA256}")
    if d6_sha != EXPECTED_D6_SHA256:
        raise SystemExit("BLOCKED: D6 SHA mismatch")

    # ------------------------------------------------------------------
    # 3. Import historical operational v37.153 stack.
    # ------------------------------------------------------------------
    driver_path = root / DRIVER_REL
    if not driver_path.is_file():
        raise SystemExit(f"BLOCKED: missing driver: {driver_path}")

    os.chdir(root)
    sys.path.insert(0, str(root / "src"))
    sys.path.insert(0, str(root / "scripts"))

    spec = importlib.util.spec_from_file_location("cubelab_v37_153_full24_driver", driver_path)
    if spec is None or spec.loader is None:
        raise SystemExit("BLOCKED: cannot import v37.153")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)

    v151 = getattr(mod, "v151", None)
    if v151 is None:
        raise SystemExit("BLOCKED: v37.153 has no imported v151 module")
    original_generate = getattr(v151, "_generate_scrambles", None)
    if not callable(original_generate):
        raise SystemExit("BLOCKED: v151._generate_scrambles is not callable")

    calls: list[dict[str, int]] = []

    def sealed_generate_scrambles(*, fresh_count: int, fresh_seed: int, scramble_length: int):
        calls.append({
            "fresh_count": int(fresh_count),
            "fresh_seed": int(fresh_seed),
            "scramble_length": int(scramble_length),
        })
        if int(fresh_count) != CASE_COUNT:
            raise RuntimeError(
                f"SEALED CORPUS CONTRACT: driver requested fresh_count={fresh_count}, "
                f"expected {CASE_COUNT}"
            )
        if int(scramble_length) != SCRAMBLE_LEN:
            raise RuntimeError(
                f"SEALED CORPUS CONTRACT: driver requested scramble_length={scramble_length}, "
                f"expected {SCRAMBLE_LEN}"
            )
        # fresh_seed is deliberately ignored because corpus is already sealed.
        return [tuple(w) for w in sealed_words]

    v151._generate_scrambles = sealed_generate_scrambles

    raw_analysis = outdir / "RAW_v37_153_FULL24.json"
    macro_json = outdir / "RAW_v37_153_FULL24_macro.json"
    bootstrap_json = outdir / "RAW_v37_153_FULL24_bootstrap.json"
    log_path = outdir / "FULL24_RUN_v37_527.log"

    # Historical v37.153 frozen operational configuration.
    argv = [
        str(driver_path),
        "--fresh-count", str(CASE_COUNT),
        "--fresh-seed", "20261054",
        "--scramble-length", str(SCRAMBLE_LEN),
        "--start-h-offset", "3",
        "--max-h-offset", "5",
        "--portfolio-time-limit", "30",
        "--cert-time-limit", "30",
        "--prefix-node-cap", "1000000",
        "--k2-probe-cap", "22",
        "--terminal-call-cap", "16000",
        "--progress-every", "1",
        "--analysis-output", str(raw_analysis),
        "--macro-max-suffixes", "2",
        "--macro-min-hq", "3",
        "--macro-max-hq", "4",
        "--macro-telemetry-output", str(macro_json),
        "--d6-table", str(D6_REL).replace("\\", "/"),
        "--long-horizons", "6", "7",
        "--planted-k2-length", "4",
        "--p2-order", "auto",
        "--p2-auto-probe-nodes", "128",
        "--seed", "20260919",
        "--support-cache-max", "100000",
        "--fast-max-witnesses", "1",
        "--fast-probe-time-limit", "3",
        "--fast-probe-node-cap", "50000",
        "--fast-support-build-cap", "100000",
        "--output", str(bootstrap_json),
    ]

    print(f"driver                  : {DRIVER_REL.as_posix()}")
    print("scramble injection      : v151._generate_scrambles -> sealed FULL-24")
    print("frozen budgets          : H+3..+5, portfolio=30s, cert=30s, prefix=1,000,000, K2<=22, calls=16,000")

    old_argv = sys.argv[:]
    start = time.perf_counter()
    rc = None
    error = None

    with log_path.open("w", encoding="utf-8") as lf:
        tee_out = Tee(sys.__stdout__, lf)
        tee_err = Tee(sys.__stderr__, lf)
        try:
            sys.argv = argv
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                result = mod.main()
                rc = int(result or 0)
        except BaseException as exc:
            error = repr(exc)
            with contextlib.redirect_stdout(tee_out), contextlib.redirect_stderr(tee_err):
                print()
                print("FULL24 RUN ERROR:", error)
        finally:
            sys.argv = old_argv
            v151._generate_scrambles = original_generate

    total_wall = time.perf_counter() - start

    analysis = None
    if raw_analysis.is_file():
        try:
            analysis = json.loads(raw_analysis.read_text(encoding="utf-8"))
        except Exception as exc:
            error = error or f"analysis-json-read-failed:{exc!r}"

    found_pairs = recursive_scrambles(analysis) if analysis is not None else []
    found_unique = dedupe_in_order(v for _, v in found_pairs)
    actual_25 = [v for v in found_unique if len(v.split()) == SCRAMBLE_LEN]

    # We accept either explicit all-case reporting or no all-case scramble list.
    # If explicit case scrambles exist, they MUST match the seal exactly.
    if len(actual_25) >= CASE_COUNT:
        report_match = actual_25[:CASE_COUNT] == sealed_strings
        report_match_mode = "explicit_24"
    elif actual_25:
        # Partial explicit scrambles are allowed only if each is a member of the sealed corpus
        # and appears in the same relative order.
        indices = []
        ok = True
        last = -1
        for s in actual_25:
            try:
                idx = sealed_strings.index(s)
            except ValueError:
                ok = False
                break
            if idx <= last:
                ok = False
                break
            last = idx
            indices.append(idx)
        report_match = ok
        report_match_mode = f"partial_explicit_{len(actual_25)}"
    else:
        # The monkeypatched generator call contract itself proves the supplied corpus.
        report_match = True
        report_match_mode = "generator_contract_only"

    generator_contract_ok = (
        len(calls) == 1
        and calls[0]["fresh_count"] == CASE_COUNT
        and calls[0]["scramble_length"] == SCRAMBLE_LEN
    )

    receipt = {
        "schema": "cubelab.presentation.sealed-full24-operational.v37.527",
        "corpus_sha256": seal["corpus_sha256"],
        "seal_path": str(seal_path),
        "d6_sha256": d6_sha,
        "driver": DRIVER_REL.as_posix(),
        "scramble_injection": "v151._generate_scrambles",
        "generator_calls": calls,
        "generator_contract_ok": generator_contract_ok,
        "report_scramble_match": report_match,
        "report_scramble_match_mode": report_match_mode,
        "raw_analysis_exists": raw_analysis.is_file(),
        "raw_analysis_summary": classify_summary(analysis),
        "driver_rc": rc,
        "driver_error": error,
        "total_wrapper_wall_s": total_wall,
        "raw_analysis": str(raw_analysis),
        "raw_macro": str(macro_json),
        "raw_bootstrap": str(bootstrap_json),
        "log": str(log_path),
    }

    valid = bool(
        rc == 0
        and error is None
        and generator_contract_ok
        and report_match
        and raw_analysis.is_file()
    )
    receipt["valid_benchmark"] = valid

    receipt_path = outdir / "FULL24_OPERATIONAL_RECEIPT_v37_527.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, ensure_ascii=False), encoding="utf-8")

    print()
    print("# FULL-24 OPERATIONAL SUMMARY")
    print(f"driver rc               : {rc}")
    print(f"wrapper wall            : {total_wall:.3f}s")
    print(f"generator calls         : {calls}")
    print(f"generator contract OK   : {generator_contract_ok}")
    print(f"report match            : {report_match} ({report_match_mode})")
    print(f"analysis JSON exists    : {raw_analysis.is_file()}")
    print(f"receipt                 : {receipt_path}")
    print(f"log                     : {log_path}")

    if valid:
        print("DECISION                : SEALED_FULL24_OPERATIONAL_RUN_VALID")
        return 0

    print("DECISION                : SEALED_FULL24_OPERATIONAL_RUN_NOT_VALIDATED")
    if error:
        print(f"error                   : {error}")
    return 2

if __name__ == "__main__":
    raise SystemExit(main())

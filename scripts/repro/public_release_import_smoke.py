from __future__ import annotations

"""WeaveCube public-release import and exact-compatibility smoke test."""

import importlib

MODULES = (
    "cubelab.ato.six_coset_terminal",
    "cubelab.constructive.c4_first_witness_backend",
    "cubelab.constructive.exact_closure",
    "cubelab.column_families.context_aware_pair_engine_v37_360",
    "cubelab.global_field.shared_word_bitmap_v37_458",
    "cubelab.pdcc.engine",
    "cubelab_global_backbone.ida_kernel",
    "cubelab.constraint_engine.engine",
)


def main() -> int:
    failures: list[str] = []
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"FAIL import {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"PASS import {name}")

    try:
        from cubelab.domino_reduction_short_census import (
            IDENTITY_KEY,
            compose_packed,
            effect_from_word,
            inverse_packed,
            state_codes,
        )

        samples = (
            (),
            ("R",),
            ("R", "U", "R'", "U'"),
            ("F2", "U", "L'", "D2", "B"),
        )
        assert len(IDENTITY_KEY) == 40
        assert state_codes(IDENTITY_KEY) == tuple(range(20))
        for word in samples:
            effect = effect_from_word(word)
            assert len(effect) == 40
            assert compose_packed(inverse_packed(effect), effect) == IDENTITY_KEY
            assert compose_packed(effect, inverse_packed(effect)) == IDENTITY_KEY
        print("PASS exact packed-effect compatibility roundtrip")
    except Exception as exc:
        failures.append(f"packed-effect compatibility: {type(exc).__name__}: {exc}")
        print(f"FAIL packed-effect compatibility: {type(exc).__name__}: {exc}")

    if failures:
        print("\nPUBLIC RELEASE IMPORT SMOKE: FAIL")
        for failure in failures:
            print(" -", failure)
        return 1

    print("\nPUBLIC RELEASE IMPORT SMOKE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
from time import perf_counter
from typing import Any, Mapping, Sequence

import numpy as np

from cubelab.column_families.symbolic_mdd_v2 import (
    ResourceCapUnknown,
    SymbolicMDD,
)

from .direct_q_restrict import DirectRestrictUnknown
from .prepared_q_restrict import (
    execute_prepared_one,
    execute_prepared_two,
    prepare_pair_mask,
)
from .vectorized_one_quote import (
    VectorizedOneQuoteBank,
    prepare_vectorized_piece_masks,
    quote_vectorized_one_candidates_prepared,
)


FROZEN = "FROZEN_HYBRID"
DYNAMIC = "DYNAMIC_ONE"
RESCUE = "SELECTIVE_TWO_RESCUE"

COMPLETE = "COMPLETE"
EMPTY = "EXACT_EMPTY"
UNKNOWN = "UNKNOWN_RESOURCE"


@dataclass(frozen=True, slots=True)
class ClosureRequest:
    language: SymbolicMDD
    pending_pieces: tuple[int, ...]
    q_before: Sequence[int]
    target_masks: Sequence[int]
    rewrite: Any
    factor_mdds: Mapping[int, SymbolicMDD] = field(default_factory=dict)

    def pending(self) -> tuple[int, ...]:
        return tuple(sorted({int(v) for v in self.pending_pieces}))


@dataclass(frozen=True, slots=True)
class FrozenHybridPlan:
    name: str
    signature: str
    one_before_pair: tuple[int, ...]
    joint_pairs: tuple[tuple[int, int], ...]
    one_after_pair: tuple[int, ...]
    generic_tail: tuple[int, ...]

    def covered(self) -> tuple[int, ...]:
        values: list[int] = []
        values.extend(int(v) for v in self.one_before_pair)
        for a, b in self.joint_pairs:
            values.extend((int(a), int(b)))
        values.extend(int(v) for v in self.one_after_pair)
        values.extend(int(v) for v in self.generic_tail)

        if len(values) != len(set(values)):
            raise ValueError(f"duplicate piece in plan {self.name!r}")

        return tuple(sorted(values))


@dataclass(frozen=True, slots=True)
class SelectiveTwoPlan:
    name: str
    signature: str
    pair: tuple[int, int]


@dataclass(slots=True)
class ClosureResult:
    status: str
    backend: str
    signature: str
    language: SymbolicMDD | None
    trace: list[dict[str, Any]]
    timings: dict[str, float]
    pending_after: tuple[int, ...]
    reason: str | None = None


def _u32(value: int) -> bytes:
    value = int(value)
    if not 0 <= value < (1 << 32):
        raise ValueError("uint32 encoding overflow")
    return value.to_bytes(4, "little", signed=False)


def exact_closure_signature(request: ClosureRequest) -> str:
    """
    Exact execution-plan identity, not a geometry heuristic.
    """
    pending = request.pending()
    q_before = tuple(int(v) for v in request.q_before)
    targets = tuple(int(v) for v in request.target_masks)

    if len(q_before) < 20 or len(targets) < 20:
        raise ValueError("expected q_before/target masks for 20 pieces")

    transitions = np.asarray(
        request.rewrite.q_next,
        dtype=np.uint8,
    )

    if transitions.ndim != 3 or transitions.shape[1:] != (24, 18):
        raise ValueError("rewrite.q_next must be (piece_count,24,18)")

    digest = sha256()
    digest.update(b"CubeLab.ExactClosureSignature.v1\0")

    digest.update(
        str(request.language.structural_hash()).encode("utf-8")
    )
    digest.update(b"\0")
    digest.update(_u32(int(request.language.horizon)))
    digest.update(repr(request.language.previous_face).encode("utf-8"))
    digest.update(b"\0")
    digest.update(repr(request.language.scope_hash).encode("utf-8"))
    digest.update(b"\0")

    for piece in pending:
        if not 0 <= piece < transitions.shape[0]:
            raise ValueError(f"invalid pending piece {piece}")
        digest.update(bytes([piece]))
        digest.update(bytes([q_before[piece]]))
        digest.update(_u32(targets[piece]))
        digest.update(
            np.ascontiguousarray(
                transitions[piece],
                dtype=np.uint8,
            ).tobytes()
        )

    move_names = getattr(request.rewrite, "move_names", None)
    if move_names is not None:
        digest.update(b"MOVES\0")
        for name in move_names:
            digest.update(str(name).encode("utf-8"))
            digest.update(b"\0")

    if request.factor_mdds:
        digest.update(b"FACTORS\0")
        for piece in sorted(int(v) for v in request.factor_mdds):
            factor = request.factor_mdds[piece]
            digest.update(bytes([piece]))
            digest.update(str(factor.structural_hash()).encode("utf-8"))
            digest.update(b"\0")
            digest.update(_u32(int(factor.horizon)))
            digest.update(repr(factor.previous_face).encode("utf-8"))
            digest.update(b"\0")
            digest.update(repr(factor.scope_hash).encode("utf-8"))
            digest.update(b"\0")

    return digest.hexdigest()


class ExactClosureDispatcher:
    """
    Chooses only the execution kernel for one mandatory conjunction.
    It has zero solution-branch authority.
    """

    def __init__(
        self,
        *,
        quote_bank: VectorizedOneQuoteBank,
        frozen_plans: Sequence[FrozenHybridPlan] = (),
        rescue_plans: Sequence[SelectiveTwoPlan] = (),
        node_cap: int = 1_500_000,
        single_product_cap: int = 3_000_000,
        joint_product_cap: int = 4_000_000,
        one_deadline_s: float = 90.0,
        two_deadline_s: float = 110.0,
    ):
        self.quote_bank = quote_bank
        self.node_cap = int(node_cap)
        self.single_product_cap = int(single_product_cap)
        self.joint_product_cap = int(joint_product_cap)
        self.one_deadline_s = float(one_deadline_s)
        self.two_deadline_s = float(two_deadline_s)

        self.frozen = {}
        self.rescue = {}

        for plan in frozen_plans:
            if plan.signature in self.frozen:
                raise ValueError("duplicate frozen signature")
            self.frozen[plan.signature] = plan

        for plan in rescue_plans:
            if plan.signature in self.rescue:
                raise ValueError("duplicate rescue signature")
            self.rescue[plan.signature] = plan

    def select_backend(
        self,
        request: ClosureRequest,
    ) -> tuple[str, str, str | None]:
        signature = exact_closure_signature(request)

        frozen = self.frozen.get(signature)
        if frozen is not None:
            if frozen.covered() != request.pending():
                raise RuntimeError("frozen-plan coverage drift")
            return FROZEN, signature, frozen.name

        rescue = self.rescue.get(signature)
        if rescue is not None:
            a, b = map(int, rescue.pair)
            if a == b or a not in request.pending() or b not in request.pending():
                raise RuntimeError("rescue-plan coverage drift")
            return RESCUE, signature, rescue.name

        return DYNAMIC, signature, None

    def execute(
        self,
        request: ClosureRequest,
        *,
        force_backend: str | None = None,
    ) -> ClosureResult:
        selected, signature, _name = self.select_backend(request)
        backend = selected if force_backend is None else str(force_backend)

        try:
            if backend == FROZEN:
                plan = self.frozen.get(signature)
                if plan is None:
                    raise ValueError("no exact frozen plan for forced backend")
                return self._frozen(request, signature, plan)

            if backend == RESCUE:
                plan = self.rescue.get(signature)
                if plan is None:
                    raise ValueError("no exact rescue plan for forced backend")
                return self._rescue(request, signature, plan)

            if backend == DYNAMIC:
                return self._dynamic(request, signature)

            raise ValueError(f"unknown backend {backend!r}")

        except (DirectRestrictUnknown, ResourceCapUnknown) as exc:
            return ClosureResult(
                status=UNKNOWN,
                backend=backend,
                signature=signature,
                language=None,
                trace=[],
                timings={},
                pending_after=request.pending(),
                reason=str(exc),
            )

    def _one(
        self,
        language: SymbolicMDD,
        piece: int,
        request: ClosureRequest,
    ) -> tuple[SymbolicMDD | None, dict[str, Any], float, float]:
        t0 = perf_counter()

        prepared_map, _ = prepare_vectorized_piece_masks(
            language,
            pieces=(piece,),
            bank=self.quote_bank,
            q_before=request.q_before,
            targets=request.target_masks,
        )

        prepare_wall = perf_counter() - t0
        prepared = prepared_map[piece]

        if not prepared.root_live:
            return None, {"result": EMPTY}, prepare_wall, 0.0

        t0 = perf_counter()

        output, metrics = execute_prepared_one(
            prepared,
            node_cap=self.node_cap,
            product_cap=self.single_product_cap,
            deadline=perf_counter() + self.one_deadline_s,
        )

        execute_wall = perf_counter() - t0

        if int(output.root) == 0:
            return None, {"result": EMPTY}, prepare_wall, execute_wall

        return output, metrics, prepare_wall, execute_wall

    def _two(
        self,
        language: SymbolicMDD,
        a: int,
        b: int,
        request: ClosureRequest,
    ) -> tuple[SymbolicMDD | None, dict[str, Any], float, float]:
        t0 = perf_counter()

        prepared_map, _ = prepare_vectorized_piece_masks(
            language,
            pieces=(a, b),
            bank=self.quote_bank,
            q_before=request.q_before,
            targets=request.target_masks,
        )

        pair = prepare_pair_mask(
            prepared_map[a],
            prepared_map[b],
            deadline=perf_counter() + min(15.0, self.two_deadline_s),
        )

        prepare_wall = perf_counter() - t0

        if not prepared_map[a].root_live or not prepared_map[b].root_live:
            return None, {"result": EMPTY}, prepare_wall, 0.0

        t0 = perf_counter()

        output, metrics = execute_prepared_two(
            pair,
            node_cap=self.node_cap,
            product_cap=self.joint_product_cap,
            deadline=perf_counter() + self.two_deadline_s,
        )

        execute_wall = perf_counter() - t0

        if int(output.root) == 0:
            return None, {"result": EMPTY}, prepare_wall, execute_wall

        return output, metrics, prepare_wall, execute_wall

    def _frozen(
        self,
        request: ClosureRequest,
        signature: str,
        plan: FrozenHybridPlan,
    ) -> ClosureResult:
        language = request.language
        pending = set(request.pending())

        trace = []
        timings = {
            "prepare_s": 0.0,
            "execute_s": 0.0,
            "generic_s": 0.0,
        }

        started = perf_counter()

        def finish_empty() -> ClosureResult:
            timings["wall_s"] = perf_counter() - started
            return ClosureResult(
                status=EMPTY,
                backend=FROZEN,
                signature=signature,
                language=None,
                trace=trace,
                timings=timings,
                pending_after=tuple(sorted(pending)),
            )

        for piece in plan.one_before_pair:
            piece = int(piece)
            output, metrics, prep, exe = self._one(
                language, piece, request
            )
            timings["prepare_s"] += prep
            timings["execute_s"] += exe
            trace.append({
                "op": "ONE",
                "piece": piece,
                "result": EMPTY if output is None else COMPLETE,
                "output_nodes": None if output is None else int(output.node_count),
            })
            if output is None:
                return finish_empty()
            language = output
            pending.remove(piece)

        for a, b in plan.joint_pairs:
            a, b = int(a), int(b)
            output, metrics, prep, exe = self._two(
                language, a, b, request
            )
            timings["prepare_s"] += prep
            timings["execute_s"] += exe
            trace.append({
                "op": "TWO",
                "pieces": [a, b],
                "result": EMPTY if output is None else COMPLETE,
                "output_nodes": None if output is None else int(output.node_count),
            })
            if output is None:
                return finish_empty()
            language = output
            pending.remove(a)
            pending.remove(b)

        for piece in plan.one_after_pair:
            piece = int(piece)
            output, metrics, prep, exe = self._one(
                language, piece, request
            )
            timings["prepare_s"] += prep
            timings["execute_s"] += exe
            trace.append({
                "op": "ONE",
                "piece": piece,
                "result": EMPTY if output is None else COMPLETE,
                "output_nodes": None if output is None else int(output.node_count),
            })
            if output is None:
                return finish_empty()
            language = output
            pending.remove(piece)

        for piece in plan.generic_tail:
            piece = int(piece)
            factor = request.factor_mdds.get(piece)
            if factor is None:
                raise ValueError(f"missing factor_mdds[{piece}]")

            t0 = perf_counter()
            language = language.intersect(
                factor,
                node_cap=self.node_cap,
            )
            timings["generic_s"] += perf_counter() - t0

            pending.remove(piece)

            trace.append({
                "op": "GENERIC",
                "piece": piece,
                "output_nodes": int(language.node_count),
            })

            if int(language.root) == 0:
                return finish_empty()

        if pending:
            raise RuntimeError(f"frozen plan left pending {sorted(pending)}")

        timings["wall_s"] = perf_counter() - started

        return ClosureResult(
            status=COMPLETE,
            backend=FROZEN,
            signature=signature,
            language=language,
            trace=trace,
            timings=timings,
            pending_after=(),
        )

    def _dynamic(
        self,
        request: ClosureRequest,
        signature: str,
    ) -> ClosureResult:
        language = request.language
        pending = set(request.pending())

        trace = []
        timings = {
            "quote_s": 0.0,
            "execute_s": 0.0,
        }

        started = perf_counter()

        while pending:
            t0 = perf_counter()

            selected, quote_report = quote_vectorized_one_candidates_prepared(
                language,
                pieces=sorted(pending),
                bank=self.quote_bank,
                q_before=request.q_before,
                targets=request.target_masks,
            )

            timings["quote_s"] += perf_counter() - t0

            if not selected.root_live:
                timings["wall_s"] = perf_counter() - started
                return ClosureResult(
                    status=EMPTY,
                    backend=DYNAMIC,
                    signature=signature,
                    language=None,
                    trace=trace,
                    timings=timings,
                    pending_after=tuple(sorted(pending)),
                )

            t0 = perf_counter()

            output, metrics = execute_prepared_one(
                selected.prepared,
                node_cap=self.node_cap,
                product_cap=self.single_product_cap,
                deadline=perf_counter() + self.one_deadline_s,
            )

            timings["execute_s"] += perf_counter() - t0

            if int(output.root) == 0:
                timings["wall_s"] = perf_counter() - started
                return ClosureResult(
                    status=EMPTY,
                    backend=DYNAMIC,
                    signature=signature,
                    language=None,
                    trace=trace,
                    timings=timings,
                    pending_after=tuple(sorted(pending)),
                )

            trace.append({
                "op": "ONE",
                "piece": int(selected.piece),
                "exact_states": int(selected.exact_states),
                "exact_edges": int(selected.exact_edges),
                "quote_candidates": int(quote_report["candidate_count"]),
                "output_nodes": int(output.node_count),
            })

            language = output
            pending.remove(int(selected.piece))

        timings["wall_s"] = perf_counter() - started

        return ClosureResult(
            status=COMPLETE,
            backend=DYNAMIC,
            signature=signature,
            language=language,
            trace=trace,
            timings=timings,
            pending_after=(),
        )

    def _rescue(
        self,
        request: ClosureRequest,
        signature: str,
        plan: SelectiveTwoPlan,
    ) -> ClosureResult:
        pending = set(request.pending())
        a, b = int(plan.pair[0]), int(plan.pair[1])

        if a == b or a not in pending or b not in pending:
            raise RuntimeError("invalid registered rescue pair")

        started = perf_counter()

        output, metrics, prep, exe = self._two(
            request.language,
            a,
            b,
            request,
        )

        trace = [{
            "op": "TWO_RESCUE",
            "plan": plan.name,
            "pieces": [a, b],
            "result": EMPTY if output is None else COMPLETE,
        }]

        if output is None:
            return ClosureResult(
                status=EMPTY,
                backend=RESCUE,
                signature=signature,
                language=None,
                trace=trace,
                timings={
                    "prepare_s": prep,
                    "execute_s": exe,
                    "wall_s": perf_counter() - started,
                },
                pending_after=tuple(sorted(pending)),
            )

        pending.remove(a)
        pending.remove(b)

        remainder = ClosureRequest(
            language=output,
            pending_pieces=tuple(sorted(pending)),
            q_before=request.q_before,
            target_masks=request.target_masks,
            rewrite=request.rewrite,
            factor_mdds=request.factor_mdds,
        )

        dynamic = self._dynamic(
            remainder,
            exact_closure_signature(remainder),
        )

        dynamic.backend = RESCUE
        dynamic.signature = signature
        dynamic.trace = trace + dynamic.trace
        dynamic.timings["rescue_prepare_s"] = prep
        dynamic.timings["rescue_execute_s"] = exe
        dynamic.timings["wall_s"] = perf_counter() - started

        return dynamic


__all__ = [
    "FROZEN",
    "DYNAMIC",
    "RESCUE",
    "COMPLETE",
    "EMPTY",
    "UNKNOWN",
    "ClosureRequest",
    "ClosureResult",
    "ExactClosureDispatcher",
    "FrozenHybridPlan",
    "SelectiveTwoPlan",
    "exact_closure_signature",
]

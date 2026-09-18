"""Constraint-first obligation and gap contracts for later column composition."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping

from .model import PIECE_INDEX, PIECE_NAMES
from .moves import MOVES, normalize_move_token
from .state import PDCCState, transition_piece_pose
from .trace import ColumnRecord, ColumnTrace, GapAudit


class FeasibilityStatus(str, Enum):
    SAT = "SAT"
    UNSAT_PROVEN = "UNSAT_PROVEN"
    UNKNOWN_EMBEDDING_CAP = "UNKNOWN_EMBEDDING_CAP"
    UNKNOWN_NODE_CAP = "UNKNOWN_NODE_CAP"
    UNKNOWN_ENUMERATION_CAP = "UNKNOWN_ENUMERATION_CAP"


class ObligationTag(str, Enum):
    TRANSPORT = "transport"
    PERMUTATION = "permutation"
    ORIENTATION = "orientation"
    RELATION = "relation"
    MERGE = "merge"
    CLOSURE = "closure"
    FUSION = "fusion"


@dataclass(frozen=True, slots=True)
class AtomicObligation:
    """Exact local pre/move/post contract owned by one or more pieces."""

    obligation_id: str
    scope: tuple[str, ...]
    move: str
    pre_poses: tuple[tuple[str, int], ...]
    post_poses: tuple[tuple[str, int], ...]
    tags: frozenset[ObligationTag]

    @classmethod
    def create(
        cls,
        obligation_id: str,
        scope: Iterable[str],
        move: str,
        pre_poses: Mapping[str, int],
        post_poses: Mapping[str, int],
        tags: Iterable[ObligationTag],
    ) -> "AtomicObligation":
        normalized_scope = tuple(sorted(set(scope), key=PIECE_INDEX.__getitem__))
        if not normalized_scope:
            raise ValueError("Obligation scope cannot be empty")
        if set(normalized_scope) - set(PIECE_NAMES):
            raise ValueError("Obligation contains an unknown piece")
        if set(pre_poses) != set(normalized_scope):
            raise ValueError("pre_poses keys must exactly match scope")
        if set(post_poses) != set(normalized_scope):
            raise ValueError("post_poses keys must exactly match scope")
        obligation = cls(
            obligation_id=obligation_id,
            scope=normalized_scope,
            move=normalize_move_token(move),
            pre_poses=tuple((piece, pre_poses[piece]) for piece in normalized_scope),
            post_poses=tuple((piece, post_poses[piece]) for piece in normalized_scope),
            tags=frozenset(tags),
        )
        errors = obligation.static_errors()
        if errors:
            raise ValueError("Invalid obligation: " + "; ".join(errors))
        return obligation

    @property
    def pre(self) -> dict[str, int]:
        return dict(self.pre_poses)

    @property
    def post(self) -> dict[str, int]:
        return dict(self.post_poses)

    def static_errors(self) -> tuple[str, ...]:
        errors: list[str] = []
        if self.move not in MOVES:
            errors.append(f"unknown move {self.move}")
            return tuple(errors)
        pre = self.pre
        post = self.post
        for piece in self.scope:
            expected, _ = transition_piece_pose(piece, pre[piece], self.move)
            if expected != post[piece]:
                errors.append(
                    f"{piece}: expected post pose {expected}, got {post[piece]}"
                )
        return tuple(errors)

    def audit_column(self, column: ColumnRecord) -> tuple[bool, tuple[str, ...]]:
        errors: list[str] = []
        if column.move != self.move:
            errors.append(f"move mismatch: {column.move} != {self.move}")
        pre = self.pre
        post = self.post
        for piece in self.scope:
            if column.before.pose_id(piece) != pre[piece]:
                errors.append(f"{piece}: pre-pose mismatch")
            if column.after.pose_id(piece) != post[piece]:
                errors.append(f"{piece}: post-pose mismatch")
        return not errors, tuple(errors)


@dataclass(frozen=True, slots=True)
class StaticCompatibility:
    compatible: bool
    reasons: tuple[str, ...]


def static_compatibility(
    left: AtomicObligation, right: AtomicObligation
) -> StaticCompatibility:
    reasons: list[str] = []
    if left.move != right.move:
        reasons.append(f"move mismatch: {left.move}/{right.move}")
    overlap = set(left.scope) & set(right.scope)
    left_pre, right_pre = left.pre, right.pre
    left_post, right_post = left.post, right.post
    for piece in sorted(overlap, key=PIECE_INDEX.__getitem__):
        if left_pre[piece] != right_pre[piece]:
            reasons.append(f"{piece}: overlapping pre-pose conflict")
        if left_post[piece] != right_post[piece]:
            reasons.append(f"{piece}: overlapping post-pose conflict")
    return StaticCompatibility(compatible=not reasons, reasons=tuple(reasons))


@dataclass(frozen=True, slots=True)
class GapContract:
    piece: str
    entry_pose: int
    exit_pose: int
    require_pure_inactive: bool = False

    def audit(
        self, trace: ColumnTrace, start_column: int, end_column: int
    ) -> tuple[bool, GapAudit, tuple[str, ...]]:
        gap = trace.gap_audit(self.piece, start_column, end_column)
        errors: list[str] = []
        if gap.entry_pose != self.entry_pose:
            errors.append("entry pose mismatch")
        if gap.exit_pose != self.exit_pose:
            errors.append("exit pose mismatch")
        if self.require_pure_inactive and not gap.pure_inactive:
            errors.append("pure inactivity required but active columns exist")
        return not errors, gap, tuple(errors)


@dataclass(frozen=True, slots=True)
class ScheduleAudit:
    status: FeasibilityStatus
    obligation_results: tuple[tuple[str, int, bool, tuple[str, ...]], ...]
    gap_results: tuple[tuple[str, int, int, bool, tuple[str, ...]], ...]
    full_replay_valid: bool
    final_solved: bool


def audit_schedule(
    trace: ColumnTrace,
    placements: Mapping[str, tuple[AtomicObligation, int]],
    gaps: Iterable[tuple[GapContract, int, int]] = (),
    *,
    require_solved: bool = False,
) -> ScheduleAudit:
    obligation_results: list[tuple[str, int, bool, tuple[str, ...]]] = []
    all_valid = True
    for obligation_id, (obligation, column_index) in placements.items():
        if obligation_id != obligation.obligation_id:
            valid = False
            errors = ("placement key differs from obligation_id",)
        elif not 0 <= column_index < len(trace.columns):
            valid = False
            errors = ("column index out of range",)
        else:
            valid, errors = obligation.audit_column(trace.columns[column_index])
        all_valid &= valid
        obligation_results.append((obligation_id, column_index, valid, errors))

    gap_results: list[tuple[str, int, int, bool, tuple[str, ...]]] = []
    for contract, start, end in gaps:
        valid, _, errors = contract.audit(trace, start, end)
        all_valid &= valid
        gap_results.append((contract.piece, start, end, valid, errors))

    replay_valid = trace.full_replay_matches()
    all_valid &= replay_valid
    solved = trace.final.is_solved()
    if require_solved:
        all_valid &= solved

    return ScheduleAudit(
        status=FeasibilityStatus.SAT if all_valid else FeasibilityStatus.UNSAT_PROVEN,
        obligation_results=tuple(obligation_results),
        gap_results=tuple(gap_results),
        full_replay_valid=replay_valid,
        final_solved=solved,
    )

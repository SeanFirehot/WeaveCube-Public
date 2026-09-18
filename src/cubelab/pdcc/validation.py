"""Deterministic structural validation for the PDCC pose core."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import random

from .analysis import native_boundary, rigid_bundles, structure_signature
from .group import CUBE_ROTATIONS
from .model import CORNER_NAMES, EDGE_NAMES, PIECE_NAMES, PIECE_SPECS, slots_for_kind
from .moves import MOVE_ORDER, MOVES, inverse_move
from .orientation import ORIENTATION_SYSTEM
from .state import PDCCState, transition_piece_pose
from .tables import LOCAL_TRANSITION_TABLE


@dataclass(frozen=True, slots=True)
class ValidationReport:
    passed: bool
    checks: tuple[tuple[str, bool, str], ...]
    local_transition_count: int
    cocycle_case_count: int
    random_replay_steps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_structural_validation(
    *, random_seed: int = 102001, random_steps: int = 5000
) -> ValidationReport:
    checks: list[tuple[str, bool, str]] = []

    def record(name: str, condition: bool, detail: str = "") -> None:
        checks.append((name, bool(condition), detail))

    group = CUBE_ROTATIONS
    record("rotation_group_order", len(group.elements) == 24, str(len(group.elements)))
    record("identity_is_zero", group.identity == 0)

    group_ok = True
    for left in range(24):
        if group.compose(group.identity, left) != left:
            group_ok = False
        if group.compose(left, group.identity) != left:
            group_ok = False
        inverse = group.inverse(left)
        if group.compose(left, inverse) != group.identity:
            group_ok = False
        if group.compose(inverse, left) != group.identity:
            group_ok = False
    record("group_identity_inverse", group_ok)

    solved = PDCCState.solved()
    move_relations_ok = True
    for face in "URFDLB":
        quarter = face
        inverse = inverse_move(quarter)
        half = f"{face}2"
        if not solved.apply_word((quarter, inverse)).is_solved():
            move_relations_ok = False
        if not solved.apply_word((quarter,) * 4).is_solved():
            move_relations_ok = False
        if not solved.apply_word((half, half)).is_solved():
            move_relations_ok = False
    record("move_inverse_full_turn_relations", move_relations_ok)

    opposite_commute_ok = True
    for left, right in (("U", "D"), ("R", "L"), ("F", "B")):
        if solved.apply_word((left, right)) != solved.apply_word((right, left)):
            opposite_commute_ok = False
    record("opposite_faces_commute", opposite_commute_ok)

    active_support_ok = True
    for move in MOVE_ORDER:
        _, active_corners, active_edges = solved.transition(move)
        if len(active_corners) != 4 or len(active_edges) != 4:
            active_support_ok = False
    record("dual_table_4_corner_4_edge_support", active_support_ok)

    local_transition_count = len(LOCAL_TRANSITION_TABLE)
    local_transition_ok = all(
        0 <= entry.pose_after < 24
        for entry in LOCAL_TRANSITION_TABLE.values()
    )
    record(
        "local_transition_table",
        local_transition_ok and local_transition_count == 8640,
        str(local_transition_count),
    )

    coordinate_table_ok = True
    for piece, spec in PIECE_SPECS.items():
        seen: set[int] = set()
        for slot in slots_for_kind(spec.kind):
            for orientation in range(spec.orientation_order):
                pose = ORIENTATION_SYSTEM.pose(piece, slot, orientation)
                seen.add(pose)
                if ORIENTATION_SYSTEM.orientation(piece, pose) != orientation:
                    coordinate_table_ok = False
        if len(seen) != 24:
            coordinate_table_ok = False
    record("pose_coordinate_bijection", coordinate_table_ok)

    cocycle_case_count = 0
    cocycle_ok = True
    for piece, spec in PIECE_SPECS.items():
        for slot in slots_for_kind(spec.kind):
            for first_rotation in range(24):
                middle_slot = ORIENTATION_SYSTEM.target_slot(
                    piece, first_rotation, slot
                )
                first_delta = ORIENTATION_SYSTEM.cocycle(
                    piece, first_rotation, slot
                )
                for second_rotation in range(24):
                    combined = group.compose(second_rotation, first_rotation)
                    left_delta = ORIENTATION_SYSTEM.cocycle(piece, combined, slot)
                    right_delta = (
                        ORIENTATION_SYSTEM.cocycle(
                            piece, second_rotation, middle_slot
                        )
                        + first_delta
                    ) % spec.orientation_order
                    cocycle_case_count += 1
                    if left_delta != right_delta:
                        cocycle_ok = False
                        break
                if not cocycle_ok:
                    break
            if not cocycle_ok:
                break
        if not cocycle_ok:
            break
    record("orientation_cocycle_law", cocycle_ok, str(cocycle_case_count))

    rng = random.Random(random_seed)
    state = solved
    random_legality_ok = True
    for _ in range(random_steps):
        state = state.apply(rng.choice(MOVE_ORDER))
        if not state.legality().legal:
            random_legality_ok = False
            break
    record("deterministic_random_legality_replay", random_legality_ok, str(random_steps))

    f_state = solved.apply("F")
    f_sizes = tuple(sorted((len(bundle) for bundle in rigid_bundles(f_state)), reverse=True))
    record("F_bundle_control", f_sizes == (12, 8), str(f_sizes))
    record("F_native_boundary_control", len(native_boundary(f_state)) == 4, str(len(native_boundary(f_state))))

    fr_state = solved.apply_word("F R")
    fr_sizes = tuple(sorted((len(bundle) for bundle in rigid_bundles(fr_state)), reverse=True))
    record("FR_bundle_control", fr_sizes == (7, 5, 5, 3), str(fr_sizes))
    record("FR_native_boundary_control", len(native_boundary(fr_state)) == 8, str(len(native_boundary(fr_state))))

    superflip = PDCCState.superflip()
    superflip_report = superflip.legality()
    superflip_ok = (
        superflip_report.legal
        and all(superflip.slot(edge) == edge for edge in EDGE_NAMES)
        and all(superflip.orientation(edge) == 1 for edge in EDGE_NAMES)
        and all(superflip.pose_id(corner) == group.identity for corner in CORNER_NAMES)
    )
    record("superflip_coordinate_control", superflip_ok)

    roundtrip_word = "F R U R' U' F'"
    roundtrip = solved.apply_word(roundtrip_word)
    replay_inverse_ok = roundtrip.apply_word(
        tuple(inverse_move(move) for move in reversed(roundtrip_word.split()))
    ).is_solved()
    record("word_inverse_roundtrip", replay_inverse_ok)

    signature_ok = structure_signature(solved).rigid_bundle_sizes == (20,)
    record("solved_structure_signature", signature_ok)

    return ValidationReport(
        passed=all(result for _, result, _ in checks),
        checks=tuple(checks),
        local_transition_count=local_transition_count,
        cocycle_case_count=cocycle_case_count,
        random_replay_steps=random_steps,
    )

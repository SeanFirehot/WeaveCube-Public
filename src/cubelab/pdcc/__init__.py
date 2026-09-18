"""Pose–Difference Column Calculus (PDCC) exact pose core."""

from .analysis import (
    StructureSignature,
    current_adjacency,
    native_boundary,
    relative_difference,
    rigid_bundles,
    structure_signature,
    task_bundles,
    twist_clusters,
)
from .engine import PDCCEngine
from .group import CUBE_ROTATIONS, Matrix3, RotationGroup, Vector3
from .model import (
    CORNER_NAMES,
    EDGE_NAMES,
    NATIVE_ADJACENCY,
    PIECE_NAMES,
    PieceKind,
)
from .moves import MOVE_ORDER, MOVES, MoveSpec, inverse_move, inverse_word, parse_word
from .obligations import (
    AtomicObligation,
    FeasibilityStatus,
    GapContract,
    ObligationTag,
    ScheduleAudit,
    StaticCompatibility,
    audit_schedule,
    static_compatibility,
)
from .orientation import ORIENTATION_SYSTEM, OrientationSystem
from .serialization import (
    FORMAT_VERSION,
    core_table_snapshot,
    load_core_table_snapshot,
    snapshot_sha256,
    write_core_table_snapshot,
)
from .state import LegalityReport, PDCCState, slot_for_pose, transition_piece_pose
from .tables import LOCAL_TRANSITION_TABLE, LocalTransition, local_transition
from .trace import (
    ColumnRecord,
    ColumnTrace,
    GapAudit,
    PieceCell,
    TwistEpisode,
    build_trace,
)
from .validation import ValidationReport, run_structural_validation

__all__ = [
    "AtomicObligation",
    "CUBE_ROTATIONS",
    "CORNER_NAMES",
    "ColumnRecord",
    "ColumnTrace",
    "EDGE_NAMES",
    "FORMAT_VERSION",
    "FeasibilityStatus",
    "GapAudit",
    "GapContract",
    "LOCAL_TRANSITION_TABLE",
    "LegalityReport",
    "LocalTransition",
    "MOVE_ORDER",
    "MOVES",
    "Matrix3",
    "MoveSpec",
    "NATIVE_ADJACENCY",
    "ORIENTATION_SYSTEM",
    "ObligationTag",
    "OrientationSystem",
    "PDCCEngine",
    "PDCCState",
    "PIECE_NAMES",
    "PieceCell",
    "PieceKind",
    "RotationGroup",
    "ScheduleAudit",
    "StaticCompatibility",
    "StructureSignature",
    "TwistEpisode",
    "ValidationReport",
    "Vector3",
    "audit_schedule",
    "build_trace",
    "core_table_snapshot",
    "current_adjacency",
    "inverse_move",
    "load_core_table_snapshot",
    "local_transition",
    "inverse_word",
    "native_boundary",
    "parse_word",
    "relative_difference",
    "rigid_bundles",
    "run_structural_validation",
    "slot_for_pose",
    "snapshot_sha256",
    "static_compatibility",
    "structure_signature",
    "task_bundles",
    "transition_piece_pose",
    "twist_clusters",
    "write_core_table_snapshot",
]

"""Deterministic serialization of PDCC core definitions and transition tables."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .group import CUBE_ROTATIONS
from .model import NATIVE_ADJACENCY, PIECE_NAMES, PIECE_SPECS, slots_for_kind
from .moves import MOVE_ORDER, MOVES
from .orientation import ORIENTATION_SYSTEM
from .tables import LOCAL_TRANSITION_TABLE

FORMAT_VERSION = "pdcc-core-tables-v1"


def core_table_snapshot() -> dict[str, Any]:
    pieces = []
    for piece in PIECE_NAMES:
        spec = PIECE_SPECS[piece]
        pieces.append(
            {
                "name": piece,
                "kind": spec.kind.value,
                "solved_position": list(spec.solved_position),
                "ordered_stickers": [list(vector) for vector in spec.ordered_stickers],
                "orientation_order": spec.orientation_order,
                "generator_pose": ORIENTATION_SYSTEM.generator[piece],
                "sections": {
                    slot: ORIENTATION_SYSTEM.section_pose[(piece, slot)]
                    for slot in slots_for_kind(spec.kind)
                },
            }
        )

    moves = [
        {
            "name": move,
            "face": MOVES[move].face,
            "rotation_id": MOVES[move].rotation_id,
            "axis": MOVES[move].axis,
            "layer": MOVES[move].layer,
            "amount": MOVES[move].amount,
        }
        for move in MOVE_ORDER
    ]

    transitions = []
    for piece in PIECE_NAMES:
        for pose in range(24):
            for move in MOVE_ORDER:
                entry = LOCAL_TRANSITION_TABLE[(piece, pose, move)]
                transitions.append(
                    {
                        "piece": piece,
                        "pose_before": pose,
                        "move": move,
                        "active": entry.active,
                        "pose_after": entry.pose_after,
                        "slot_before": entry.slot_before,
                        "slot_after": entry.slot_after,
                        "orientation_before": entry.orientation_before,
                        "orientation_after": entry.orientation_after,
                    }
                )

    return {
        "format_version": FORMAT_VERSION,
        "rotation_count": len(CUBE_ROTATIONS.elements),
        "rotations": [
            [list(row) for row in matrix] for matrix in CUBE_ROTATIONS.elements
        ],
        "moves": moves,
        "pieces": pieces,
        "native_adjacency": [list(link) for link in NATIVE_ADJACENCY],
        "local_transition_count": len(transitions),
        "local_transitions": transitions,
    }


def canonical_json_bytes(snapshot: dict[str, Any]) -> bytes:
    return json.dumps(
        snapshot,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def snapshot_sha256(snapshot: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(snapshot)).hexdigest()


def write_core_table_snapshot(path: str | Path) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot = core_table_snapshot()
    output.write_text(
        json.dumps(snapshot, ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    digest = snapshot_sha256(snapshot)
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{digest}  {output.name}\n", encoding="ascii"
    )
    return digest


def load_core_table_snapshot(path: str | Path) -> dict[str, Any]:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    if snapshot.get("format_version") != FORMAT_VERSION:
        raise ValueError("Unsupported PDCC core-table format")
    if snapshot.get("rotation_count") != 24:
        raise ValueError("Snapshot does not contain 24 rotations")
    if snapshot.get("local_transition_count") != 8640:
        raise ValueError("Snapshot does not contain 8640 local transitions")
    return snapshot

"""
CubeLab piece mathematics.

This module treats movable cube pieces as primary mathematical objects.

A cube has 20 movable pieces:
- 8 corners
- 12 edges

Each piece is identified by its home position:
- Corners: UFR, UBR, UBL, UFL, DFR, DRB, DBL, DLF
- Edges:   UF, UR, UB, UL, FR, BR, BL, FL, DF, DR, DB, DL

Orientation is derived from sticker normals and ordered local frames.
It is not stored as independent state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


Vec3 = Tuple[int, int, int]
Face = str
PieceName = str


FACE_TO_NORMAL: Dict[Face, Vec3] = {
    "U": (0, 1, 0),
    "D": (0, -1, 0),
    "F": (0, 0, 1),
    "B": (0, 0, -1),
    "R": (1, 0, 0),
    "L": (-1, 0, 0),
}

NORMAL_TO_FACE: Dict[Vec3, Face] = {v: k for k, v in FACE_TO_NORMAL.items()}


# Ordered local frames.
#
# These are not merely sets of faces.
# The order is part of the orientation convention.
CORNER_FRAMES: Dict[PieceName, Tuple[Face, Face, Face]] = {
    "UFR": ("U", "R", "F"),
    "UBR": ("U", "B", "R"),
    "UBL": ("U", "L", "B"),
    "UFL": ("U", "F", "L"),
    "DFR": ("D", "F", "R"),
    "DRB": ("D", "R", "B"),
    "DBL": ("D", "B", "L"),
    "DLF": ("D", "L", "F"),
}

EDGE_FRAMES: Dict[PieceName, Tuple[Face, Face]] = {
    "UF": ("U", "F"),
    "UR": ("U", "R"),
    "UB": ("U", "B"),
    "UL": ("U", "L"),
    "FR": ("F", "R"),
    "BR": ("B", "R"),
    "BL": ("B", "L"),
    "FL": ("F", "L"),
    "DF": ("D", "F"),
    "DR": ("D", "R"),
    "DB": ("D", "B"),
    "DL": ("D", "L"),
}

CORNER_ORDER: Tuple[PieceName, ...] = tuple(CORNER_FRAMES.keys())
EDGE_ORDER: Tuple[PieceName, ...] = tuple(EDGE_FRAMES.keys())
PIECE_ORDER: Tuple[PieceName, ...] = CORNER_ORDER + EDGE_ORDER


@dataclass
class PieceState:
    """
    Mathematical state of a movable cube piece.

    `piece` is the home position ID, such as UFR or UF.
    `position` is the current position ID.
    `orientation` is derived from sticker normals.

    For corners:
        orientation is 0, 1, or 2.

    For edges:
        orientation is 0 or 1.
    """

    piece: PieceName
    kind: str
    position: PieceName
    orientation: int

    home_frame: Tuple[Face, ...]
    current_frame: Tuple[Face, ...]

    reference_face: Face
    current_reference_face: Face

    # Maps each original face of the piece to its current face.
    #
    # Example:
    #   UFR piece after some moves:
    #   {
    #       "U": "R",
    #       "F": "U",
    #       "R": "F",
    #   }
    face_map: Dict[Face, Face]

    @property
    def orientation_label(self) -> str:
        if self.kind == "corner":
            if self.orientation == 0:
                return "Normal"
            return f"Twist {self.orientation}"
        if self.kind == "edge":
            return "Normal" if self.orientation == 0 else "Flipped"
        return str(self.orientation)


def face_from_normal(normal: Sequence[int]) -> Face:
    vec = _as_vec3(normal)

    if vec not in NORMAL_TO_FACE:
        raise ValueError(f"Invalid sticker normal: {normal}")

    return NORMAL_TO_FACE[vec]


def normal_from_face(face: Face) -> Vec3:
    if face not in FACE_TO_NORMAL:
        raise ValueError(f"Invalid face: {face}")

    return FACE_TO_NORMAL[face]


def position_from_coord(pos: Sequence[int]) -> Optional[PieceName]:
    """
    Converts a cubie coordinate to a canonical piece position label.

    Examples:
        (1, 1, 1)   -> UFR
        (1, 1, 0)   -> UR
        (0, 1, 1)   -> UF
        (0, 0, 1)   -> None   # center, not movable piece
    """

    faces = faces_from_coord(pos)

    if len(faces) == 3:
        return _find_position_by_face_set(CORNER_FRAMES, faces)

    if len(faces) == 2:
        return _find_position_by_face_set(EDGE_FRAMES, faces)

    return None


def coord_from_position(position: PieceName) -> Vec3:
    """
    Converts a piece position label to a cubie coordinate.

    Example:
        UFR -> (1, 1, 1)
        UF  -> (0, 1, 1)
    """

    if position in CORNER_FRAMES:
        frame = CORNER_FRAMES[position]
    elif position in EDGE_FRAMES:
        frame = EDGE_FRAMES[position]
    else:
        raise ValueError(f"Unknown piece position: {position}")

    x = 0
    y = 0
    z = 0

    if "R" in frame:
        x = 1
    elif "L" in frame:
        x = -1

    if "U" in frame:
        y = 1
    elif "D" in frame:
        y = -1

    if "F" in frame:
        z = 1
    elif "B" in frame:
        z = -1

    return (x, y, z)


def faces_from_coord(pos: Sequence[int]) -> Tuple[Face, ...]:
    x, y, z = _as_vec3(pos)
    faces: List[Face] = []

    if y == 1:
        faces.append("U")
    elif y == -1:
        faces.append("D")

    if z == 1:
        faces.append("F")
    elif z == -1:
        faces.append("B")

    if x == 1:
        faces.append("R")
    elif x == -1:
        faces.append("L")

    return tuple(faces)


def kind_of_piece(piece: PieceName) -> str:
    if piece in CORNER_FRAMES:
        return "corner"

    if piece in EDGE_FRAMES:
        return "edge"

    raise ValueError(f"Unknown piece: {piece}")


def frame_of_position(position: PieceName) -> Tuple[Face, ...]:
    if position in CORNER_FRAMES:
        return CORNER_FRAMES[position]

    if position in EDGE_FRAMES:
        return EDGE_FRAMES[position]

    raise ValueError(f"Unknown position: {position}")


def analyze_cube(cube: object) -> List[PieceState]:
    """
    Extracts piece states from a CubeModel-like object.

    The object must have a `stickers` attribute.
    Each sticker is expected to have:
        - pos
        - normal
        - origin or origin_key

    This is intentionally loose so it can work with the current prototype.
    """

    stickers = getattr(cube, "stickers", None)

    if stickers is None:
        raise TypeError("cube object must have a 'stickers' attribute")

    return analyze_stickers(stickers)


def analyze_stickers(stickers: Iterable[dict]) -> List[PieceState]:
    """
    Converts sticker-level state into 20 piece-level states.

    Centers are ignored.
    Movable stickers are grouped by their home piece.
    """

    groups: Dict[PieceName, List[dict]] = {}

    for sticker in stickers:
        home_piece = infer_home_piece(sticker)

        if home_piece is None:
            continue

        if home_piece not in PIECE_ORDER:
            continue

        groups.setdefault(home_piece, []).append(sticker)

    states: List[PieceState] = []

    for piece in PIECE_ORDER:
        if piece not in groups:
            continue

        states.append(analyze_piece(piece, groups[piece]))

    return states


def analyze_piece(piece: PieceName, stickers: Sequence[dict]) -> PieceState:
    kind = kind_of_piece(piece)
    home_frame = frame_of_position(piece)

    expected_sticker_count = 3 if kind == "corner" else 2

    if len(stickers) != expected_sticker_count:
        raise ValueError(
            f"{piece} should have {expected_sticker_count} stickers, "
            f"but got {len(stickers)}"
        )

    current_pos = position_from_coord(stickers[0]["pos"])

    if current_pos is None:
        raise ValueError(f"Cannot determine current position for piece {piece}")

    current_frame = frame_of_position(current_pos)

    face_map: Dict[Face, Face] = {}

    for sticker in stickers:
        original_face = infer_original_face(sticker)
        current_face = face_from_normal(sticker["normal"])
        face_map[original_face] = current_face

    if kind == "corner":
        orientation, reference_face, current_reference_face = compute_corner_orientation(
            home_frame=home_frame,
            current_frame=current_frame,
            face_map=face_map,
        )
    else:
        orientation, reference_face, current_reference_face = compute_edge_orientation(
            home_frame=home_frame,
            current_frame=current_frame,
            face_map=face_map,
        )

    return PieceState(
        piece=piece,
        kind=kind,
        position=current_pos,
        orientation=orientation,
        home_frame=home_frame,
        current_frame=current_frame,
        reference_face=reference_face,
        current_reference_face=current_reference_face,
        face_map=face_map,
    )


def compute_corner_orientation(
    home_frame: Tuple[Face, Face, Face],
    current_frame: Tuple[Face, Face, Face],
    face_map: Dict[Face, Face],
) -> Tuple[int, Face, Face]:
    """
    Corner orientation convention.

    The reference sticker is the U/D sticker of the corner.
    Orientation is the index of the current reference face
    inside the current position's ordered frame.

    Example:
        current frame = (U, R, F)

        U/D sticker on U -> 0
        U/D sticker on R -> 1
        U/D sticker on F -> 2
    """

    reference_face = "U" if "U" in home_frame else "D"
    current_reference_face = face_map[reference_face]

    if current_reference_face not in current_frame:
        raise ValueError(
            f"Reference face {current_reference_face} is not in current frame "
            f"{current_frame}"
        )

    orientation = current_frame.index(current_reference_face)

    return orientation, reference_face, current_reference_face


def compute_edge_orientation(
    home_frame: Tuple[Face, Face],
    current_frame: Tuple[Face, Face],
    face_map: Dict[Face, Face],
) -> Tuple[int, Face, Face]:
    """
    Edge orientation convention.

    The edge is normal if the ordered image of its home frame
    matches the ordered current frame.

    It is flipped if the order is reversed.
    """

    reference_face = home_frame[0]
    current_reference_face = face_map[reference_face]

    mapped_frame = tuple(face_map[face] for face in home_frame)

    if mapped_frame == current_frame:
        orientation = 0
    elif mapped_frame == tuple(reversed(current_frame)):
        orientation = 1
    else:
        raise ValueError(
            f"Invalid edge frame mapping. "
            f"home_frame={home_frame}, mapped_frame={mapped_frame}, "
            f"current_frame={current_frame}"
        )

    return orientation, reference_face, current_reference_face


def orientation_checksums(states: Iterable[PieceState]) -> Dict[str, int]:
    corner_sum = 0
    edge_sum = 0

    for state in states:
        if state.kind == "corner":
            corner_sum += state.orientation
        elif state.kind == "edge":
            edge_sum += state.orientation

    return {
        "corner_orientation_sum_mod_3": corner_sum % 3,
        "edge_orientation_sum_mod_2": edge_sum % 2,
    }


def state_by_piece(states: Iterable[PieceState]) -> Dict[PieceName, PieceState]:
    return {state.piece: state for state in states}


def format_piece_state(state: PieceState) -> str:
    face_map_text = ", ".join(
        f"{original}->{current}"
        for original, current in sorted(state.face_map.items())
    )

    return (
        f"Piece: {state.piece}\n"
        f"Type: {state.kind}\n"
        f"Home frame: {state.home_frame}\n"
        f"Current position: {state.position}\n"
        f"Current frame: {state.current_frame}\n"
        f"Reference sticker: {state.reference_face}\n"
        f"Reference now points to: {state.current_reference_face}\n"
        f"Orientation: {state.orientation} ({state.orientation_label})\n"
        f"Face map: {face_map_text}"
    )


def infer_home_piece(sticker: dict) -> Optional[PieceName]:
    """
    Infers the home piece of a sticker.

    This function is deliberately tolerant because the prototype has evolved.
    It supports several possible sticker formats.

    Preferred prototype format:
        sticker["origin"] = original cubie coordinate

    Also supported:
        sticker["origin_key"] = (original_position, original_normal)
    """

    for key in ("piece", "home_piece", "home", "home_position"):
        value = sticker.get(key)

        if isinstance(value, str):
            return canonical_piece_name(value)

    origin = sticker.get("origin")

    if isinstance(origin, str):
        return canonical_piece_name(origin)

    if _looks_like_vec3(origin):
        return position_from_coord(origin)

    origin_key = sticker.get("origin_key")

    if isinstance(origin_key, tuple) and len(origin_key) >= 1:
        origin_pos = origin_key[0]

        if _looks_like_vec3(origin_pos):
            return position_from_coord(origin_pos)

    return None


def infer_original_face(sticker: dict) -> Face:
    """
    Infers the original face of a sticker.

    Preferred prototype format:
        sticker["origin_key"] = (original_position, original_normal)

    Also supported:
        sticker["origin_normal"] = original normal
        sticker["home_normal"] = original normal
        sticker["face"] = original face letter
        sticker["color"] = original face letter, if color is U/D/F/B/R/L
    """

    for key in ("origin_face", "home_face", "face"):
        value = sticker.get(key)

        if isinstance(value, str) and value in FACE_TO_NORMAL:
            return value

    for key in ("origin_normal", "home_normal"):
        value = sticker.get(key)

        if _looks_like_vec3(value):
            return face_from_normal(value)

    origin_key = sticker.get("origin_key")

    if isinstance(origin_key, tuple) and len(origin_key) >= 2:
        origin_normal = origin_key[1]

        if _looks_like_vec3(origin_normal):
            return face_from_normal(origin_normal)

    color = sticker.get("color")

    if isinstance(color, str) and color in FACE_TO_NORMAL:
        return color

    raise ValueError(
        "Cannot infer original face from sticker. "
        "Expected origin_key, origin_normal, home_normal, face, or face-letter color."
    )


def canonical_piece_name(value: str) -> Optional[PieceName]:
    """
    Converts a face-label string to CubeLab's canonical piece name.

    Example:
        URF -> UFR
        RFU -> UFR
    """

    letters = tuple(value.strip().upper())

    if len(letters) == 3:
        return _find_position_by_face_set(CORNER_FRAMES, letters)

    if len(letters) == 2:
        return _find_position_by_face_set(EDGE_FRAMES, letters)

    return None


def _find_position_by_face_set(
    frames: Dict[PieceName, Tuple[Face, ...]],
    faces: Iterable[Face],
) -> Optional[PieceName]:
    face_set = set(faces)

    for position, frame in frames.items():
        if set(frame) == face_set:
            return position

    return None


def _as_vec3(value: Sequence[int]) -> Vec3:
    if not _looks_like_vec3(value):
        raise ValueError(f"Expected Vec3, got {value}")

    return (int(value[0]), int(value[1]), int(value[2]))


def _looks_like_vec3(value: object) -> bool:
    if not isinstance(value, (tuple, list)):
        return False

    if len(value) != 3:
        return False

    return all(isinstance(v, int) for v in value)


__all__ = [
    "Vec3",
    "Face",
    "PieceName",
    "FACE_TO_NORMAL",
    "NORMAL_TO_FACE",
    "CORNER_FRAMES",
    "EDGE_FRAMES",
    "CORNER_ORDER",
    "EDGE_ORDER",
    "PIECE_ORDER",
    "PieceState",
    "analyze_cube",
    "analyze_stickers",
    "analyze_piece",
    "orientation_checksums",
    "state_by_piece",
    "format_piece_state",
    "position_from_coord",
    "coord_from_position",
    "faces_from_coord",
    "face_from_normal",
    "normal_from_face",
]
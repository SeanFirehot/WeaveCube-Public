"""Dual-table column traces, active skeletons, gaps, and twist episodes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .model import CORNER_NAMES, EDGE_NAMES, PIECE_NAMES
from .moves import parse_word
from .state import PDCCState


@dataclass(frozen=True, slots=True)
class PieceCell:
    piece: str
    active: bool
    before_pose: int
    after_pose: int
    before_orientation: int
    after_orientation: int


@dataclass(frozen=True, slots=True)
class ColumnRecord:
    index: int
    move: str
    before: PDCCState
    after: PDCCState
    active_corners: tuple[str, ...]
    active_edges: tuple[str, ...]

    def cell(self, piece: str) -> PieceCell:
        before_pose = self.before.pose_id(piece)
        after_pose = self.after.pose_id(piece)
        return PieceCell(
            piece=piece,
            active=before_pose != after_pose,
            before_pose=before_pose,
            after_pose=after_pose,
            before_orientation=self.before.orientation(piece),
            after_orientation=self.after.orientation(piece),
        )


@dataclass(frozen=True, slots=True)
class TwistEpisode:
    piece: str
    start_state_index: int
    end_state_index: int
    orientations: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class GapAudit:
    piece: str
    start_column: int
    end_column: int
    entry_pose: int
    exit_pose: int
    pure_inactive: bool
    identity_closure: bool
    active_columns: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class ColumnTrace:
    initial: PDCCState
    columns: tuple[ColumnRecord, ...]
    final: PDCCState

    @property
    def word(self) -> tuple[str, ...]:
        return tuple(column.move for column in self.columns)

    @property
    def states(self) -> tuple[PDCCState, ...]:
        return (self.initial, *(column.after for column in self.columns))

    def active_skeleton(self, piece: str) -> tuple[tuple[int, str, int, int], ...]:
        return tuple(
            (column.index, column.move, column.before.pose_id(piece), column.after.pose_id(piece))
            for column in self.columns
            if column.before.pose_id(piece) != column.after.pose_id(piece)
        )

    def corner_table(self) -> tuple[tuple[PieceCell, ...], ...]:
        return tuple(
            tuple(column.cell(piece) for piece in CORNER_NAMES) for column in self.columns
        )

    def edge_table(self) -> tuple[tuple[PieceCell, ...], ...]:
        return tuple(
            tuple(column.cell(piece) for piece in EDGE_NAMES) for column in self.columns
        )

    def gap_audit(self, piece: str, start_column: int, end_column: int) -> GapAudit:
        if not 0 <= start_column <= end_column <= len(self.columns):
            raise ValueError("Gap bounds must satisfy 0 <= start <= end <= length")
        entry = self.states[start_column].pose_id(piece)
        exit_pose = self.states[end_column].pose_id(piece)
        active_columns = tuple(
            column.index
            for column in self.columns[start_column:end_column]
            if column.before.pose_id(piece) != column.after.pose_id(piece)
        )
        return GapAudit(
            piece=piece,
            start_column=start_column,
            end_column=end_column,
            entry_pose=entry,
            exit_pose=exit_pose,
            pure_inactive=not active_columns,
            identity_closure=entry == exit_pose,
            active_columns=active_columns,
        )

    def twist_episodes(self, piece: str) -> tuple[TwistEpisode, ...]:
        orientations = tuple(state.orientation(piece) for state in self.states)
        episodes: list[TwistEpisode] = []
        start: int | None = None
        for state_index, orientation in enumerate(orientations):
            if orientation != 0 and start is None:
                start = state_index
            if orientation == 0 and start is not None:
                episodes.append(
                    TwistEpisode(
                        piece=piece,
                        start_state_index=start,
                        end_state_index=state_index,
                        orientations=orientations[start : state_index + 1],
                    )
                )
                start = None
        if start is not None:
            episodes.append(
                TwistEpisode(
                    piece=piece,
                    start_state_index=start,
                    end_state_index=len(orientations) - 1,
                    orientations=orientations[start:],
                )
            )
        return tuple(episodes)

    def full_replay_matches(self) -> bool:
        state = self.initial
        for column in self.columns:
            if state != column.before:
                return False
            state = state.apply(column.move)
            if state != column.after:
                return False
        return state == self.final


def build_trace(
    initial: PDCCState, word: str | Iterable[str]
) -> ColumnTrace:
    state = initial
    columns: list[ColumnRecord] = []
    for index, move in enumerate(parse_word(word)):
        after, active_corners, active_edges = state.transition(move)
        columns.append(
            ColumnRecord(
                index=index,
                move=move,
                before=state,
                after=after,
                active_corners=active_corners,
                active_edges=active_edges,
            )
        )
        state = after
    return ColumnTrace(initial=initial, columns=tuple(columns), final=state)

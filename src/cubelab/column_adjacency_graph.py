from __future__ import annotations

"""Column adjacency and exact state-carrying linearization for v101.4.2."""

from collections import defaultdict
from dataclasses import asdict, dataclass
from typing import Sequence

from .move_demand_balance import (
    MoveDemandVector,
    PackedColumn,
    PrecedenceResult,
)
from .state_flow_conservation import (
    InterColumnStateFlowResult,
    evaluate_intercolumn_state_flow,
)


@dataclass(frozen=True, slots=True)
class AdjacencyEdge:
    before_column_id: str
    after_column_id: str
    compatible: bool
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "before_column_id": self.before_column_id,
            "after_column_id": self.after_column_id,
            "compatible": self.compatible,
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class ColumnAdjacencyGraphResult:
    status: str
    nodes: tuple[str, ...]
    compatible_edges: tuple[tuple[str, str], ...]
    rejected_edges: tuple[AdjacencyEdge, ...]
    start_supported_nodes: tuple[str, ...]
    end_supported_nodes: tuple[str, ...]
    zero_indegree_nodes: tuple[str, ...]
    zero_outdegree_nodes: tuple[str, ...]
    strongly_connected_components: tuple[tuple[str, ...], ...]
    necessary_conditions_satisfied: bool
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "nodes": list(self.nodes),
            "compatible_edges": [
                list(value) for value in self.compatible_edges
            ],
            "rejected_edges": [edge.row() for edge in self.rejected_edges],
            "start_supported_nodes": list(self.start_supported_nodes),
            "end_supported_nodes": list(self.end_supported_nodes),
            "zero_indegree_nodes": list(self.zero_indegree_nodes),
            "zero_outdegree_nodes": list(self.zero_outdegree_nodes),
            "strongly_connected_components": [
                list(value) for value in self.strongly_connected_components
            ],
            "necessary_conditions_satisfied": (
                self.necessary_conditions_satisfied
            ),
            "failure_reasons": list(self.failure_reasons),
        }


@dataclass(frozen=True, slots=True)
class LinearizationResult:
    status: str
    valid_orders: tuple[tuple[str, ...], ...]
    valid_words: tuple[tuple[str, ...], ...]
    flow_results: tuple[InterColumnStateFlowResult, ...]
    explored_nodes: int
    cap_hit: bool
    precedence_pruned: int
    adjacency_pruned: int
    state_flow_pruned: int
    failure_reasons: tuple[str, ...]

    def row(self) -> dict[str, object]:
        return {
            "status": self.status,
            "valid_orders": [list(value) for value in self.valid_orders],
            "valid_words": [list(value) for value in self.valid_words],
            "flow_results": [value.row() for value in self.flow_results],
            "explored_nodes": self.explored_nodes,
            "cap_hit": self.cap_hit,
            "precedence_pruned": self.precedence_pruned,
            "adjacency_pruned": self.adjacency_pruned,
            "state_flow_pruned": self.state_flow_pruned,
            "failure_reasons": list(self.failure_reasons),
        }


def _events_by_piece(column: PackedColumn):
    return {event.piece_id: event for event in column.all_events}


def evaluate_direct_adjacency(
    before: PackedColumn,
    after: PackedColumn,
) -> AdjacencyEdge:
    """Evaluate pairwise necessary conditions for directly adjacent columns."""

    failures = []
    left = _events_by_piece(before)
    right = _events_by_piece(after)
    for piece in sorted(set(left) & set(right)):
        first = left[piece]
        second = right[piece]
        if first.state_after != second.state_before:
            failures.append(f"{piece}:STATE_FLOW_DISCONNECT")
        if (
            first.candidate_id == second.candidate_id
            and first.active_ordinal >= second.active_ordinal
        ):
            failures.append(f"{piece}:CANDIDATE_ORDINAL")
    return AdjacencyEdge(
        before_column_id=before.column_id,
        after_column_id=after.column_id,
        compatible=not failures,
        failure_reasons=tuple(failures),
    )


def _strong_components(
    nodes: Sequence[str],
    edges: Sequence[tuple[str, str]],
) -> tuple[tuple[str, ...], ...]:
    outgoing = defaultdict(list)
    for left, right in edges:
        outgoing[left].append(right)
    index = 0
    indices: dict[str, int] = {}
    low: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    components: list[tuple[str, ...]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        low[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)
        for target in outgoing[node]:
            if target not in indices:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in on_stack:
                low[node] = min(low[node], indices[target])
        if low[node] != indices[node]:
            return
        component = []
        while stack:
            value = stack.pop()
            on_stack.remove(value)
            component.append(value)
            if value == node:
                break
        components.append(tuple(sorted(component)))

    for node in nodes:
        if node not in indices:
            visit(node)
    return tuple(sorted(components))


def build_column_adjacency_graph(
    candidates: Sequence[MoveDemandVector],
    columns: Sequence[PackedColumn],
) -> ColumnAdjacencyGraphResult:
    nodes = tuple(column.column_id for column in columns)
    edges = []
    rejected = []
    for before in columns:
        for after in columns:
            if before.column_id == after.column_id:
                continue
            result = evaluate_direct_adjacency(before, after)
            if result.compatible:
                edges.append((before.column_id, after.column_id))
            else:
                rejected.append(result)

    by_piece = {candidate.piece_id: candidate for candidate in candidates}
    starts = []
    ends = []
    for column in columns:
        events = _events_by_piece(column)
        if all(
            by_piece[piece].initial_state == event.state_before
            for piece, event in events.items()
        ):
            starts.append(column.column_id)
        if all(
            by_piece[piece].final_state == event.state_after
            for piece, event in events.items()
        ):
            ends.append(column.column_id)

    indegree = {node: 0 for node in nodes}
    outdegree = {node: 0 for node in nodes}
    for left, right in edges:
        outdegree[left] += 1
        indegree[right] += 1
    failures = []
    if nodes and not starts:
        failures.append("NO_INITIAL_STATE_SUPPORTED_COLUMN")
    if nodes and not ends:
        failures.append("NO_FINAL_STATE_SUPPORTED_COLUMN")
    if len(nodes) > 1:
        for node in nodes:
            if node not in starts and indegree[node] == 0:
                failures.append(f"NO_PREDECESSOR_SUPPORT:{node}")
            if node not in ends and outdegree[node] == 0:
                failures.append(f"NO_SUCCESSOR_SUPPORT:{node}")
    return ColumnAdjacencyGraphResult(
        status=(
            "SAT_ADJACENCY_NECESSARY"
            if not failures
            else "UNSAT_ADJACENCY_NECESSARY"
        ),
        nodes=nodes,
        compatible_edges=tuple(sorted(edges)),
        rejected_edges=tuple(rejected),
        start_supported_nodes=tuple(sorted(starts)),
        end_supported_nodes=tuple(sorted(ends)),
        zero_indegree_nodes=tuple(
            sorted(node for node in nodes if indegree[node] == 0)
        ),
        zero_outdegree_nodes=tuple(
            sorted(node for node in nodes if outdegree[node] == 0)
        ),
        strongly_connected_components=_strong_components(nodes, edges),
        necessary_conditions_satisfied=not failures,
        failure_reasons=tuple(failures),
    )


def _empty_precedence(columns: Sequence[PackedColumn]) -> PrecedenceResult:
    return PrecedenceResult(
        status="SAT_ACYCLIC",
        edges=(),
        topological_order=tuple(column.column_id for column in columns),
        cycle_nodes=(),
        same_column_ordinal_collapses=(),
    )


def linearize_columns_with_state_flow(
    candidates: Sequence[MoveDemandVector],
    columns: Sequence[PackedColumn],
    precedence: PrecedenceResult | None,
    *,
    node_cap: int = 100_000,
    solution_limit: int = 10_000,
    require_adjacency_graph: bool = True,
) -> LinearizationResult:
    """Enumerate topological column orders and validate full 20-piece flow."""

    if node_cap < 1 or solution_limit < 1:
        raise ValueError("node_cap and solution_limit must be positive")
    precedence = precedence or _empty_precedence(columns)
    if precedence.status != "SAT_ACYCLIC":
        return LinearizationResult(
            status="UNSAT_PRECEDENCE",
            valid_orders=(),
            valid_words=(),
            flow_results=(),
            explored_nodes=0,
            cap_hit=False,
            precedence_pruned=0,
            adjacency_pruned=0,
            state_flow_pruned=0,
            failure_reasons=(precedence.status,),
        )

    adjacency = build_column_adjacency_graph(candidates, columns)
    if require_adjacency_graph and not adjacency.necessary_conditions_satisfied:
        return LinearizationResult(
            status="UNSAT_ADJACENCY",
            valid_orders=(),
            valid_words=(),
            flow_results=(),
            explored_nodes=0,
            cap_hit=False,
            precedence_pruned=0,
            adjacency_pruned=1,
            state_flow_pruned=0,
            failure_reasons=adjacency.failure_reasons,
        )

    by_id = {column.column_id: column for column in columns}
    predecessors = {column_id: set() for column_id in by_id}
    for left, right in precedence.edges:
        predecessors[right].add(left)
    compatible = set(adjacency.compatible_edges)
    explored = 0
    cap_hit = False
    precedence_pruned = 0
    adjacency_pruned = 0
    state_pruned = 0
    valid_orders = []
    valid_words = []
    flows = []

    def visit(prefix: tuple[str, ...], unused: frozenset[str]) -> None:
        nonlocal explored, cap_hit
        nonlocal precedence_pruned, adjacency_pruned, state_pruned
        if cap_hit or len(valid_orders) >= solution_limit:
            return
        explored += 1
        if explored > node_cap:
            cap_hit = True
            return
        if not unused:
            flow = evaluate_intercolumn_state_flow(
                candidates, columns, prefix
            )
            if flow.valid:
                valid_orders.append(prefix)
                valid_words.append(
                    tuple(by_id[column_id].move for column_id in prefix)
                )
                flows.append(flow)
            else:
                state_pruned += 1
            return
        for column_id in sorted(unused):
            if not predecessors[column_id].issubset(prefix):
                precedence_pruned += 1
                continue
            if (
                require_adjacency_graph
                and prefix
                and (prefix[-1], column_id) not in compatible
            ):
                adjacency_pruned += 1
                continue
            # A prefix can be invalid even though every adjacent pair is
            # supported.  Exact state-vector flow decides that boundary.
            partial_flow = evaluate_intercolumn_state_flow(
                candidates,
                tuple(by_id[value] for value in prefix + (column_id,)),
                prefix + (column_id,),
            )
            # The partial helper expects its candidate final boundary, so only
            # use its event-level mismatches as a sound early test.
            hard_mismatch = any(
                reason.endswith(
                    (
                        ":ACTIVE_IDENTITY_MISMATCH",
                        ":MOVE_MISMATCH",
                        ":STATE_SUPPLY_MISMATCH",
                        ":STATE_DEMAND_MISMATCH",
                    )
                )
                for reason in partial_flow.failure_reasons
            )
            if hard_mismatch:
                state_pruned += 1
                continue
            visit(prefix + (column_id,), unused - {column_id})
            if cap_hit or len(valid_orders) >= solution_limit:
                break

    visit((), frozenset(by_id))
    status = (
        "SAT_LINEARIZED"
        if valid_orders
        else "UNKNOWN_LINEARIZATION_CAP"
        if cap_hit
        else "UNSAT_LINEARIZATION"
    )
    return LinearizationResult(
        status=status,
        valid_orders=tuple(valid_orders),
        valid_words=tuple(valid_words),
        flow_results=tuple(flows),
        explored_nodes=explored,
        cap_hit=cap_hit,
        precedence_pruned=precedence_pruned,
        adjacency_pruned=adjacency_pruned,
        state_flow_pruned=state_pruned,
        failure_reasons=(
            ("NODE_CAP",)
            if cap_hit
            else ()
            if valid_orders
            else ("NO_FULL_STATE_FLOW_ORDER",)
        ),
    )


__all__ = [
    "AdjacencyEdge",
    "ColumnAdjacencyGraphResult",
    "LinearizationResult",
    "evaluate_direct_adjacency",
    "build_column_adjacency_graph",
    "linearize_columns_with_state_flow",
]

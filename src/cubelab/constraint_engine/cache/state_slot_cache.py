from __future__ import annotations

"""Sound incremental front-end for the v101.4.2 state-slot solver."""

from collections import Counter
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from time import perf_counter_ns
from typing import Mapping

from cubelab.move_demand_balance import ColumnInventory, MoveDemandVector
from cubelab.state_slot_propagation import (
    StateSlotKey,
    StateSlotPropagationResult,
    inventory_state_slot_target,
    propagate_state_slot_requirements,
)

from ..domains import PropagationState
from ..subproblem import (
    ExactSuffixMemoStore,
    JointSupportMemoResult,
    JointSupportSolveMetrics,
    propagate_state_slot_requirements_memoized,
)
from .cache_delta import CacheDelta
from .joint_support_cache import JointSupportCache, JointSupportKey
from .slot_contribution_index import CandidateSlotContributionIndex


def _digest(payload: object) -> str:
    return sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SlotSupportEntry:
    slot_key: StateSlotKey
    active_candidate_ids: tuple[str, ...]
    selected_count: int
    reachable_min: int
    reachable_max: int
    reachable_future_counts: tuple[int, ...]
    exact_target: int | None
    status: str
    version: int
    dirty: bool

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["slot_key"] = asdict(self.slot_key)
        payload["active_candidate_ids"] = list(
            self.active_candidate_ids
        )
        payload["reachable_future_counts"] = list(
            self.reachable_future_counts
        )
        return payload


@dataclass(frozen=True, slots=True)
class StateSlotProfile:
    call_id: str
    propagation_round: int
    candidate_scan_ns: int
    contribution_update_ns: int
    min_max_ns: int
    joint_support_ns: int
    requirement_generation_ns: int
    candidate_filtering_ns: int
    provenance_ns: int
    deduplication_ns: int
    cache_serialization_ns: int
    total_ns: int
    affected_moves: tuple[str, ...]
    affected_slots: int
    domain_changes_since_last_call: int
    full_recompute: bool
    joint_cache_hit: bool
    execution_path: str
    subproblem_memo_hits: int = 0
    subproblem_memo_misses: int = 0
    subproblem_expanded_nodes: int = 0
    subproblem_reused_nodes: int = 0

    def row(self) -> dict[str, object]:
        payload = asdict(self)
        payload["affected_moves"] = list(self.affected_moves)
        return payload


@dataclass(frozen=True, slots=True)
class StateSlotCacheEvaluation:
    result: StateSlotPropagationResult
    profile: StateSlotProfile
    dirty_slots: tuple[StateSlotKey, ...]
    dirty_move_families: tuple[tuple[str, str], ...]
    delta_rows: tuple[CacheDelta, ...]


class StateSlotSupportCache:
    """Incremental scalar support plus exact full-result memo.

    Scalar entries are updated only for conservatively dirty piece
    footprints.  The hard pruning result is either an immutable exact solver
    result under an identical signature or a fresh call to the reference full
    recomputation.  Consequently a stale scalar entry can never prune.
    """

    def __init__(
        self,
        candidates: tuple[MoveDemandVector, ...],
        *,
        maximum_joint_entries: int = 256,
        maximum_subproblem_entries: int = 100_000,
        dirty_updates_enabled: bool = True,
        joint_cache_enabled: bool = True,
        subproblem_memo_enabled: bool = False,
        subproblem_ordering_policy: str = "smallest_domain",
        fine_grained_dirty_enabled: bool = False,
    ) -> None:
        self._known_candidates = {
            candidate.candidate_id: candidate for candidate in candidates
        }
        self.index = CandidateSlotContributionIndex.build(candidates)
        self.entries: dict[StateSlotKey, SlotSupportEntry] = {}
        self.candidate_active_flags: dict[str, bool] = {
            candidate_id: True
            for candidate_id in self.index.candidate_counts
        }
        self.dirty_slots: set[StateSlotKey] = set(
            self.index.all_slots()
        )
        self.dirty_move_families: set[tuple[str, str]] = {
            (key.move, key.piece_type) for key in self.dirty_slots
        }
        self.global_version = 0
        self._active_domains: dict[str, tuple[str, ...]] = {}
        self._selected: dict[str, str] = {}
        self._inventory_signature: str | None = None
        self._hot_key: JointSupportKey | None = None
        self._hot_result: StateSlotPropagationResult | None = None
        self.joint_cache: JointSupportCache[
            StateSlotPropagationResult
        ] = JointSupportCache(maximum_entries=maximum_joint_entries)
        self.subproblem_memo: ExactSuffixMemoStore[
            JointSupportMemoResult
        ] = ExactSuffixMemoStore(
            maximum_entries=maximum_subproblem_entries
        )
        self.dirty_updates_enabled = dirty_updates_enabled
        self.joint_cache_enabled = joint_cache_enabled
        self.subproblem_memo_enabled = subproblem_memo_enabled
        self.subproblem_ordering_policy = subproblem_ordering_policy
        self.fine_grained_dirty_enabled = fine_grained_dirty_enabled
        self.last_subproblem_metrics: JointSupportSolveMetrics | None = None
        self.last_delta_rows: tuple[CacheDelta, ...] = ()
        self.profiles: list[StateSlotProfile] = []
        self.deltas: list[CacheDelta] = []
        self.full_recompute_calls = 0
        self.incremental_update_calls = 0
        self.zero_dirty_calls = 0
        self.total_dirty_slots = 0
        self.total_dirty_move_families = 0

    @classmethod
    def from_state(
        cls,
        state: PropagationState,
        *,
        maximum_joint_entries: int = 256,
        maximum_subproblem_entries: int = 100_000,
        dirty_updates_enabled: bool = True,
        joint_cache_enabled: bool = True,
        subproblem_memo_enabled: bool = False,
        subproblem_ordering_policy: str = "smallest_domain",
        fine_grained_dirty_enabled: bool = False,
    ) -> "StateSlotSupportCache":
        candidates: dict[str, MoveDemandVector] = {}
        for domain in state.domains.values():
            candidates.update(domain.candidates)
        candidates.update(
            (
                candidate.candidate_id,
                candidate,
            )
            for candidate in state.selected_candidates.values()
        )
        return cls(
            tuple(candidates.values()),
            maximum_joint_entries=maximum_joint_entries,
            maximum_subproblem_entries=maximum_subproblem_entries,
            dirty_updates_enabled=dirty_updates_enabled,
            joint_cache_enabled=joint_cache_enabled,
            subproblem_memo_enabled=subproblem_memo_enabled,
            subproblem_ordering_policy=subproblem_ordering_policy,
            fine_grained_dirty_enabled=fine_grained_dirty_enabled,
        )

    def ensure_candidates(self, state: PropagationState) -> None:
        changed = False
        for domain in state.domains.values():
            for candidate_id, candidate in domain.candidates.items():
                if candidate_id not in self._known_candidates:
                    self._known_candidates[candidate_id] = candidate
                    changed = True
        for candidate_id, candidate in (
            (value.candidate_id, value)
            for value in state.selected_candidates.values()
        ):
            if candidate_id not in self._known_candidates:
                self._known_candidates[candidate_id] = candidate
                changed = True
        if not changed:
            return
        self.index = CandidateSlotContributionIndex.build(
            self._known_candidates.values()
        )
        self.dirty_slots.update(self.index.all_slots())
        self.dirty_move_families.update(
            (key.move, key.piece_type)
            for key in self.dirty_slots
        )

    def candidate_affects_slots(self, candidate_id: str) -> bool:
        piece = self.index.candidate_to_piece.get(candidate_id)
        return bool(piece and self.index.piece_to_slots.get(piece))

    @staticmethod
    def _inventory_payload(
        inventory: ColumnInventory | None,
    ) -> object:
        return None if inventory is None else inventory.row()

    def _signature_key(
        self,
        *,
        inventory: ColumnInventory,
        node_cap: int,
        compute_candidate_support: bool,
    ) -> JointSupportKey:
        return JointSupportKey(
            inventory_signature=_digest(
                self._inventory_payload(inventory)
            ),
            active_domain_signature=_digest(self._active_domains),
            selected_signature=_digest(self._selected),
            node_cap=node_cap,
            compute_candidate_support=compute_candidate_support,
        )

    def _recompute_entry(
        self,
        key: StateSlotKey,
        state: PropagationState,
        target: Mapping[StateSlotKey, int],
    ) -> tuple[SlotSupportEntry, int, int]:
        contribution_started = perf_counter_ns()
        selected_count = sum(
            self.index.contribution(candidate.candidate_id, key)
            for candidate in state.selected_candidates.values()
        )
        per_piece_values: list[set[int]] = []
        active_candidate_ids: list[str] = []
        for piece, domain in sorted(state.domains.items()):
            if piece in state.selected_candidates:
                continue
            values = set()
            for candidate_id in domain.candidates:
                active_candidate_ids.append(candidate_id)
                values.add(self.index.contribution(candidate_id, key))
            per_piece_values.append(values or {0})
        contribution_ns = perf_counter_ns() - contribution_started

        bounds_started = perf_counter_ns()
        minimum = sum(min(values) for values in per_piece_values)
        maximum = sum(max(values) for values in per_piece_values)
        exact_target = int(target.get(key, 0))
        future_limit = max(0, exact_target - selected_count)
        reachable = {0}
        for values in per_piece_values:
            reachable = {
                left + right
                for left in reachable
                for right in values
                if left + right <= future_limit
            }
        needed = exact_target - selected_count
        if selected_count > exact_target:
            status = "CURRENT_EXCEEDS_TARGET"
        elif needed < minimum or needed > maximum:
            status = "SCALAR_BOUND_UNREACHABLE"
        elif needed not in reachable:
            status = "SCALAR_RESIDUE_UNREACHABLE"
        else:
            status = "FEASIBLE"
        bounds_ns = perf_counter_ns() - bounds_started
        return (
            SlotSupportEntry(
                slot_key=key,
                active_candidate_ids=tuple(sorted(set(active_candidate_ids))),
                selected_count=selected_count,
                reachable_min=minimum,
                reachable_max=maximum,
                reachable_future_counts=tuple(sorted(reachable)),
                exact_target=exact_target,
                status=status,
                version=self.global_version,
                dirty=False,
            ),
            contribution_ns,
            bounds_ns,
        )

    def _synchronize(
        self,
        state: PropagationState,
        inventory: ColumnInventory,
    ) -> tuple[
        tuple[StateSlotKey, ...],
        tuple[tuple[str, str], ...],
        tuple[CacheDelta, ...],
        int,
        int,
        int,
        int,
    ]:
        scan_started = perf_counter_ns()
        self.ensure_candidates(state)
        active_domains = {
            piece: tuple(sorted(domain.candidates))
            for piece, domain in sorted(state.domains.items())
        }
        selected = {
            piece: candidate.candidate_id
            for piece, candidate in sorted(
                state.selected_candidates.items()
            )
        }
        inventory_signature = _digest(inventory.row())
        changed_pieces = tuple(
            piece
            for piece in sorted(
                set(self._active_domains) | set(active_domains)
            )
            if self._active_domains.get(piece)
            != active_domains.get(piece)
            or self._selected.get(piece) != selected.get(piece)
        )
        domain_changes = sum(
            len(
                set(self._active_domains.get(piece, ()))
                ^ set(active_domains.get(piece, ()))
            )
            for piece in changed_pieces
        )
        inventory_changed = (
            self._inventory_signature is not None
            and self._inventory_signature != inventory_signature
        )
        candidate_scan_ns = perf_counter_ns() - scan_started

        dirty: set[StateSlotKey] = set(self.dirty_slots)
        deltas: list[CacheDelta] = []
        previous_version = self.global_version
        if changed_pieces or inventory_changed or not self.entries:
            self.global_version += 1
        for piece in changed_pieces:
            candidate_footprint = set(
                self.index.piece_to_slots.get(piece, ())
            )
            before_ids = set(self._active_domains.get(piece, ()))
            after_ids = set(active_domains.get(piece, ()))
            before_selected = self._selected.get(piece)
            after_selected = selected.get(piece)

            def piece_slot_signature(
                key: StateSlotKey,
                *,
                candidate_ids: set[str],
                selected_id: str | None,
            ) -> tuple[object, ...]:
                if selected_id is not None:
                    return (
                        "SELECTED",
                        self.index.contribution(selected_id, key),
                    )
                return (
                    "DOMAIN",
                    tuple(
                        sorted(
                            {
                                self.index.contribution(
                                    candidate_id,
                                    key,
                                )
                                for candidate_id in candidate_ids
                            }
                        )
                    ),
                )

            footprint = (
                {
                    key
                    for key in candidate_footprint
                    if piece_slot_signature(
                        key,
                        candidate_ids=before_ids,
                        selected_id=before_selected,
                    )
                    != piece_slot_signature(
                        key,
                        candidate_ids=after_ids,
                        selected_id=after_selected,
                    )
                }
                if self.fine_grained_dirty_enabled
                else candidate_footprint
            )
            dirty.update(footprint)
            changed_ids = sorted(before_ids ^ after_ids)
            deltas.append(
                CacheDelta(
                    event_type=(
                        "CANDIDATE_DOMAIN_CHANGED"
                        if self._selected.get(piece) == selected.get(piece)
                        else "CANDIDATE_SELECTION_CHANGED"
                    ),
                    candidate_id=(
                        changed_ids[0] if len(changed_ids) == 1 else None
                    ),
                    piece_id=piece,
                    dirty_slots=tuple(sorted(footprint)),
                    dirty_move_families=tuple(
                        sorted(
                            {
                                (key.move, key.piece_type)
                                for key in footprint
                            }
                        )
                    ),
                    previous_version=previous_version,
                    next_version=self.global_version,
                )
            )
        target = inventory_state_slot_target(inventory).counts
        if not self.dirty_updates_enabled:
            dirty.update(self.index.all_slots())
            dirty.update(target)
        if inventory_changed or not self.entries:
            inventory_slots = set(target)
            dirty.update(inventory_slots)
            if inventory_changed:
                deltas.append(
                    CacheDelta(
                        event_type="INVENTORY_CHANGED",
                        candidate_id=None,
                        piece_id=None,
                        dirty_slots=tuple(sorted(inventory_slots)),
                        dirty_move_families=tuple(
                            sorted(
                                {
                                    (key.move, key.piece_type)
                                    for key in inventory_slots
                                }
                            )
                        ),
                        previous_version=previous_version,
                        next_version=self.global_version,
                    )
                )

        contribution_ns = 0
        min_max_ns = 0
        for key in sorted(dirty):
            entry, contribution_part, bounds_part = self._recompute_entry(
                key,
                state,
                target,
            )
            self.entries[key] = entry
            contribution_ns += contribution_part
            min_max_ns += bounds_part

        self._active_domains = active_domains
        self._selected = selected
        self._inventory_signature = inventory_signature
        self.candidate_active_flags = {
            candidate_id: any(
                candidate_id in values
                for values in active_domains.values()
            )
            for candidate_id in self.index.candidate_counts
        }
        self.dirty_slots.clear()
        move_families = tuple(
            sorted({(key.move, key.piece_type) for key in dirty})
        )
        self.dirty_move_families.clear()
        self.deltas.extend(deltas)
        if dirty:
            self.incremental_update_calls += 1
        else:
            self.zero_dirty_calls += 1
        self.total_dirty_slots += len(dirty)
        self.total_dirty_move_families += len(move_families)
        return (
            tuple(sorted(dirty)),
            move_families,
            tuple(deltas),
            domain_changes,
            candidate_scan_ns,
            contribution_ns,
            min_max_ns,
        )

    def evaluate(
        self,
        state: PropagationState,
        *,
        inventory: ColumnInventory,
        node_cap: int,
        compute_candidate_support: bool = True,
        propagation_round: int = 0,
    ) -> StateSlotCacheEvaluation:
        total_started = perf_counter_ns()
        (
            dirty_slots,
            dirty_moves,
            deltas,
            domain_changes,
            candidate_scan_ns,
            contribution_ns,
            min_max_ns,
        ) = self._synchronize(state, inventory)
        key = self._signature_key(
            inventory=inventory,
            node_cap=node_cap,
            compute_candidate_support=compute_candidate_support,
        )
        joint_started = perf_counter_ns()
        result = (
            self._hot_result
            if self.joint_cache_enabled and key == self._hot_key
            else None
        )
        cache_hit = result is not None
        if result is None and self.joint_cache_enabled:
            result = self.joint_cache.get(key)
            cache_hit = result is not None
        if result is None:
            if self.subproblem_memo_enabled:
                contract_signature = _digest(
                    {
                        candidate_id: getattr(contract, "row", lambda: str(contract))()
                        for candidate_id, contract in sorted(
                            state.contracts.items()
                        )
                    }
                )
                result, self.last_subproblem_metrics = (
                    propagate_state_slot_requirements_memoized(
                        selected=state.selected(),
                        remaining_domains=state.unselected_domains(),
                        inventory=inventory,
                        node_cap=node_cap,
                        compute_candidate_support=compute_candidate_support,
                        memo_store=self.subproblem_memo,
                        ordering_policy=self.subproblem_ordering_policy,
                        contract_signature=contract_signature,
                        extension_signature=contract_signature,
                    )
                )
            else:
                self.last_subproblem_metrics = None
                result = propagate_state_slot_requirements(
                    selected=state.selected(),
                    remaining_domains=state.unselected_domains(),
                    inventory=inventory,
                    node_cap=node_cap,
                    compute_candidate_support=compute_candidate_support,
                )
            self.full_recompute_calls += 1
            if self.joint_cache_enabled:
                self.joint_cache.put(key, result)
        self._hot_key = key if self.joint_cache_enabled else None
        self._hot_result = result if self.joint_cache_enabled else None
        joint_ns = perf_counter_ns() - joint_started

        serialization_started = perf_counter_ns()
        self.logical_hash()
        serialization_ns = perf_counter_ns() - serialization_started
        profile = StateSlotProfile(
            call_id=f"SS-{len(self.profiles) + 1:08d}",
            propagation_round=propagation_round,
            candidate_scan_ns=candidate_scan_ns,
            contribution_update_ns=contribution_ns,
            min_max_ns=min_max_ns,
            joint_support_ns=joint_ns,
            requirement_generation_ns=0,
            candidate_filtering_ns=0,
            provenance_ns=0,
            deduplication_ns=0,
            cache_serialization_ns=serialization_ns,
            total_ns=perf_counter_ns() - total_started,
            affected_moves=tuple(
                sorted({move for move, _ in dirty_moves})
            ),
            affected_slots=len(dirty_slots),
            domain_changes_since_last_call=domain_changes,
            full_recompute=not cache_hit,
            joint_cache_hit=cache_hit,
            execution_path="INVENTORY_FIXED_PATH",
            subproblem_memo_hits=(
                0
                if self.last_subproblem_metrics is None
                else self.last_subproblem_metrics.memo_hits
            ),
            subproblem_memo_misses=(
                0
                if self.last_subproblem_metrics is None
                else self.last_subproblem_metrics.memo_misses
            ),
            subproblem_expanded_nodes=(
                0
                if self.last_subproblem_metrics is None
                else self.last_subproblem_metrics.expanded_nodes
            ),
            subproblem_reused_nodes=(
                0
                if self.last_subproblem_metrics is None
                else self.last_subproblem_metrics.reused_logical_nodes
            ),
        )
        self.profiles.append(profile)
        self.last_delta_rows = deltas
        return StateSlotCacheEvaluation(
            result=result,
            profile=profile,
            dirty_slots=dirty_slots,
            dirty_move_families=dirty_moves,
            delta_rows=deltas,
        )

    def replace_last_profile(
        self,
        **updates: int,
    ) -> StateSlotProfile:
        if not self.profiles:
            raise RuntimeError("no state-slot profile to update")
        profile = replace(self.profiles[-1], **updates)
        self.profiles[-1] = profile
        return profile

    def logical_snapshot(self) -> dict[str, object]:
        # SlotSupportEntry and StateSlotPropagationResult are immutable.
        # Shallow container copies therefore form a sound copy-on-write
        # snapshot and avoid copying the exact solver witness on every branch.
        return {
            "entries": dict(self.entries),
            "candidate_active_flags": dict(
                self.candidate_active_flags
            ),
            "dirty_slots": set(self.dirty_slots),
            "dirty_move_families": set(self.dirty_move_families),
            "global_version": self.global_version,
            "active_domains": dict(self._active_domains),
            "selected": dict(self._selected),
            "inventory_signature": self._inventory_signature,
            "hot_key": self._hot_key,
            "hot_result": self._hot_result,
        }

    def restore_logical_snapshot(
        self,
        snapshot: Mapping[str, object],
    ) -> None:
        self.entries = dict(snapshot["entries"])
        self.candidate_active_flags = dict(
            snapshot["candidate_active_flags"]
        )
        self.dirty_slots = set(snapshot["dirty_slots"])
        self.dirty_move_families = set(
            snapshot["dirty_move_families"]
        )
        self.global_version = int(snapshot["global_version"])
        self._active_domains = dict(snapshot["active_domains"])
        self._selected = dict(snapshot["selected"])
        self._inventory_signature = snapshot["inventory_signature"]
        self._hot_key = snapshot["hot_key"]
        self._hot_result = snapshot["hot_result"]

    def logical_hash(self) -> str:
        return _digest(
            {
                "entries": [
                    entry.row()
                    for _, entry in sorted(self.entries.items())
                ],
                "candidate_active_flags": dict(
                    sorted(self.candidate_active_flags.items())
                ),
                "dirty_slots": [
                    asdict(key) for key in sorted(self.dirty_slots)
                ],
                "dirty_move_families": sorted(
                    self.dirty_move_families
                ),
                "global_version": self.global_version,
                "active_domains": self._active_domains,
                "selected": self._selected,
                "inventory_signature": self._inventory_signature,
                "hot_key": (
                    None
                    if self._hot_key is None
                    else self._hot_key.row()
                ),
                "hot_result_status": (
                    None
                    if self._hot_result is None
                    else self._hot_result.status
                ),
                "dirty_updates_enabled": self.dirty_updates_enabled,
                "joint_cache_enabled": self.joint_cache_enabled,
                "subproblem_memo_enabled": self.subproblem_memo_enabled,
                "subproblem_ordering_policy": (
                    self.subproblem_ordering_policy
                ),
                "fine_grained_dirty_enabled": (
                    self.fine_grained_dirty_enabled
                ),
            }
        )

    def metrics_row(self) -> dict[str, object]:
        calls = len(self.profiles)
        return {
            "profile_calls": calls,
            "full_recompute_calls": self.full_recompute_calls,
            "incremental_update_calls": self.incremental_update_calls,
            "zero_dirty_calls": self.zero_dirty_calls,
            "average_dirty_slots": (
                self.total_dirty_slots / calls if calls else 0.0
            ),
            "average_dirty_move_families": (
                self.total_dirty_move_families / calls
                if calls
                else 0.0
            ),
            "joint_support_cache": self.joint_cache.metrics_row(),
            "subproblem_memo": self.subproblem_memo.metrics_row(),
            "subproblem_expanded_nodes": sum(
                profile.subproblem_expanded_nodes
                for profile in self.profiles
            ),
            "subproblem_reused_nodes": sum(
                profile.subproblem_reused_nodes
                for profile in self.profiles
            ),
            "dirty_updates_enabled": self.dirty_updates_enabled,
            "joint_cache_enabled": self.joint_cache_enabled,
            "subproblem_memo_enabled": self.subproblem_memo_enabled,
            "subproblem_ordering_policy": (
                self.subproblem_ordering_policy
            ),
            "fine_grained_dirty_enabled": (
                self.fine_grained_dirty_enabled
            ),
            "entry_count": len(self.entries),
            "global_version": self.global_version,
        }


__all__ = [
    "SlotSupportEntry",
    "StateSlotProfile",
    "StateSlotCacheEvaluation",
    "StateSlotSupportCache",
]

#!/usr/bin/env python3
"""Search exact CubeLab solutions and rank their phase-1 twist skeletons.

Normal form for the phase-1 prefix:
    K0 t1 K1 t2 ... tr Kr
where K blocks use only the DR-preserving move set
    K = <U,D,R2,L2,F2,B2>
and t events are quarter turns
    R,R',L,L',F,F',B,B'.

This searcher does NOT claim that fewer twist events always imply a shorter solve.
It searches exact phase-1 paths with admissible Kociemba-style pruning, completes
phase 2 exactly within the current total-length incumbent, then ranks valid full
solutions by total HTM first and twist-event count second.

Research status:
- SAT solutions are always full-replay verified.
- If time/node limits cut the search, lack of a better skeleton is UNKNOWN.
"""
from __future__ import annotations

import argparse
import itertools
import pickle
import time
from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from cubelab.pdcc import (
    CORNER_NAMES,
    EDGE_NAMES,
    MOVE_ORDER,
    MOVES,
    ORIENTATION_SYSTEM,
    PDCCEngine,
    inverse_move,
    inverse_word,
)
from cubelab.pdcc.model import PIECE_SPECS

K_MOVES = ("U", "U'", "U2", "D", "D'", "D2", "R2", "L2", "F2", "B2")
K_SET = set(K_MOVES)
T_MOVES = {"R", "R'", "L", "L'", "F", "F'", "B", "B'"}
E_SLICE = ("FR", "FL", "BL", "BR")
E_SLICE_SET = set(E_SLICE)
UD_EDGES = tuple(e for e in EDGE_NAMES if e not in E_SLICE_SET)
PIECES = CORNER_NAMES + EDGE_NAMES

CI = {s: i for i, s in enumerate(CORNER_NAMES)}
EI = {s: i for i, s in enumerate(EDGE_NAMES)}
UI = {s: i for i, s in enumerate(UD_EDGES)}
SI = {s: i for i, s in enumerate(E_SLICE)}
MI = {m: i for i, m in enumerate(MOVE_ORDER)}
KI = {m: i for i, m in enumerate(K_MOVES)}
NM = len(MOVE_ORDER)
NK = len(K_MOVES)
NCO = 3**7
NEO = 2**11
MASKS = tuple(m for m in range(1 << 12) if m.bit_count() == 4)
MR = {m: i for i, m in enumerate(MASKS)}
NS = len(MASKS)

engine = PDCCEngine()
SOLVED = engine.solved()

FO = {f: i for i, f in enumerate("URFDLB")}
OPP = {"U": "D", "D": "U", "R": "L", "L": "R", "F": "B", "B": "F"}


def allow(last_face: str | None, move: str) -> bool:
    face = move[0]
    if last_face == face:
        return False
    if last_face and OPP[face] == last_face and FO[face] < FO[last_face]:
        return False
    return True


def normalize(word: Sequence[str]) -> tuple[str, ...]:
    value = {"": 1, "2": 2, "'": 3}
    suffix = {1: "", 2: "2", 3: "'"}
    stack: list[tuple[str, int]] = []
    for move in word:
        face = move[0]
        amount = value[move[1:] if len(move) > 1 else ""]
        if stack and stack[-1][0] == face:
            old_face, old_amount = stack.pop()
            total = (old_amount + amount) % 4
            if total:
                stack.append((old_face, total))
        else:
            stack.append((face, amount))
    return tuple(face + suffix[amount] for face, amount in stack)


def slot_maps(move: str):
    after = SOLVED.apply(move)
    return (
        {s: after.slot(s) for s in CORNER_NAMES},
        {s: after.slot(s) for s in EDGE_NAMES},
    )


SLOT = {m: slot_maps(m) for m in MOVE_ORDER}


def orientation_delta(names, move: str, slot: str) -> int:
    spec = MOVES[move]
    if not spec.is_active_position(PIECE_SPECS[slot].solved_position):
        return 0
    return ORIENTATION_SYSTEM.cocycle(names[0], spec.rotation_id, slot)


CD = {m: tuple(orientation_delta(CORNER_NAMES, m, s) for s in CORNER_NAMES) for m in MOVE_ORDER}
ED = {m: tuple(orientation_delta(EDGE_NAMES, m, s) for s in EDGE_NAMES) for m in MOVE_ORDER}


def decode_co(x: int):
    values = []
    total = 0
    for _ in range(7):
        values.append(x % 3)
        total += values[-1]
        x //= 3
    values.append((-total) % 3)
    return values


def encode_co(values):
    x = 0
    place = 1
    for i in range(7):
        x += values[i] * place
        place *= 3
    return x


def decode_eo(x: int):
    values = []
    parity = 0
    for _ in range(11):
        values.append(x & 1)
        parity ^= values[-1]
        x >>= 1
    values.append(parity)
    return values


def encode_eo(values):
    x = 0
    for i in range(10, -1, -1):
        x = (x << 1) | values[i]
    return x


def build_phase1_tables():
    co_table = array('H', [0]) * (NCO * NM)
    eo_table = array('H', [0]) * (NEO * NM)
    slice_table = array('H', [0]) * (NS * NM)

    for x in range(NCO):
        values = decode_co(x)
        for mi, move in enumerate(MOVE_ORDER):
            corner_map, _ = SLOT[move]
            out = [0] * 8
            delta = CD[move]
            for src in CORNER_NAMES:
                ii = CI[src]
                jj = CI[corner_map[src]]
                out[jj] = (values[ii] + delta[ii]) % 3
            co_table[x * NM + mi] = encode_co(out)

    for x in range(NEO):
        values = decode_eo(x)
        for mi, move in enumerate(MOVE_ORDER):
            _, edge_map = SLOT[move]
            out = [0] * 12
            delta = ED[move]
            for src in EDGE_NAMES:
                ii = EI[src]
                jj = EI[edge_map[src]]
                out[jj] = (values[ii] + delta[ii]) % 2
            eo_table[x * NM + mi] = encode_eo(out)

    for si, mask in enumerate(MASKS):
        for mi, move in enumerate(MOVE_ORDER):
            _, edge_map = SLOT[move]
            new_mask = 0
            for src in EDGE_NAMES:
                ii = EI[src]
                if mask & (1 << ii):
                    new_mask |= 1 << EI[edge_map[src]]
            slice_table[si * NM + mi] = MR[new_mask]

    return co_table, eo_table, slice_table


def joint_pdb(n1, table1, slice_table):
    size = n1 * NS
    goal = SLICE_GOAL
    dist = array('B', [255]) * size
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        x = queue.popleft()
        nd = dist[x] + 1
        a = x // NS
        s = x % NS
        for mi in range(NM):
            y = table1[a * NM + mi] * NS + slice_table[s * NM + mi]
            if dist[y] == 255:
                dist[y] = nd
                queue.append(y)
    return dist


def pack_q(co: int, eo: int, sl: int) -> int:
    return (co * NEO + eo) * NS + sl


def unpack_q(key: int):
    sl = key % NS
    key //= NS
    eo = key % NEO
    co = key // NEO
    return co, eo, sl


def q_move(key: int, mi: int, co_table, eo_table, slice_table) -> int:
    co, eo, sl = unpack_q(key)
    return pack_q(
        co_table[co * NM + mi],
        eo_table[eo * NM + mi],
        slice_table[sl * NM + mi],
    )


def q_of(state) -> int:
    co = [0] * 8
    eo = [0] * 12
    mask = 0
    for piece in CORNER_NAMES:
        co[CI[state.slot(piece)]] = state.orientation(piece)
    for piece in EDGE_NAMES:
        slot = state.slot(piece)
        i = EI[slot]
        eo[i] = state.orientation(piece)
        if piece in E_SLICE_SET:
            mask |= 1 << i
    return pack_q(encode_co(co), encode_eo(eo), MR[mask])


SLICE_GOAL = MR[sum(1 << EI[e] for e in E_SLICE)]
GOAL_Q = q_of(SOLVED)


def phase1_lb(key, co_slice_pdb, eo_slice_pdb) -> int:
    co, eo, sl = unpack_q(key)
    return max(co_slice_pdb[co * NS + sl], eo_slice_pdb[eo * NS + sl])


# Phase 2 compact coordinates -------------------------------------------------
P8 = tuple(itertools.permutations(range(8)))
R8 = {p: i for i, p in enumerate(P8)}
NP8 = len(P8)
P4 = tuple(itertools.permutations(range(4)))
R4 = {p: i for i, p in enumerate(P4)}
NP4 = len(P4)


def build_phase2_tables():
    cp = array('H', [0]) * (NP8 * NK)
    up = array('H', [0]) * (NP8 * NK)
    sp = array('B', [0]) * (NP4 * NK)
    corner_targets = []
    ud_targets = []
    slice_targets = []
    for move in K_MOVES:
        corner_map, edge_map = SLOT[move]
        corner_targets.append(tuple(CI[corner_map[s]] for s in CORNER_NAMES))
        ud_targets.append(tuple(UI[edge_map[s]] for s in UD_EDGES))
        slice_targets.append(tuple(SI[edge_map[s]] for s in E_SLICE))

    for pi, perm in enumerate(P8):
        for mi, target in enumerate(corner_targets):
            out = [0] * 8
            for i in range(8):
                out[target[i]] = perm[i]
            cp[pi * NK + mi] = R8[tuple(out)]
        for mi, target in enumerate(ud_targets):
            out = [0] * 8
            for i in range(8):
                out[target[i]] = perm[i]
            up[pi * NK + mi] = R8[tuple(out)]

    for pi, perm in enumerate(P4):
        for mi, target in enumerate(slice_targets):
            out = [0] * 4
            for i in range(4):
                out[target[i]] = perm[i]
            sp[pi * NK + mi] = R4[tuple(out)]
    return cp, up, sp


def pack_p(c: int, u: int, s: int) -> int:
    return (c * NP8 + u) * NP4 + s


def unpack_p(key: int):
    s = key % NP4
    key //= NP4
    u = key % NP8
    c = key // NP8
    return c, u, s


def p_move(key: int, mi: int, cp, up, sp) -> int:
    c, u, s = unpack_p(key)
    return pack_p(cp[c * NK + mi], up[u * NK + mi], sp[s * NK + mi])


def p_of(state):
    cp = [0] * 8
    up = [0] * 8
    sp = [0] * 4
    for i, piece in enumerate(CORNER_NAMES):
        cp[CI[state.slot(piece)]] = i
    for i, piece in enumerate(UD_EDGES):
        slot = state.slot(piece)
        if slot not in UI:
            return None
        up[UI[slot]] = i
    for i, piece in enumerate(E_SLICE):
        slot = state.slot(piece)
        if slot not in SI:
            return None
        sp[SI[slot]] = i
    return pack_p(R8[tuple(cp)], R8[tuple(up)], R4[tuple(sp)])


GOAL_P = p_of(SOLVED)


def single_pdb(size, goal, table):
    dist = array('B', [255]) * size
    dist[goal] = 0
    queue = deque([goal])
    while queue:
        x = queue.popleft()
        nd = dist[x] + 1
        for mi in range(NK):
            y = table[x * NK + mi]
            if dist[y] == 255:
                dist[y] = nd
                queue.append(y)
    return dist


def phase2_lb(key, cpd, upd, spd) -> int:
    c, u, s = unpack_p(key)
    return max(cpd[c], upd[u], spd[s])


@dataclass
class BackP:
    dist: dict
    parent: dict
    depth: int


def build_pback(depth: int, cp, up, sp):
    dist = {GOAL_P: 0}
    parent = {GOAL_P: (None, None)}
    frontier = [GOAL_P]
    for d in range(1, depth + 1):
        new_frontier = []
        for key in frontier:
            for mi, move in enumerate(K_MOVES):
                y = p_move(key, mi, cp, up, sp)
                if y in dist:
                    continue
                dist[y] = d
                parent[y] = (key, inverse_move(move))
                new_frontier.append(y)
        frontier = new_frontier
    return BackP(dist, parent, depth)


def p_suffix(key, back: BackP):
    out = []
    while back.parent[key][0] is not None:
        next_key, move = back.parent[key]
        out.append(move)
        key = next_key
    return tuple(out)


def p_prefix(key, parent):
    out = []
    while parent[key][0] is not None:
        prev, move = parent[key]
        out.append(move)
        key = prev
    out.reverse()
    return tuple(out)


def shortest_b(key, back: BackP, max_depth: int, cp, up, sp):
    if key in back.dist:
        word = p_suffix(key, back)
        return (word, len(word)) if len(word) <= max_depth else (None, None)
    forward_depth = max_depth - back.depth
    if forward_depth < 0:
        return None, None
    parent = {key: (None, None)}
    frontier = [key]
    for _ in range(1, forward_depth + 1):
        new_frontier = []
        meets = []
        for x in frontier:
            for mi, move in enumerate(K_MOVES):
                y = p_move(x, mi, cp, up, sp)
                if y in parent:
                    continue
                parent[y] = (x, move)
                new_frontier.append(y)
                if y in back.dist:
                    meets.append(y)
        if meets:
            meet = min(meets, key=lambda y: (back.dist[y], y))
            word = normalize(p_prefix(meet, parent) + p_suffix(meet, back))
            return word, len(word)
        frontier = new_frontier
    return None, None


def is_dr(state) -> bool:
    return all(state.orientation(p) == 0 for p in PIECES) and all(state.slot(p) in E_SLICE_SET for p in E_SLICE)


def segment_phase1(word: Sequence[str]):
    out = []
    block = []
    for move in word:
        if move in K_SET:
            block.append(move)
            continue
        if move not in T_MOVES:
            raise ValueError(f"Unexpected move {move}")
        out.append(("K", tuple(block)))
        block = []
        out.append(("T", (move,)))
    out.append(("K", tuple(block)))
    return tuple(out)


def skeleton_metrics(phase1: Sequence[str]):
    parts = segment_phase1(phase1)
    twist_events = sum(1 for kind, _ in parts if kind == "T")
    k_blocks = [word for kind, word in parts if kind == "K"]
    return {
        "parts": parts,
        "twist_events": twist_events,
        "k_moves": sum(len(w) for w in k_blocks),
        "nonempty_k_blocks": sum(bool(w) for w in k_blocks),
        "max_k_block": max((len(w) for w in k_blocks), default=0),
    }


def earliest_dr_split(start, solution: Sequence[str]):
    trace = engine.trace(solution, initial=start)
    for i, state in enumerate(trace.states):
        if is_dr(state):
            return tuple(solution[:i]), tuple(solution[i:])
    raise RuntimeError("Solved endpoint should be DR")


def short_structure(state):
    sig = engine.signature(state)
    return (
        sig.twisted_corner_count,
        sig.flipped_edge_count,
        sig.native_boundary_count,
        sig.rigid_bundle_count,
        sig.moved_corner_count + sig.moved_edge_count,
    )


def block_effects(start, phase1):
    effects = []
    state = start
    for kind, word in segment_phase1(phase1):
        before = state
        for move in word:
            state = state.apply(move)
        if kind == "K" and word:
            a = short_structure(before)
            b = short_structure(state)
            effects.append({
                "word": word,
                "boundary_delta": b[2] - a[2],
                "bundle_delta": b[3] - a[3],
                "moved_delta": b[4] - a[4],
            })
    return effects


@dataclass
class Candidate:
    total: int
    phase1: tuple[str, ...]
    phase2: tuple[str, ...]
    solution: tuple[str, ...]
    twist_events: int
    k_moves: int
    nonempty_k_blocks: int
    max_k_block: int

    def rank(self):
        return (self.total, self.twist_events, len(self.phase1), len(self.phase2), self.solution)


def cache_load_or_build(path: Path):
    version = 1
    if path.exists():
        try:
            with path.open("rb") as fh:
                payload = pickle.load(fh)
            if payload.get("version") == version:
                return payload["tables"], True
        except Exception:
            pass

    co, eo, sl = build_phase1_tables()
    cos = joint_pdb(NCO, co, sl)
    eos = joint_pdb(NEO, eo, sl)
    cp, up, sp = build_phase2_tables()
    cpd = single_pdb(NP8, 0, cp)
    upd = single_pdb(NP8, 0, up)
    spd = single_pdb(NP4, 0, sp)
    tables = (co, eo, sl, cos, eos, cp, up, sp, cpd, upd, spd)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"version": version, "tables": tables}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return tables, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scramble", required=True)
    ap.add_argument("--phase1-slack", type=int, default=3,
                    help="Search phase-1 depths from PDB lower bound through lb+slack, capped by incumbent")
    ap.add_argument("--b-back-depth", type=int, default=6)
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--node-cap", type=int, default=5_000_000)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cache", default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl")
    args = ap.parse_args()

    print("loading/building pruning tables...", flush=True)
    t_tables = time.time()
    tables, loaded = cache_load_or_build(Path(args.cache))
    co, eo, sl, cos, eos, cp, up, sp, cpd, upd, spd = tables
    print(f"tables {'loaded from cache' if loaded else 'built and cached'} in {time.time()-t_tables:.2f}s", flush=True)

    back = build_pback(args.b_back_depth, cp, up, sp)
    start = engine.from_word(args.scramble)

    seed_solution = normalize(inverse_word(tuple(args.scramble.split())))
    if not start.apply_word(seed_solution).is_solved():
        raise RuntimeError("inverse scramble seed replay failed")
    seed_p1, seed_p2 = earliest_dr_split(start, seed_solution)
    seed_m = skeleton_metrics(seed_p1)
    incumbent = len(seed_solution)
    candidates: dict[tuple[str, ...], Candidate] = {}

    def record(phase1, phase2, solution):
        nonlocal incumbent
        metrics = skeleton_metrics(phase1)
        cand = Candidate(
            total=len(solution),
            phase1=tuple(phase1),
            phase2=tuple(phase2),
            solution=tuple(solution),
            twist_events=metrics["twist_events"],
            k_moves=metrics["k_moves"],
            nonempty_k_blocks=metrics["nonempty_k_blocks"],
            max_k_block=metrics["max_k_block"],
        )
        old = candidates.get(cand.solution)
        if old is None or cand.rank() < old.rank():
            candidates[cand.solution] = cand
        if cand.total < incumbent:
            incumbent = cand.total
            print(f"NEW TOTAL BEST {incumbent}: {' '.join(cand.solution)}", flush=True)

    record(seed_p1, seed_p2, seed_solution)
    print(
        f"SEED total={len(seed_solution)} phase1={len(seed_p1)} phase2={len(seed_p2)} "
        f"twist-events={seed_m['twist_events']} solution={' '.join(seed_solution)}",
        flush=True,
    )

    q0 = q_of(start)
    lb = phase1_lb(q0, cos, eos)
    max_depth = min(lb + args.phase1_slack, incumbent)
    print(f"phase1 PDB lower bound={lb}; search depths {lb}..{max_depth}", flush=True)

    t0 = time.time()
    nodes = 0
    goals = 0
    exact_b = 0
    cut = False
    b_cache = {}

    for depth in range(lb, max_depth + 1):
        if time.time() - t0 >= args.time_limit:
            cut = True
            break
        path: list[str] = []
        local_nodes = 0
        local_goals = 0

        def dfs(q, remaining, last_face=None):
            nonlocal nodes, goals, exact_b, cut, local_nodes, local_goals, incumbent
            if cut:
                return
            nodes += 1
            local_nodes += 1
            if nodes > args.node_cap or time.time() - t0 >= args.time_limit:
                cut = True
                return
            if phase1_lb(q, cos, eos) > remaining:
                return
            if q == GOAL_Q:
                if remaining != 0:
                    return
                goals += 1
                local_goals += 1
                phase1 = tuple(path)
                dr_state = start.apply_word(phase1)
                pk = p_of(dr_state)
                if pk is None:
                    raise RuntimeError("phase1 goal without valid phase2 coordinate")
                p2lower = phase2_lb(pk, cpd, upd, spd)
                # One cross-boundary same-face normalization can save at most one HTM.
                if depth + p2lower - 1 > incumbent:
                    return
                max_b = max(0, incumbent - depth + 1)
                cached = b_cache.get((pk, max_b))
                if cached is None:
                    bword, bdist = shortest_b(pk, back, max_b, cp, up, sp)
                    b_cache[(pk, max_b)] = (bword, bdist)
                    exact_b += 1
                else:
                    bword, bdist = cached
                if bword is None:
                    return
                solution = normalize(phase1 + bword)
                if len(solution) > incumbent:
                    return
                if not start.apply_word(solution).is_solved():
                    raise RuntimeError("candidate full replay failed")
                # Recompute earliest DR split on normalized solution; normalization may merge boundary moves.
                p1n, p2n = earliest_dr_split(start, solution)
                record(p1n, p2n, solution)
                return
            if remaining == 0:
                return

            children = []
            for mi, move in enumerate(MOVE_ORDER):
                if not allow(last_face, move):
                    continue
                nq = q_move(q, mi, co, eo, sl)
                hv = phase1_lb(nq, cos, eos)
                if hv <= remaining - 1:
                    # Prefer true twist events when quotient progress ties; this is ordering only, not pruning.
                    event_rank = 0 if move in T_MOVES else 1
                    children.append((hv, event_rank, move, nq))
            children.sort(key=lambda x: (x[0], x[1], x[2]))
            for _, _, move, nq in children:
                path.append(move)
                dfs(nq, remaining - 1, move[0])
                path.pop()
                if cut:
                    return

        dfs(q0, depth, None)
        print(
            f"phase1 depth {depth}: nodes={local_nodes:,}, DR-goals={local_goals:,}, "
            f"solutions={len(candidates)}, incumbent={incumbent}",
            flush=True,
        )
        if cut:
            break

    ranked = sorted(candidates.values(), key=lambda c: c.rank())

    print("\nTWIST-SKELETON SEARCH")
    print("=" * 72)
    print("scramble      :", args.scramble)
    print("status        :", "SAT" if ranked else "UNKNOWN")
    print("search cut    :", cut)
    print("nodes         :", f"{nodes:,}")
    print("DR goals      :", f"{goals:,}")
    print("exact B calls :", f"{exact_b:,}")
    print("elapsed       :", f"{time.time()-t0:.2f}s")
    print("candidates    :", len(ranked))

    if not ranked:
        return

    best_total = ranked[0].total
    equal_best = [c for c in ranked if c.total == best_total]
    best_skeleton = min(equal_best, key=lambda c: (c.twist_events, len(c.phase1), c.solution))
    print("\nBEST TOTAL")
    print("total         :", best_skeleton.total)
    print("phase1/phase2 :", (len(best_skeleton.phase1), len(best_skeleton.phase2)))
    print("twist events  :", best_skeleton.twist_events)
    print("K moves P1    :", best_skeleton.k_moves)
    print("solution      :", " ".join(best_skeleton.solution))
    print("phase1        :", " ".join(best_skeleton.phase1) or "(empty)")
    print("phase2        :", " ".join(best_skeleton.phase2) or "(empty)")

    print("\nNORMAL FORM")
    for kind, word in segment_phase1(best_skeleton.phase1):
        print(f"{kind}: {' '.join(word) or '(empty)'}")

    effects = block_effects(start, best_skeleton.phase1)
    if effects:
        print("\nK-BLOCK POSE/COLUMN EFFECTS")
        for i, item in enumerate(effects, 1):
            print(
                f"K{i}: {' '.join(item['word'])} | "
                f"boundary {item['boundary_delta']:+d}, "
                f"bundles {item['bundle_delta']:+d}, moved {item['moved_delta']:+d}"
            )

    print("\nTOP CANDIDATES")
    for i, cand in enumerate(ranked[: args.top], 1):
        print(
            f"{i:2}: total={cand.total:2} twist={cand.twist_events:2} "
            f"P1/P2=({len(cand.phase1)},{len(cand.phase2)}) "
            f"Kmoves={cand.k_moves:2} maxK={cand.max_k_block:2} "
            f"{' '.join(cand.solution)}"
        )


if __name__ == "__main__":
    main()

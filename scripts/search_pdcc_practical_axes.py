#!/usr/bin/env python3
"""CubeLab PDCC v22 practical multi-axis construction.

v22 keeps the v21 budget-invariant per-axis practical stream and adds an
adaptive *axis race* for construction mode:

  1. give UD/FB/RL the same tiny fixed P1-node pilot,
  2. rank axes by the pilot incumbent,
  3. break pilot ties only with cheap root quotient/T-event diagnostics,
  4. spend the remaining wall budget on the selected axis.

The pilot is deliberately node-budgeted rather than time-sliced, so cache/CPU
speed does not change which P1 prefix is sampled.  The selected main run starts
from the beginning; pilot work is therefore an explicit exploration overhead,
not hidden continuation.  All returned solutions are mapped back and replay
verified.  This is construction-only; no optimality claim is made.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import search_pdcc_guided as pg
import search_pdcc_practical as practical
import search_twist_skeleton as ts

AXES = {
    "UD": {"U":"U","D":"D","R":"R","L":"L","F":"F","B":"B"},
    "FB": {"U":"B","B":"D","D":"F","F":"U","R":"R","L":"L"},
    "RL": {"R":"U","U":"L","L":"D","D":"R","F":"F","B":"B"},
}


def _inverse_map(m):
    return {v:k for k,v in m.items()}


def map_move(move: str, fmap: dict[str,str]) -> str:
    return fmap[move[0]] + move[1:]


def map_word(word: Sequence[str], fmap: dict[str,str]) -> tuple[str,...]:
    return tuple(map_move(m, fmap) for m in word)


@dataclass(frozen=True)
class AxisFeatures:
    p1_lb: int
    best_t_lb: int
    t_desc: int
    best_kt_lb: int
    kt_desc: int
    best_t_relation: tuple

    def tie_key(self):
        # Lower is better.  The counts are negated because more descending
        # event options are useful when all earlier diagnostics tie.
        return (
            self.p1_lb,
            self.best_kt_lb,
            self.best_t_lb,
            -self.kt_desc,
            -self.t_desc,
            self.best_t_relation,
        )


@dataclass
class AxisRun:
    axis: str
    rotated_best: int
    solution: tuple[str,...]
    nodes: int
    dr_entries: int
    exact_b_calls: int
    elapsed: float
    features: AxisFeatures | None = None


@dataclass
class AxisRaceResult:
    best: AxisRun
    selected_axis: str
    pilots: tuple[AxisRun, ...]
    main: AxisRun
    pilot_elapsed: float
    remaining_budget: float
    total_elapsed: float


def axis_root_features(scramble: Sequence[str], tables, axis: str) -> AxisFeatures:
    """Cheap axis diagnostics; no P2/exact-solution information is used."""
    co, eo, sl, cos, eos, *_ = tables
    fmap = AXES[axis]
    rscr = map_word(scramble, fmap)
    st = ts.engine.from_word(" ".join(rscr))
    q = ts.q_of(st)
    h0 = ts.phase1_lb(q, cos, eos)

    best_t = 10**9
    t_desc = 0
    best_rel = None
    for tm in sorted(ts.T_MOVES):
        mi = ts.MI[tm]
        tq = ts.q_move(q, mi, co, eo, sl)
        th = ts.phase1_lb(tq, cos, eos)
        if th < h0:
            t_desc += 1
        nposes = pg.move_poses(st.poses, mi)
        rk = pg.relation_key(nposes)
        if th < best_t or (th == best_t and (best_rel is None or rk < best_rel)):
            best_t = th
            best_rel = rk

    best_kt = 10**9
    kt_desc = 0
    # One K transport followed by one T event.  This is still only a root
    # diagnostic: it asks how cheaply this DR axis can expose a productive
    # twist event after one legal transport column.
    for km in ts.K_MOVES:
        kmi = ts.MI[km]
        kq = ts.q_move(q, kmi, co, eo, sl)
        kface = km[0]
        local_best = 10**9
        for tm in ts.T_MOVES:
            if not ts.allow(kface, tm):
                continue
            tq = ts.q_move(kq, ts.MI[tm], co, eo, sl)
            th = ts.phase1_lb(tq, cos, eos)
            local_best = min(local_best, th)
        if local_best < h0:
            kt_desc += 1
        best_kt = min(best_kt, local_best)

    if best_rel is None:
        best_rel = (10**9,)
    return AxisFeatures(h0, best_t, t_desc, best_kt, kt_desc, best_rel)


def solve_axis(
    scramble: Sequence[str], tables, axis: str, *,
    time_limit: float,
    node_cap: int = 10_000_000,
    depth_slice_nodes: int = 20_000,
    p1_policy: str="event_relation", terminal_batch: int=32, terminal_top: int=4,
    use_tt: bool=False,
    compute_features: bool=False,
) -> AxisRun:
    fmap=AXES[axis]; inv=_inverse_map(fmap)
    rscr=map_word(scramble,fmap)
    r=practical.solve_practical(
        rscr,tables,time_limit=time_limit,node_cap=node_cap,
        depth_slice_nodes=depth_slice_nodes,p1_policy=p1_policy,
        terminal_batch=terminal_batch,terminal_top=terminal_top,use_tt=use_tt,
    )
    sol=map_word(r.solution,inv)
    orig=ts.engine.from_word(" ".join(scramble))
    if not orig.apply_word(sol).is_solved():
        raise RuntimeError(f"axis {axis} conjugated solution replay failed")
    feat = axis_root_features(scramble, tables, axis) if compute_features else None
    return AxisRun(axis,r.best_total,sol,r.nodes,r.dr_entries,r.exact_b_calls,r.elapsed,feat)


def solve_three_axis(
    scramble: Sequence[str], tables, *, total_time: float=3.0,
    per_axis_node_cap: int | None = None,
    depth_slice_nodes: int = 20_000,
    p1_policy: str="event_relation", terminal_batch: int=32, terminal_top: int=4,
    use_tt: bool=False,
):
    per=max(0.01,total_time/3.0)
    cap=10_000_000 if per_axis_node_cap is None else max(1,int(per_axis_node_cap))
    runs=[solve_axis(scramble,tables,a,time_limit=per,node_cap=cap,
                    depth_slice_nodes=depth_slice_nodes,p1_policy=p1_policy,
                    terminal_batch=terminal_batch,terminal_top=terminal_top,
                    use_tt=use_tt,compute_features=True) for a in ("UD","FB","RL")]
    runs.sort(key=lambda r:(r.rotated_best,r.exact_b_calls,r.elapsed,r.axis))
    return runs[0], runs


def solve_axis_race(
    scramble: Sequence[str], tables, *, total_time: float=3.0,
    pilot_nodes: int=4_000,
    pilot_time_cap: float=0.75,
    depth_slice_nodes: int=20_000,
    p1_policy: str="event_relation", terminal_batch: int=32, terminal_top: int=4,
    use_tt: bool=False,
) -> AxisRaceResult:
    """Tiny equal-node axis pilots, then concentrate remaining wall budget."""
    t0 = time.time()
    pilots=[]
    for axis in ("UD","FB","RL"):
        elapsed = time.time() - t0
        left = max(0.01, total_time - elapsed)
        # A generous time cap protects the total budget, but the pilot's normal
        # stopping condition is the fixed P1 node count.
        tcap = min(max(0.01,pilot_time_cap), left)
        pilots.append(solve_axis(
            scramble,tables,axis,time_limit=tcap,node_cap=max(1,pilot_nodes),
            depth_slice_nodes=depth_slice_nodes,p1_policy=p1_policy,
            terminal_batch=terminal_batch,terminal_top=terminal_top,use_tt=use_tt,
            compute_features=True,
        ))
        if time.time() - t0 >= total_time:
            break

    pilot_elapsed = time.time() - t0
    if not pilots:
        raise RuntimeError("axis race produced no pilot")

    # Pilot incumbent is primary.  Cheap root diagnostics only break equal
    # incumbent ties; they never override an actually better pilot solution.
    def pkey(r: AxisRun):
        feat = r.features or axis_root_features(scramble,tables,r.axis)
        return (r.rotated_best, feat.tie_key(), r.exact_b_calls, r.axis)

    selected=min(pilots,key=pkey)
    remaining=max(0.01,total_time-(time.time()-t0))
    main=solve_axis(
        scramble,tables,selected.axis,time_limit=remaining,node_cap=10_000_000,
        depth_slice_nodes=depth_slice_nodes,p1_policy=p1_policy,
        terminal_batch=terminal_batch,terminal_top=terminal_top,use_tt=use_tt,
        compute_features=True,
    )

    # Preserve any pilot incumbent even if the selected fresh run receives a
    # smaller remaining wall budget than its pilot happened to consume.
    candidates=pilots+[main]
    best=min(candidates,key=lambda r:(r.rotated_best,r.exact_b_calls,r.elapsed,r.axis))
    total_elapsed=time.time()-t0
    return AxisRaceResult(best,selected.axis,tuple(pilots),main,pilot_elapsed,remaining,total_elapsed)



def solve_axis_allocation(
    scramble: Sequence[str], tables, *, allocation: tuple[float,float,float],
    depth_slice_nodes: int = 20_000,
    p1_policy: str="event_relation", terminal_batch: int=32, terminal_top: int=4,
    use_tt: bool=False,
):
    """Run independent UD/FB/RL construction streams with fixed wall allocations.

    This is intentionally selector-free: each axis receives its declared budget,
    and the best replay-verified solution wins.  It is useful as a robust hedge
    when pilot winner selection is noisy.
    """
    vals=tuple(max(0.0,float(x)) for x in allocation)
    if sum(vals) <= 0:
        raise ValueError("axis allocation must contain positive total time")
    runs=[]
    for axis,secs in zip(("UD","FB","RL"), vals):
        if secs <= 0:
            continue
        runs.append(solve_axis(
            scramble,tables,axis,time_limit=secs,node_cap=10_000_000,
            depth_slice_nodes=depth_slice_nodes,p1_policy=p1_policy,
            terminal_batch=terminal_batch,terminal_top=terminal_top,
            use_tt=use_tt,compute_features=True,
        ))
    best=min(runs,key=lambda r:(r.rotated_best,r.exact_b_calls,r.elapsed,r.axis))
    return best,runs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--scramble',required=True)
    ap.add_argument('--time-limit',type=float,default=3.0)
    ap.add_argument('--mode',choices=('equal','race','allocation'),default='race')
    ap.add_argument('--axis-allocation',default='2,0.5,0.5',help='UD,FB,RL seconds for --mode allocation')
    ap.add_argument('--pilot-nodes',type=int,default=4_000)
    ap.add_argument('--pilot-time-cap',type=float,default=0.75)
    ap.add_argument('--per-axis-node-cap',type=int,default=None)
    ap.add_argument('--depth-slice-nodes',type=int,default=20_000)
    ap.add_argument('--p1-policy',choices=('relation_first','event_relation','p1_relation','star_first'),default='event_relation')
    ap.add_argument('--terminal-batch',type=int,default=32)
    ap.add_argument('--terminal-top',type=int,default=4)
    ap.add_argument('--tt',action='store_true')
    ap.add_argument('--cache',default='reports/pdcc_cache/twist_skeleton_tables_v1.pkl')
    a=ap.parse_args()
    tables,_=ts.cache_load_or_build(Path(a.cache))
    scr=tuple(a.scramble.split())

    print('# PDCC PRACTICAL v23 -- AXIS ALLOCATION / RACE')
    print('scramble       :',' '.join(scr))
    print('mode           :',a.mode)
    print('total budget   :',f'{a.time_limit:.3f}s')
    print('P1 policy      :',a.p1_policy)
    print('P1 slice nodes :',f'{a.depth_slice_nodes:,}')

    if a.mode=='equal':
        best,runs=solve_three_axis(scr,tables,total_time=a.time_limit,
                                  per_axis_node_cap=a.per_axis_node_cap,
                                  depth_slice_nodes=a.depth_slice_nodes,
                                  p1_policy=a.p1_policy,
                                  terminal_batch=a.terminal_batch,
                                  terminal_top=a.terminal_top,use_tt=a.tt)
        for r in runs:
            f=r.features
            print(f'{r.axis}: best={r.rotated_best:2d} nodes={r.nodes:8,d} E={r.dr_entries:5,d} B={r.exact_b_calls:4,d} '
                  f'rootLB={f.p1_lb if f else -1} KT={f.best_kt_lb if f else -1} t={r.elapsed:6.3f}s')
        minlen=min(r.rotated_best for r in runs)
        print('winner set     :',','.join(sorted(r.axis for r in runs if r.rotated_best==minlen)))
        print('selected winner:',best.axis)
    elif a.mode=='allocation':
        vals=tuple(float(x) for x in a.axis_allocation.split(','))
        if len(vals)!=3:
            raise SystemExit('--axis-allocation must be UD,FB,RL seconds')
        if abs(sum(vals)-a.time_limit)>1e-9:
            print(f'NOTE: allocation total {sum(vals):.3f}s overrides nominal --time-limit {a.time_limit:.3f}s')
        best,runs=solve_axis_allocation(
            scr,tables,allocation=vals,depth_slice_nodes=a.depth_slice_nodes,
            p1_policy=a.p1_policy,terminal_batch=a.terminal_batch,
            terminal_top=a.terminal_top,use_tt=a.tt,
        )
        print('allocation     :',a.axis_allocation)
        for r in runs:
            f=r.features
            print(f'{r.axis}: best={r.rotated_best:2d} nodes={r.nodes:8,d} E={r.dr_entries:5,d} B={r.exact_b_calls:4,d} '
                  f'rootLB={f.p1_lb if f else -1} KT={f.best_kt_lb if f else -1} t={r.elapsed:6.3f}s')
        minlen=min(r.rotated_best for r in runs)
        print('winner set     :',','.join(sorted(r.axis for r in runs if r.rotated_best==minlen)))
        print('selected winner:',best.axis)
    else:
        rr=solve_axis_race(scr,tables,total_time=a.time_limit,pilot_nodes=a.pilot_nodes,
                           pilot_time_cap=a.pilot_time_cap,depth_slice_nodes=a.depth_slice_nodes,
                           p1_policy=a.p1_policy,terminal_batch=a.terminal_batch,
                           terminal_top=a.terminal_top,use_tt=a.tt)
        for r in rr.pilots:
            f=r.features
            print(f'pilot {r.axis}: best={r.rotated_best:2d} nodes={r.nodes:7,d} E={r.dr_entries:4,d} B={r.exact_b_calls:3,d} '
                  f'LB={f.p1_lb} T={f.best_t_lb}/{f.t_desc} KT={f.best_kt_lb}/{f.kt_desc} t={r.elapsed:6.3f}s')
        print('selected axis  :',rr.selected_axis)
        print(f'main {rr.main.axis}: best={rr.main.rotated_best:2d} nodes={rr.main.nodes:8,d} E={rr.main.dr_entries:5,d} B={rr.main.exact_b_calls:4,d} t={rr.main.elapsed:6.3f}s')
        print(f'pilot elapsed  : {rr.pilot_elapsed:.3f}s')
        print(f'main budget    : {rr.remaining_budget:.3f}s')
        print(f'total elapsed  : {rr.total_elapsed:.3f}s')
        best=rr.best

    print('best total     :',best.rotated_best)
    print('best axis      :',best.axis)
    print('solution       :',' '.join(best.solution))
    print('full replay    :',ts.engine.from_word(' '.join(scr)).apply_word(best.solution).is_solved())
    print('certified      : False (construction mode)')
    return 0

if __name__=='__main__': raise SystemExit(main())

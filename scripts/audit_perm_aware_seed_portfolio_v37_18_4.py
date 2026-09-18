#!/usr/bin/env python3
"""CubeLab v37.18.4 - permutation-aware bounded seed portfolio calibration.

Research-only. Production changes: NONE.

Targeted quick calibration on the v37.18.2 live outliers (default: 6,8,9).

Baseline seed:
    shortest feasible exact-H
    -> first deterministic Orientation completion
    -> full exact K2 up to max_k2

Portfolio seed:
    1. materialize a small deterministic set of exact-H DR terminals across
       ALL supported H depths;
    2. score each terminal cheaply by:
           H + max(P2LB, DStarLB)
    3. search total seed budgets from the smallest lower bound upward;
    4. for every eligible candidate ask only:
           exact K2 <= total_budget - H ?
       using the monotone bounded K2 oracle;
    5. first success is the exact best seed inside the materialized portfolio.

The resulting seed is only an upper bound for the main solver.  The subsequent
sound no-closed-cache BnB still proves the global exact optimum, so seed
portfolio incompleteness cannot damage exactness.

This audit compares live DET vs PORT in one process with counterbalanced order.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
SRC=ROOT/"src"
for p in (str(SRC),str(HERE),str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0,p)

import pdcc_double_star as dstar
import pdcc_p1_pose as p1pose
import pdcc_phase2 as p2
import search_pdcc_guided as pg
import search_twist_skeleton as ts
from cubelab.pdcc import PDCCState

from p1_orientation_domains_v37_0 import D6MoveDomainOracle, MOVE_INDEX
from audit_bnb_soundness_v37_18_1 import NoClosedCacheBnB
from audit_bnb_k2_attribution_v37_18_3_1 import TracedNoClosedBnB


def clear_p2():
    fn=getattr(p2,"clear_double_star_query_cache",None)
    if callable(fn):
        fn()


def terminal_lb(pk,tables):
    c,u,s=ts.unpack_p(pk)
    p2lb=max(
        int(tables[8][c]),
        int(tables[9][u]),
        int(tables[10][s]),
    )
    ds=max((int(x) for x in dstar.profile(pk)),default=0)
    return p2lb,ds,max(p2lb,ds)


def collect_depth_terminals(
    start,
    q0,
    previous_face,
    depth,
    limit,
    oo,
    tables,
):
    """Collect up to limit exact true-DR H completions at one exact depth."""
    out=[]
    seen=set()

    def rec(poses,q,rem,last,parent_q,prefix):
        if len(out)>=limit:
            return

        if rem==0:
            if int(q)!=int(ts.GOAL_Q):
                return
            if parent_q==ts.GOAL_Q:
                return

            state=PDCCState(poses)
            if not ts.is_dr(state):
                raise RuntimeError(
                    f"q=GOAL depth={depth} terminal is not full DR"
                )
            pk=ts.p_of(state)
            if pk is None:
                raise RuntimeError("DR terminal has no Phase2 coordinate")

            boundary_face=last
            dedupe=(int(pk),boundary_face)
            if dedupe in seen:
                return
            seen.add(dedupe)

            p2lb,ds,combined=terminal_lb(pk,tables)
            out.append({
                "h":int(depth),
                "h_word":tuple(prefix),
                "pk":int(pk),
                "boundary_face":boundary_face,
                "p2lb":int(p2lb),
                "dstar_lb":int(ds),
                "combined_lb":int(combined),
                "lower_total":int(depth)+int(combined),
            })
            return

        dom=oo.allowed_domain(int(q),int(rem),last)
        for move in ts.MOVE_ORDER:
            if not dom.contains(move):
                continue
            mi=MOVE_INDEX[move]
            nq=ts.q_move(
                int(q),mi,oo.co,oo.eo,oo.sl
            )
            nposes=pg.move_poses(poses,mi)
            rec(
                nposes,
                int(nq),
                rem-1,
                move[0],
                q,
                prefix+(move,),
            )
            if len(out)>=limit:
                return

    rec(
        start.poses,
        int(q0),
        int(depth),
        previous_face,
        None,
        (),
    )
    return out


def build_terminal_pool(
    start,q0,previous_face,depths,per_depth,oo,tables
):
    t0=time.perf_counter()
    by_depth={}
    pool=[]

    for d in depths:
        got=collect_depth_terminals(
            start,q0,previous_face,
            int(d),int(per_depth),
            oo,tables,
        )
        by_depth[str(d)]=len(got)
        pool.extend(got)

    # Dedupe across depths by Phase2 state/boundary face.  Keep shorter H, then
    # lower cheap total bound.
    best={}
    for c in pool:
        key=(c["pk"],c["boundary_face"])
        old=best.get(key)
        if old is None or (
            c["h"],c["lower_total"],c["h_word"]
        ) < (
            old["h"],old["lower_total"],old["h_word"]
        ):
            best[key]=c

    pool=list(best.values())
    pool.sort(
        key=lambda c:(
            c["lower_total"],
            c["combined_lb"],
            c["h"],
            c["h_word"],
        )
    )

    return pool,by_depth,time.perf_counter()-t0


class PortfolioSeedBnB(NoClosedCacheBnB):
    def __init__(
        self,*args,
        seed_depths,
        seed_per_depth,
        **kwargs
    ):
        super().__init__(*args,**kwargs)
        self.seed_depths=tuple(seed_depths)
        self.seed_per_depth=int(seed_per_depth)
        self.seed_pool=[]
        self.seed_pool_by_depth={}
        self.seed_enum_wall=0.0
        self.seed_search_wall=0.0
        self.seed_k2_calls_before=0
        self.seed_k2_nodes_before=0
        self.seed_k2_wall_before=0.0
        self.seed_attempts=[]
        self.seed_chosen_rank=None

    def seed_incumbent(self,start,q0,previous_face,depths):
        pool,counts,enum_wall=build_terminal_pool(
            start,q0,previous_face,
            self.seed_depths,
            self.seed_per_depth,
            self.oo,
            self.tables,
        )
        if not pool:
            raise RuntimeError("portfolio seed materialized no DR terminals")

        self.seed_pool=pool
        self.seed_pool_by_depth=counts
        self.seed_enum_wall=enum_wall

        self.seed_k2_calls_before=self.k2.calls
        self.seed_k2_nodes_before=self.k2.nodes
        self.seed_k2_wall_before=self.k2.wall

        # Exact best seed inside this finite pool via total-budget thresholds.
        min_total=min(c["lower_total"] for c in pool)
        max_total=max(c["h"]+self.k2.max_depth for c in pool)

        t0=time.perf_counter()
        found=None

        for total_budget in range(int(min_total),int(max_total)+1):
            eligible=[
                (rank,c)
                for rank,c in enumerate(pool,1)
                if c["lower_total"]<=total_budget
                and c["h"]<=total_budget
            ]

            for rank,c in eligible:
                cap=int(total_budget)-int(c["h"])
                before_calls=self.k2.calls
                before_nodes=self.k2.nodes
                before_wall=self.k2.wall

                bword,bdist=self.k2.query(
                    c["pk"],c["boundary_face"],cap
                )

                self.seed_attempts.append({
                    "total_budget":int(total_budget),
                    "rank":int(rank),
                    "h":int(c["h"]),
                    "lower_total":int(c["lower_total"]),
                    "combined_lb":int(c["combined_lb"]),
                    "cap":int(cap),
                    "found":bword is not None and bdist is not None,
                    "dist":None if bdist is None else int(bdist),
                    "new_k2_calls":int(self.k2.calls-before_calls),
                    "new_k2_nodes":int(self.k2.nodes-before_nodes),
                    "new_k2_wall":float(self.k2.wall-before_wall),
                })

                if bword is None or bdist is None:
                    continue

                actual=int(c["h"])+int(bdist)
                if actual>total_budget:
                    raise RuntimeError(
                        "bounded portfolio K2 exceeded its total budget"
                    )
                found=(rank,c,tuple(bword),int(bdist),actual)
                break

            if found is not None:
                break

        self.seed_search_wall=time.perf_counter()-t0

        if found is None:
            raise RuntimeError("portfolio seed found no bounded K2 solution")

        rank,c,bword,bdist,total=found
        self.seed_chosen_rank=int(rank)
        self.incumbent=int(total)
        self.best={
            "total":int(total),
            "h_tail":tuple(c["h_word"]),
            "k2_word":tuple(bword),
            "k2_exact":int(bdist),
            "source":"PERM_AWARE_BOUNDED_SEED_PORTFOLIO",
        }

    def solve_portfolio(self,start,q0,previous_face,depths):
        t0=time.perf_counter()
        self.seed_incumbent(
            start,q0,previous_face,depths
        )
        after_seed=time.perf_counter()

        seed_total=int(self.incumbent)
        seed_best=dict(self.best)

        self.search(
            start.poses,int(q0),tuple(depths),
            previous_face,None,0,()
        )
        end=time.perf_counter()

        seed_k2_calls=self.k2.calls-self.seed_k2_calls_before
        seed_k2_nodes=self.k2.nodes-self.seed_k2_nodes_before
        seed_k2_wall=self.k2.wall-self.seed_k2_wall_before

        return {
            "seed_total":seed_total,
            "seed_witness":seed_best,
            "witness":self.best,
            "seed_pool_size":len(self.seed_pool),
            "seed_pool_by_depth":self.seed_pool_by_depth,
            "seed_chosen_rank":self.seed_chosen_rank,
            "seed_enum_wall":self.seed_enum_wall,
            "seed_search_wall":self.seed_search_wall,
            "seed_total_wall":after_seed-t0,
            "seed_k2_calls":seed_k2_calls,
            "seed_k2_nodes":seed_k2_nodes,
            "seed_k2_wall":seed_k2_wall,
            "seed_attempts":self.seed_attempts,
            "nodes":self.nodes,
            "terminals":self.terminals,
            "k2_calls":self.k2.calls,
            "k2_nodes":self.k2.nodes,
            "k2_wall":self.k2.wall,
            "state_bound_prunes":self.state_bound_prunes,
            "child_bound_prunes":self.child_bound_prunes,
            "terminal_lb_skips":self.terminal_lb_skips,
            "improvements":self.improvements,
            "proof_wall_after_seed":end-after_seed,
            "wrapper_wall":end-t0,
        }


def replay(start,w):
    return (
        w is not None
        and start.apply_word(
            tuple(w["h_tail"])+tuple(w["k2_word"])
        ).is_solved()
    )


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--v3715-json",
        default="reports/v37/unified_total_budget_v37_15.json",
    )
    ap.add_argument(
        "--v371831-json",
        default="reports/v37/bnb_k2_attribution_v37_18_3_1.json",
    )
    ap.add_argument(
        "--d6-table",
        default="reports/pdcc_cache/p1_tail_nosuffix_v36_6i13_d6.pkl",
    )
    ap.add_argument(
        "--cache",
        default="reports/pdcc_cache/twist_skeleton_tables_v1.pkl",
    )
    ap.add_argument(
        "--depths",type=int,nargs="+",default=[1,2,3,4,5,6],
    )
    ap.add_argument(
        "--cases",type=int,nargs="+",default=[6,8,9],
    )
    ap.add_argument("--seed-per-depth",type=int,default=16)
    ap.add_argument("--max-k2",type=int,default=18)
    ap.add_argument("--p2-order",default="auto")
    ap.add_argument("--p2-auto-probe-nodes",type=int,default=128)
    ap.add_argument(
        "--p1-pose-cache",
        default="reports/pdcc_cache/p1_corner_star_tables_v1.pkl",
    )
    ap.add_argument(
        "--double-star-cache",
        default="reports/pdcc_cache/double_star_k_tables_v1.pkl",
    )
    ap.add_argument(
        "--double-star-index",
        default="reports/pdcc_cache/double_star_p2_index_v1.pkl",
    )
    ap.add_argument(
        "--output",
        default="reports/v37/perm_seed_portfolio_v37_18_4.json",
    )
    args=ap.parse_args()

    gt=json.loads(Path(args.v3715_json).read_text(encoding="utf-8"))
    rows0=gt.get("rows",[])
    if not rows0:
        raise SystemExit("v37.15 JSON has no rows")

    attr={}
    p=Path(args.v371831_json)
    if p.exists():
        a=json.loads(p.read_text(encoding="utf-8"))
        attr={
            int(r["case"]):r
            for r in a.get("rows",[])
        }

    selected=[]
    for i in args.cases:
        if i<1 or i>len(rows0):
            raise SystemExit(f"case out of range: {i}")
        selected.append((i,rows0[i-1]))

    tables,loaded=ts.cache_load_or_build(Path(args.cache))
    co,eo,sl,cos,eos,*_=tables
    oo=D6MoveDomainOracle.load(
        Path(args.d6_table),co,eo,sl,cos,eos
    )

    p2.configure_double_star_cache(
        args.double_star_cache,args.double_star_index
    )
    dstar.configure(
        args.double_star_cache,args.double_star_index
    )
    p1pose.configure(args.p1_pose_cache)
    p1pose.preload()

    depths=tuple(sorted(set(
        int(d) for d in args.depths
        if 1<=int(d)<=int(oo.table.max_depth)
    )))

    print("# CubeLab v37.18.4 - PERMUTATION-AWARE BOUNDED SEED PORTFOLIO")
    print()
    print("target cases         :",[x[0] for x in selected])
    print("H depth domain       :",depths)
    print("terminals/depth      :",args.seed_per_depth)
    print("DET                   : first shortest-H + full exact K2")
    print("PORT                  : multi-H pool + LB-ranked bounded K2")
    print("exact proof          : sound no-closed-cache BnB")
    print("production changes   : NONE")
    print()

    rows=[]
    failures=[]
    t0=time.perf_counter()

    for j,(case,g) in enumerate(selected):
        scramble=tuple(g["scramble"])
        start=ts.engine.from_word(" ".join(scramble))
        q0=ts.q_of(start)
        gt_total=int(g["gt_best_total"])

        order=("DET","PORT") if j%2==0 else ("PORT","DET")
        got={}

        for arm in order:
            clear_p2()

            if arm=="DET":
                solver=TracedNoClosedBnB(
                    tables,oo,
                    p2_order=args.p2_order,
                    p2_probe=args.p2_auto_probe_nodes,
                    max_k2=args.max_k2,
                    p1_pose_cache=args.p1_pose_cache,
                )
                w0=time.perf_counter()
                res=solver.solve_traced(
                    start,q0,None,depths
                )
                wall=time.perf_counter()-w0
                got["DET"]=(res,wall)
            else:
                solver=PortfolioSeedBnB(
                    tables,oo,
                    p2_order=args.p2_order,
                    p2_probe=args.p2_auto_probe_nodes,
                    max_k2=args.max_k2,
                    p1_pose_cache=args.p1_pose_cache,
                    seed_depths=depths,
                    seed_per_depth=args.seed_per_depth,
                )
                w0=time.perf_counter()
                res=solver.solve_portfolio(
                    start,q0,None,depths
                )
                wall=time.perf_counter()-w0
                got["PORT"]=(res,wall)

        det,dwall=got["DET"]
        port,pwall=got["PORT"]
        dw=det["witness"]
        pw=port["witness"]

        parity=(
            dw is not None and pw is not None
            and int(dw["total"])==gt_total
            and int(pw["total"])==gt_total
            and replay(start,dw)
            and replay(start,pw)
        )
        if not parity:
            failures.append(case)

        oracle_wall=None
        if case in attr:
            oracle_wall=float(attr[case]["oracle_bnb"]["wall"])

        row={
            "case":case,
            "order":list(order),
            "gt_total":gt_total,
            "parity":parity,
            "det":{
                "seed_total":int(det["seed_total"]),
                "seed_h":int(det["seed_h"]),
                "seed_k2":int(det["seed_k2"]),
                "seed_wall":float(det["seed_wall"]),
                "bounded_k2_wall":float(det["bounded_k2_wall"]),
                "wrapper_wall":dwall,
                "nodes":int(det["nodes"]),
                "k2_calls":int(det["total_k2_calls"]),
            },
            "port":{
                "seed_total":int(port["seed_total"]),
                "seed_h":len(port["seed_witness"]["h_tail"]),
                "seed_k2":int(port["seed_witness"]["k2_exact"]),
                "seed_pool_size":int(port["seed_pool_size"]),
                "seed_pool_by_depth":port["seed_pool_by_depth"],
                "seed_chosen_rank":int(port["seed_chosen_rank"]),
                "seed_enum_wall":float(port["seed_enum_wall"]),
                "seed_search_wall":float(port["seed_search_wall"]),
                "seed_total_wall":float(port["seed_total_wall"]),
                "seed_k2_wall":float(port["seed_k2_wall"]),
                "seed_k2_calls":int(port["seed_k2_calls"]),
                "proof_wall_after_seed":float(port["proof_wall_after_seed"]),
                "wrapper_wall":pwall,
                "nodes":int(port["nodes"]),
                "k2_calls":int(port["k2_calls"]),
                "improvements":int(port["improvements"]),
            },
            "oracle_wall_from_v371831":oracle_wall,
            "wall_ratio_port_over_det":(
                pwall/dwall if dwall else None
            ),
        }
        rows.append(row)

        print(
            f"[case {case}] order={'->'.join(order)} GT={gt_total} "
            f"DETseed={row['det']['seed_h']}+{row['det']['seed_k2']}="
            f"{row['det']['seed_total']} "
            f"seedWall={row['det']['seed_wall']:.3f}s total={dwall:.3f}s | "
            f"PORTseed={row['port']['seed_h']}+{row['port']['seed_k2']}="
            f"{row['port']['seed_total']} "
            f"pool={row['port']['seed_pool_size']} rank={row['port']['seed_chosen_rank']} "
            f"seedK2q={row['port']['seed_k2_calls']} "
            f"seedWall={row['port']['seed_total_wall']:.3f}s "
            f"total={pwall:.3f}s x{(pwall/dwall if dwall else 0):.3f} "
            f"oracle={oracle_wall if oracle_wall is not None else '-'} "
            f"parity={'PASS' if parity else 'FAIL'}"
        )

    good=[r for r in rows if r["parity"]]
    det_wall=sum(r["det"]["wrapper_wall"] for r in good)
    port_wall=sum(r["port"]["wrapper_wall"] for r in good)
    det_seed=sum(r["det"]["seed_wall"] for r in good)
    port_seed=sum(r["port"]["seed_total_wall"] for r in good)
    det_k2=sum(r["det"]["k2_calls"] for r in good)
    port_k2=sum(r["port"]["k2_calls"] for r in good)

    seed_better=sum(
        r["port"]["seed_total"]<r["det"]["seed_total"]
        for r in good
    )
    seed_equal=sum(
        r["port"]["seed_total"]==r["det"]["seed_total"]
        for r in good
    )

    elapsed=time.perf_counter()-t0
    overall=not failures

    print()
    print("# SUMMARY")
    print("exact parity         :",f"{len(good)}/{len(rows)}")
    print("failures             :",failures or "-")
    print(
        "PORT/DET total wall :",
        f"{port_wall/det_wall:.3f}x" if det_wall else "n/a",
    )
    print(
        "PORT/DET seed wall  :",
        f"{port_seed/det_seed:.3f}x" if det_seed else "n/a",
    )
    print(
        "PORT/DET K2 calls   :",
        f"{port_k2/det_k2:.3f}x" if det_k2 else "n/a",
    )
    print("portfolio seed better:",seed_better)
    print("portfolio seed equal :",seed_equal)
    print("audit wall           :",f"{elapsed:.3f}s")

    if overall and det_wall and port_wall<0.80*det_wall:
        conclusion="PERM_AWARE_BOUNDED_SEED_STRONG_SIGNAL"
    elif overall and det_wall and port_wall<det_wall:
        conclusion="PERM_AWARE_BOUNDED_SEED_POSITIVE_SIGNAL"
    elif overall:
        conclusion="PERM_AWARE_BOUNDED_SEED_NO_WALL_GAIN"
    else:
        conclusion="PERM_AWARE_BOUNDED_SEED_PARITY_FAIL"
    print("conclusion           :",conclusion)

    payload={
        "version":"v37.18.4",
        "source_v3715_json":args.v3715_json,
        "source_v371831_json":(
            args.v371831_json if p.exists() else None
        ),
        "cases":[x[0] for x in selected],
        "depths":depths,
        "seed_per_depth":args.seed_per_depth,
        "rows":rows,
        "failures":failures,
        "weighted_port_over_det_wall":(
            None if not det_wall else port_wall/det_wall
        ),
        "weighted_port_over_det_seed_wall":(
            None if not det_seed else port_seed/det_seed
        ),
        "weighted_port_over_det_k2_calls":(
            None if not det_k2 else port_k2/det_k2
        ),
        "portfolio_seed_better_cases":seed_better,
        "portfolio_seed_equal_cases":seed_equal,
        "elapsed":elapsed,
        "overall_pass":overall,
        "conclusion":conclusion,
        "scope_note":(
            "Targeted calibration on previous live outliers. The finite seed "
            "portfolio affects only the initial upper bound; exactness still "
            "comes from the full sound no-closed-cache BnB proof."
        ),
    }

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(payload,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    print("JSON                 :",out)
    return 0 if overall else 2


if __name__=="__main__":
    raise SystemExit(main())

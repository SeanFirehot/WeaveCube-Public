#!/usr/bin/env python3
"""CubeLab v37.18.3.1 - BnB K2-cost attribution + oracle-seed calibration.

Research-only. Production changes: NONE.

v37.18.2 established:
- sound BnB exact parity 10/10;
- BnB nodes = 0.109x MEMO;
- BnB K2 calls = 0.722x MEMO;
- BnB wall = 2.105x MEMO;
- most cases favor BnB, but a few expensive outliers dominate weighted wall.

This audit separates:
1. deterministic seed exact-K2 cost;
2. later bounded-improvement K2 cost;
3. non-K2 BnB traversal cost.

It also runs an OFFLINE ORACLE-SEED calibration:
- initialize incumbent/best with the frozen v37.15 exact witness;
- run the same sound no-closed-cache BnB only to prove no better solution exists.

The oracle arm is NOT a runtime algorithm.  It isolates whether the BnB search
kernel is fast once a good incumbent is available.
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
import pdcc_phase2 as p2
import search_twist_skeleton as ts

from p1_orientation_domains_v37_0 import D6MoveDomainOracle
from p1_orientation_synthesis_v37_1 import synthesize_orientation_suffix
from audit_bnb_soundness_v37_18_1 import NoClosedCacheBnB
from op_total_value_v37_17 import OrientationTransitionCompiler
from audit_permutation_secondary_v37_2_1 import (
    HARD20,HARD20_PREFIX12,RL_MAP,map_word,
)


class TracedNoClosedBnB(NoClosedCacheBnB):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.seed_wall=0.0
        self.seed_nodes=0
        self.seed_h=None
        self.seed_k2=None
        self.seed_terminal_lb=None
        self.bounded_query_log=[]

        original_query=self.k2.query

        def traced_query(pk,prev,cap):
            before_nodes=self.k2.nodes
            t0=time.perf_counter()
            word,dist=original_query(pk,prev,cap)
            wall=time.perf_counter()-t0
            self.bounded_query_log.append({
                "cap":int(cap),
                "found":word is not None and dist is not None,
                "dist":None if dist is None else int(dist),
                "wall":wall,
                "nodes":int(self.k2.nodes-before_nodes),
            })
            return word,dist

        self.k2.query=traced_query

    def seed_incumbent(self,start,q0,previous_face,depths):
        """Same v37.18 seed policy, but separately instrumented."""
        for h in sorted(set(int(d) for d in depths if int(d)>0)):
            syn=synthesize_orientation_suffix(
                int(q0),h,previous_face,self.oo
            )
            if not syn.success:
                continue

            hword=tuple(syn.word)
            terminal=start.apply_word(hword)
            if not ts.is_dr(terminal):
                raise RuntimeError(
                    f"seed H={h} is quotient-valid but not full DR"
                )
            pk=ts.p_of(terminal)
            if pk is None:
                raise RuntimeError("seed DR missing Phase2 coordinate")

            self.seed_terminal_lb=self.terminal_lb(pk)

            stats={}
            t0=time.perf_counter()
            bword,bdist=p2.shortest_b_compatible(
                int(pk),
                hword[-1][0] if hword else previous_face,
                self.k2.max_depth,
                self.k2.cp,self.k2.up,self.k2.sp,
                self.k2.cpd,self.k2.upd,self.k2.spd,
                order=self.k2.order,
                stats=stats,
                auto_probe_nodes=self.k2.probe,
            )
            wall=time.perf_counter()-t0

            self.seed_wall+=wall
            self.seed_nodes+=int(stats.get("nodes",0))
            self.k2.wall+=wall
            self.k2.calls+=1
            self.seed_k2_calls+=1
            self.k2.nodes+=int(stats.get("nodes",0))

            if bword is None or bdist is None:
                continue

            key=(
                int(pk),
                hword[-1][0] if hword else previous_face
            )
            self.k2.success[key]=(int(bdist),tuple(bword))

            self.seed_h=h
            self.seed_k2=int(bdist)
            self.incumbent=h+int(bdist)
            self.best={
                "total":self.incumbent,
                "h_tail":hword,
                "k2_word":tuple(bword),
                "k2_exact":int(bdist),
                "source":"TRACED_DETERMINISTIC_SEED",
            }
            return

        raise RuntimeError("could not construct deterministic seed")

    def solve_traced(self,start,q0,previous_face,depths):
        t0=time.perf_counter()
        self.seed_incumbent(start,q0,previous_face,depths)
        after_seed=time.perf_counter()
        seed_best=dict(self.best)
        seed_total=int(self.incumbent)

        self.search(
            start.poses,int(q0),tuple(depths),
            previous_face,None,0,()
        )
        end=time.perf_counter()

        bounded_wall=sum(x["wall"] for x in self.bounded_query_log)
        bounded_nodes=sum(x["nodes"] for x in self.bounded_query_log)
        return {
            "seed_total":seed_total,
            "seed_witness":seed_best,
            "witness":self.best,
            "nodes":self.nodes,
            "terminals":self.terminals,
            "state_bound_prunes":self.state_bound_prunes,
            "child_bound_prunes":self.child_bound_prunes,
            "terminal_lb_skips":self.terminal_lb_skips,
            "improvements":self.improvements,
            "pose_bound_calls":self.pose_bound_calls,
            "seed_h":self.seed_h,
            "seed_k2":self.seed_k2,
            "seed_terminal_lb":self.seed_terminal_lb,
            "seed_wall":self.seed_wall,
            "seed_nodes":self.seed_nodes,
            "bounded_k2_calls":len(self.bounded_query_log),
            "bounded_k2_wall":bounded_wall,
            "bounded_k2_nodes":bounded_nodes,
            "bounded_query_log":self.bounded_query_log,
            "total_k2_calls":self.k2.calls,
            "total_k2_wall":self.k2.wall,
            "total_k2_nodes":self.k2.nodes,
            "search_wall_after_seed":end-after_seed,
            "wrapper_wall":end-t0,
            "estimated_non_k2_wall":max(
                0.0,(end-t0)-self.seed_wall-bounded_wall
            ),
        }


class OracleSeedNoClosedBnB(NoClosedCacheBnB):
    """Offline calibration: valid frozen optimal witness as initial incumbent."""

    def solve_oracle(
        self,start,q0,previous_face,depths,
        oracle_witness,
    ):
        hword=tuple(oracle_witness["h_tail"])
        kword=tuple(oracle_witness["k2_word"])
        k2d=int(oracle_witness["k2_exact"])
        total=int(oracle_witness["total"])

        if total!=len(hword)+k2d:
            raise RuntimeError("oracle witness cost mismatch")
        if not start.apply_word(hword+kword).is_solved():
            raise RuntimeError("oracle witness replay failed")

        self.incumbent=total
        self.best={
            "total":total,
            "h_tail":hword,
            "k2_word":kword,
            "k2_exact":k2d,
            "source":"OFFLINE_ORACLE_SEED",
        }

        t0=time.perf_counter()
        self.search(
            start.poses,int(q0),tuple(depths),
            previous_face,None,0,()
        )
        wall=time.perf_counter()-t0

        return {
            "witness":self.best,
            "nodes":self.nodes,
            "terminals":self.terminals,
            "k2_calls":self.k2.calls,
            "k2_wall":self.k2.wall,
            "k2_nodes":self.k2.nodes,
            "state_bound_prunes":self.state_bound_prunes,
            "child_bound_prunes":self.child_bound_prunes,
            "terminal_lb_skips":self.terminal_lb_skips,
            "improvements":self.improvements,
            "wall":wall,
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
        "--v37182-json",
        default="reports/v37/sound_bnb_vs_memo_live_v37_18_2.json",
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
        default="reports/v37/bnb_k2_attribution_v37_18_3.json",
    )
    args=ap.parse_args()

    gt=json.loads(Path(args.v3715_json).read_text(encoding="utf-8"))
    live=json.loads(Path(args.v37182_json).read_text(encoding="utf-8"))
    gt_rows=gt.get("rows",[])
    live_rows=live.get("rows",[])
    if not gt_rows or len(gt_rows)!=len(live_rows):
        raise SystemExit("frozen corpus mismatch")

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

    depths=tuple(sorted(set(
        int(d) for d in args.depths
        if 1<=int(d)<=int(oo.table.max_depth)
    )))

    print("# CubeLab v37.18.3.1 - BnB K2 COST ATTRIBUTION")
    print()
    print("cases                :",len(gt_rows))
    print("live source          :",args.v37182_json)
    print("seed policy          : v37.18 deterministic shortest-H")
    print("oracle arm           : OFFLINE CALIBRATION ONLY")
    print("production changes   : NONE")
    print()

    rows=[]
    failures=[]
    t0=time.perf_counter()

    for i,(g,l) in enumerate(zip(gt_rows,live_rows),1):
        scramble=tuple(g["scramble"])
        start=ts.engine.from_word(" ".join(scramble))
        q0=ts.q_of(start)
        gt_total_raw=g.get("gt_best_total", g.get("best_total"))
        if gt_total_raw is None:
            raise KeyError("v37.15 row lacks gt_best_total/best_total")
        gt_total=int(gt_total_raw)
        oracle=g["unified"]["witness"]

        # Traced live seed.
        trans=OrientationTransitionCompiler(oo)
        traced=TracedNoClosedBnB(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
            max_k2=args.max_k2,
            p1_pose_cache=args.p1_pose_cache,
            shared_transition_compiler=trans,
        )
        tr=traced.solve_traced(start,q0,None,depths)
        tw=tr["witness"]
        tparity=(
            tw is not None
            and int(tw["total"])==gt_total
            and replay(start,tw)
        )

        # Oracle-seed proof-only calibration.
        trans2=OrientationTransitionCompiler(oo)
        oracle_solver=OracleSeedNoClosedBnB(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
            max_k2=args.max_k2,
            p1_pose_cache=args.p1_pose_cache,
            shared_transition_compiler=trans2,
        )
        OR=oracle_solver.solve_oracle(
            start,q0,None,depths,oracle
        )
        ow=OR["witness"]
        oparity=(
            ow is not None
            and int(ow["total"])==gt_total
            and replay(start,ow)
        )

        if not (tparity and oparity):
            failures.append(i)

        live_bnb=float(l["bnb"]["wrapper_wall"])
        live_memo=float(l["memo"]["wrapper_wall"])
        seed_share=(
            tr["seed_wall"]/tr["wrapper_wall"]
            if tr["wrapper_wall"] else 0.0
        )
        total_k2_share=(
            tr["total_k2_wall"]/tr["wrapper_wall"]
            if tr["wrapper_wall"] else 0.0
        )

        row={
            "case":i,
            "scramble":list(scramble),
            "gt_total":gt_total,
            "traced_parity":tparity,
            "oracle_parity":oparity,
            "live_v37182":{
                "memo_wall":live_memo,
                "bnb_wall":live_bnb,
                "bnb_over_memo":live_bnb/live_memo if live_memo else None,
            },
            "traced_bnb":tr,
            "oracle_bnb":OR,
            "seed_excess":int(tr["seed_total"])-gt_total,
            "seed_wall_share":seed_share,
            "total_k2_wall_share":total_k2_share,
            "oracle_over_traced_wall":(
                OR["wall"]/tr["wrapper_wall"]
                if tr["wrapper_wall"] else None
            ),
        }
        rows.append(row)

        top=sorted(
            tr["bounded_query_log"],
            key=lambda x:x["wall"],
            reverse=True,
        )[:2]
        top_txt=", ".join(
            f"cap{q['cap']}:{q['wall']:.3f}s"
            for q in top
        ) or "-"

        print(
            f"[{i:2d}/{len(gt_rows)}] "
            f"GT={gt_total} seed={tr['seed_total']} "
            f"excess={row['seed_excess']} "
            f"seedK2={tr['seed_k2']} "
            f"seedWall={tr['seed_wall']:.3f}s "
            f"bounded={tr['bounded_k2_calls']}/"
            f"{tr['bounded_k2_wall']:.3f}s "
            f"nonK2~{tr['estimated_non_k2_wall']:.3f}s "
            f"total={tr['wrapper_wall']:.3f}s "
            f"oracle={OR['wall']:.3f}s "
            f"topBounded=[{top_txt}] "
            f"parity={'PASS' if tparity and oparity else 'FAIL'}"
        )

    # Hard20 traced seed only as regression.
    print()
    print("# HARD20 RL")
    hard_gt=gt.get("hard20_RL_prefix12")
    hard=None
    if hard_gt and hard_gt.get("witness") is not None:
        ms=map_word(HARD20,RL_MAP)
        mp=map_word(HARD20_PREFIX12,RL_MAP)
        start=ts.engine.from_word(" ".join(ms)).apply_word(mp)
        q0=ts.q_of(start)
        total=int(hard_gt["witness"]["total"])

        trans=OrientationTransitionCompiler(oo)
        solver=TracedNoClosedBnB(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
            max_k2=args.max_k2,
            p1_pose_cache=args.p1_pose_cache,
            shared_transition_compiler=trans,
        )
        hard=solver.solve_traced(
            start,q0,mp[-1][0],depths
        )
        hw=hard["witness"]
        parity=(
            hw is not None
            and int(hw["total"])==total
            and replay(start,hw)
        )
        hard["parity"]=parity
        if not parity:
            failures.append("hard20")
        print(
            f"GT={total} seed={hard['seed_total']} "
            f"seedWall={hard['seed_wall']:.4f}s "
            f"boundedWall={hard['bounded_k2_wall']:.4f}s "
            f"total={hard['wrapper_wall']:.4f}s "
            f"parity={'PASS' if parity else 'FAIL'}"
        )

    good=[r for r in rows if r["traced_parity"] and r["oracle_parity"]]

    total_wall=sum(r["traced_bnb"]["wrapper_wall"] for r in good)
    seed_wall=sum(r["traced_bnb"]["seed_wall"] for r in good)
    bounded_wall=sum(r["traced_bnb"]["bounded_k2_wall"] for r in good)
    k2_wall=sum(r["traced_bnb"]["total_k2_wall"] for r in good)
    oracle_wall=sum(r["oracle_bnb"]["wall"] for r in good)

    live_outliers=[
        r for r in good
        if r["live_v37182"]["bnb_over_memo"] is not None
        and r["live_v37182"]["bnb_over_memo"]>1.10
    ]
    out_total=sum(r["traced_bnb"]["wrapper_wall"] for r in live_outliers)
    out_seed=sum(r["traced_bnb"]["seed_wall"] for r in live_outliers)
    out_k2=sum(r["traced_bnb"]["total_k2_wall"] for r in live_outliers)

    seed_dominant=(
        out_total>0 and out_seed/out_total>=0.50
    )
    k2_dominant=(
        out_total>0 and out_k2/out_total>=0.75
    )

    elapsed=time.perf_counter()-t0
    overall=not failures

    print()
    print("# SUMMARY")
    print("exact parity         :",f"{len(good)}/{len(rows)}")
    print("failures             :",failures or "-")
    print(
        "all seed wall share :",
        f"{seed_wall/total_wall:.3f}"
        if total_wall else "n/a",
    )
    print(
        "all K2 wall share   :",
        f"{k2_wall/total_wall:.3f}"
        if total_wall else "n/a",
    )
    print(
        "bounded K2 wall share:",
        f"{bounded_wall/total_wall:.3f}"
        if total_wall else "n/a",
    )
    print("live outlier cases   :",[r["case"] for r in live_outliers])
    print(
        "outlier seed share  :",
        f"{out_seed/out_total:.3f}"
        if out_total else "n/a",
    )
    print(
        "outlier K2 share    :",
        f"{out_k2/out_total:.3f}"
        if out_total else "n/a",
    )
    print(
        "oracle/traced wall  :",
        f"{oracle_wall/total_wall:.3f}x"
        if total_wall else "n/a",
    )
    print("seed dominant outlier:",seed_dominant)
    print("K2 dominant outlier  :",k2_dominant)
    print("audit wall           :",f"{elapsed:.3f}s")

    if overall and seed_dominant:
        conclusion="DETERMINISTIC_SEED_K2_CONFIRMED_PRIMARY_OUTLIER_COST"
    elif overall and k2_dominant:
        conclusion="K2_COST_CONFIRMED_BUT_NOT_SEED_ONLY"
    elif overall:
        conclusion="BNB_OUTLIER_COST_NOT_EXPLAINED_BY_K2_SEED"
    else:
        conclusion="ATTRIBUTION_PARITY_FAIL"
    print("conclusion           :",conclusion)

    payload={
        "version":"v37.18.3.1",
        "source_v3715_json":args.v3715_json,
        "source_v37182_json":args.v37182_json,
        "rows":rows,
        "hard20_RL":hard,
        "failures":failures,
        "all_seed_wall_share":None if not total_wall else seed_wall/total_wall,
        "all_k2_wall_share":None if not total_wall else k2_wall/total_wall,
        "all_bounded_k2_wall_share":None if not total_wall else bounded_wall/total_wall,
        "live_outlier_cases":[r["case"] for r in live_outliers],
        "outlier_seed_wall_share":None if not out_total else out_seed/out_total,
        "outlier_k2_wall_share":None if not out_total else out_k2/out_total,
        "oracle_over_traced_wall":None if not total_wall else oracle_wall/total_wall,
        "seed_dominant_outlier":seed_dominant,
        "k2_dominant_outlier":k2_dominant,
        "elapsed":elapsed,
        "overall_pass":overall,
        "conclusion":conclusion,
        "scope_note":(
            "Oracle-seed arm is offline calibration only and cannot be used as "
            "runtime performance evidence. It isolates the cost of incumbent "
            "construction from the sound BnB proof kernel."
        ),
    }
    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(payload,indent=2,ensure_ascii=False),encoding="utf-8")
    print("JSON                 :",out)
    return 0 if overall else 2


if __name__=="__main__":
    raise SystemExit(main())

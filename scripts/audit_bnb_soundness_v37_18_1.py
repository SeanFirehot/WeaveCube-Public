#!/usr/bin/env python3
"""CubeLab v37.18.1 - soundness diagnosis for v37.18 case-5 failure.

Research-only. Production changes: NONE.

v37.18 missed one frozen exact optimum:
    ground truth H5+K2=7 = 12
    v37.18      H6+K2=7 = 13

This audit does two things.

A. Ground-truth witness path audit
   Replay each prefix of the frozen v37.15 optimal H witness and print:
   - exact supported H-depth set;
   - corner-star full-solution LB;
   - used_H + max(min_H_remaining, corner-star);
   - whether the known next move remains supported.

   Any LB >= ground-truth total on the optimal path would identify an unsafe
   static bound.

B. Disable v37.18's closed-state transposition completely
   and rerun all ten frozen cases + hard20.

The strongest current hypothesis is that `closed_threshold` is context-unsafe:
its key omits used_H / local remaining total budget.  A closure proved from a
more expensive cost-to-come can therefore be reused at a cheaper cost-to-come.

The no-closed-cache arm is intentionally conservative.  v37.16 showed that
full-state transposition reuse is weak, so losing this cache should cost little
if it is the bug.
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
from op_total_bnb_v37_18 import BranchBoundTotalValue
from op_total_value_v37_17 import OrientationTransitionCompiler
from audit_permutation_secondary_v37_2_1 import (
    HARD20,HARD20_PREFIX12,RL_MAP,map_word,
)


class NoClosedCacheBnB(BranchBoundTotalValue):
    """v37.18 search with ONLY the closed_threshold reuse removed."""

    def search(
        self,
        poses,
        q,
        depths,
        previous_face,
        parent_q,
        used_h,
        prefix,
    ):
        self.nodes+=1
        depths=tuple(sorted(set(int(d) for d in depths)))
        if not depths:
            return

        min_h=min(depths)
        state_lb=max(
            min_h,
            self.pose_lb(poses),
        )
        if int(used_h)+state_lb>=self.incumbent:
            self.state_bound_prunes+=1
            return

        # Exact true-DR boundary option.
        if 0 in depths:
            valid=(
                int(q)==int(ts.GOAL_Q)
                and parent_q!=ts.GOAL_Q
            )
            if not valid:
                raise RuntimeError(
                    "depth-set contains 0 outside true last-DR boundary"
                )

            self.terminals+=1
            state=PDCCState(poses)
            if not ts.is_dr(state):
                raise RuntimeError(
                    "q=GOAL no-cache BnB terminal failed full DR"
                )
            pk=ts.p_of(state)
            if pk is None:
                raise RuntimeError("DR terminal missing Phase2 coordinate")

            lb=self.terminal_lb(pk)
            if int(used_h)+lb<self.incumbent:
                cap=self.incumbent-int(used_h)-1
                bword,bdist=self.k2.query(
                    pk,previous_face,cap
                )
                if bword is not None and bdist is not None:
                    total=int(used_h)+int(bdist)
                    if total>=self.incumbent:
                        raise RuntimeError(
                            "bounded K2 returned non-improving candidate"
                        )
                    self.incumbent=total
                    self.best={
                        "total":total,
                        "h_tail":tuple(prefix),
                        "k2_word":tuple(bword),
                        "k2_exact":int(bdist),
                        "source":"NO_CLOSED_CACHE_IMPROVEMENT",
                    }
                    self.improvements+=1
            else:
                self.terminal_lb_skips+=1

        candidates=[]
        for move,mi,nq,child_depths in self.transitions.get(
            q,depths,previous_face
        ):
            nposes=pg.move_poses(poses,mi)
            child_lb=max(
                min(child_depths),
                self.pose_lb(nposes),
            )
            total_lb=int(used_h)+1+child_lb
            if total_lb>=self.incumbent:
                self.child_bound_prunes+=1
                continue
            candidates.append(
                (
                    total_lb,
                    move,
                    mi,
                    nq,
                    child_depths,
                    nposes,
                )
            )

        candidates.sort(
            key=lambda x:(x[0],MOVE_INDEX[x[1]])
        )

        for _lb,move,_mi,nq,child_depths,nposes in candidates:
            child_lb=max(
                min(child_depths),
                self.pose_lb(nposes),
            )
            if int(used_h)+1+child_lb>=self.incumbent:
                self.child_bound_prunes+=1
                continue

            self.search(
                nposes,
                nq,
                child_depths,
                move[0],
                q,
                int(used_h)+1,
                prefix+(move,),
            )

        # Deliberately NO closed-state transposition record.


def trace_ground_truth(
    base_row,
    tables,
    oo,
    transitions,
):
    """Audit static bounds/support along the v37.15 exact witness."""
    wit=base_row["unified"]["witness"]
    if wit is None:
        raise RuntimeError("ground-truth unified witness missing")

    hword=tuple(wit["h_tail"])
    kword=tuple(wit["k2_word"])
    total=int(wit["total"])

    scramble=tuple(base_row["scramble"])
    start=ts.engine.from_word(" ".join(scramble))
    state=start
    poses=start.poses
    q=ts.q_of(start)
    depths=(1,2,3,4,5,6)
    last=None
    parent_q=None

    rows=[]
    for used in range(len(hword)+1):
        plb=int(p1pose.lower_bound(poses))
        min_h=min(depths)
        total_lb=used+max(min_h,plb)

        row={
            "used_h":used,
            "prefix":list(hword[:used]),
            "q":int(q),
            "depths":list(depths),
            "min_h":int(min_h),
            "pose_lb":plb,
            "total_lb":int(total_lb),
            "gt_total":total,
            "margin":total-int(total_lb),
            "next_move":None,
            "next_supported":None,
        }

        if total_lb>total:
            row["STATIC_BOUND_VIOLATION"]=True

        if used==len(hword):
            state_pdcc=PDCCState(poses)
            row["full_dr"]=bool(ts.is_dr(state_pdcc))
            pk=ts.p_of(state_pdcc)
            row["terminal_pk"]=None if pk is None else int(pk)
            if pk is not None:
                c,u,s=ts.unpack_p(pk)
                p2lb=max(
                    int(tables[8][c]),
                    int(tables[9][u]),
                    int(tables[10][s]),
                )
                ds=max((int(x) for x in dstar.profile(pk)),default=0)
                row["p2lb"]=p2lb
                row["dstar_lb"]=ds
                row["combined_lb"]=max(p2lb,ds)
                row["gt_k2"]=len(kword)
            rows.append(row)
            break

        move=hword[used]
        supported=False
        child_depths=None
        child_q=None
        child_mi=None

        for m,mi,nq,cd in transitions.get(q,depths,last):
            if m==move:
                supported=True
                child_depths=cd
                child_q=nq
                child_mi=mi
                break

        row["next_move"]=move
        row["next_supported"]=supported
        row["child_depths"]=(
            None if child_depths is None else list(child_depths)
        )
        rows.append(row)

        if not supported:
            break

        poses=pg.move_poses(poses,child_mi)
        q=int(child_q)
        depths=tuple(child_depths)
        last=move[0]
        parent_q=row["q"]

    return {
        "scramble":list(scramble),
        "h_tail":list(hword),
        "k2_word":list(kword),
        "gt_total":total,
        "rows":rows,
        "all_moves_supported":(
            len(rows)==len(hword)+1
            and all(
                r.get("next_supported",True) is not False
                for r in rows
            )
        ),
        "static_bound_violation":any(
            r.get("STATIC_BOUND_VIOLATION",False)
            for r in rows
        ),
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--v3715-json",
        default="reports/v37/unified_total_budget_v37_15.json",
    )
    ap.add_argument(
        "--v3718-json",
        default="reports/v37/branch_bound_total_value_v37_18.json",
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
        default="reports/v37/bnb_soundness_v37_18_1.json",
    )
    args=ap.parse_args()

    base=json.loads(
        Path(args.v3715_json).read_text(encoding="utf-8")
    )
    old=json.loads(
        Path(args.v3718_json).read_text(encoding="utf-8")
    )
    base_rows=base.get("rows",[])
    old_rows=old.get("rows",[])
    if not base_rows or len(base_rows)!=len(old_rows):
        raise SystemExit("v37.15/v37.18 frozen corpus mismatch")

    failed=[
        int(r["case"])
        for r in old_rows
        if not r.get("parity",False)
    ]
    if not failed:
        raise SystemExit("v37.18 JSON has no parity failures to diagnose")

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

    depths=tuple(sorted(set(int(x) for x in args.depths)))
    trans=OrientationTransitionCompiler(oo)

    print("# CubeLab v37.18.1 - BnB SOUNDNESS DIAGNOSIS")
    print()
    print("failed v37.18 cases  :",failed)
    print("hypothesis           : closed_threshold context unsafe")
    print("diagnostic arm       : identical BnB with closed cache disabled")
    print("production changes   : NONE")
    print()

    traces={}
    for idx in failed:
        print(f"# WITNESS PATH CASE {idx}")
        tr=trace_ground_truth(
            base_rows[idx-1],
            tables,
            oo,
            trans,
        )
        traces[str(idx)]=tr
        print(
            "GT H/K2/total       :",
            len(tr["h_tail"]),
            len(tr["k2_word"]),
            tr["gt_total"],
        )
        for r in tr["rows"]:
            txt=(
                f"used={r['used_h']} "
                f"depths={r['depths']} "
                f"poseLB={r['pose_lb']} "
                f"totalLB={r['total_lb']} "
                f"margin={r['margin']}"
            )
            if r.get("next_move") is not None:
                txt+=(
                    f" next={r['next_move']} "
                    f"support={r['next_supported']} "
                    f"childD={r.get('child_depths')}"
                )
            else:
                txt+=(
                    f" DR={r.get('full_dr')} "
                    f"KLB={r.get('combined_lb')} "
                    f"gtK2={r.get('gt_k2')}"
                )
            print(txt)
        print("all support          :",tr["all_moves_supported"])
        print("static bound violation:",tr["static_bound_violation"])
        print()

    print("# NO-CLOSED-CACHE FULL FROZEN REPLAY")
    rows=[]
    failures=[]
    shared=OrientationTransitionCompiler(oo)
    t0=time.perf_counter()

    for i,b in enumerate(base_rows,1):
        scramble=tuple(b["scramble"])
        start=ts.engine.from_word(" ".join(scramble))
        q0=ts.q_of(start)
        gt=int(b["gt_best_total"])

        solver=NoClosedCacheBnB(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
            max_k2=args.max_k2,
            p1_pose_cache=args.p1_pose_cache,
            shared_transition_compiler=shared,
        )
        res=solver.solve(start,q0,None,depths)
        wit=res["witness"]
        replay=(
            wit is not None
            and start.apply_word(
                tuple(wit["h_tail"])+tuple(wit["k2_word"])
            ).is_solved()
        )
        parity=(
            wit is not None
            and int(wit["total"])==gt
            and replay
        )
        if not parity:
            failures.append(i)

        oldr=old_rows[i-1]
        row={
            "case":i,
            "gt":gt,
            "parity":parity,
            "replay":replay,
            "no_closed":res,
            "v3718":{
                "parity":bool(oldr.get("parity")),
                "wall":float(oldr["bnb"]["wall"]),
                "nodes":int(oldr["bnb"]["nodes"]),
                "k2_calls":int(oldr["bnb"]["k2_calls"]),
            },
        }
        rows.append(row)

        marker="RECOVERED" if (
            i in failed and parity
        ) else ""
        print(
            f"[{i:2d}/{len(base_rows)}] "
            f"GT={gt} "
            f"B={wit['total'] if wit else None} "
            f"nodes={oldr['bnb']['nodes']:,}->{res['nodes']:,} "
            f"K2={oldr['bnb']['k2_calls']}->{res['k2_calls']} "
            f"wall={oldr['bnb']['wall']:.3f}->{res['wall']:.3f}s "
            f"parity={'PASS' if parity else 'FAIL'} "
            f"{marker}"
        )

    # Hard20 regression.
    print()
    print("# HARD20 RL")
    hard0=base.get("hard20_RL_prefix12")
    hard=None
    if hard0 and hard0.get("witness") is not None:
        mapped_scramble=map_word(HARD20,RL_MAP)
        mapped_prefix=map_word(HARD20_PREFIX12,RL_MAP)
        start=ts.engine.from_word(
            " ".join(mapped_scramble)
        ).apply_word(mapped_prefix)
        q0=ts.q_of(start)
        gt=int(hard0["witness"]["total"])

        solver=NoClosedCacheBnB(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
            max_k2=args.max_k2,
            p1_pose_cache=args.p1_pose_cache,
            shared_transition_compiler=shared,
        )
        hard=solver.solve(
            start,q0,mapped_prefix[-1][0],depths
        )
        hw=hard["witness"]
        replay=(
            hw is not None
            and start.apply_word(
                tuple(hw["h_tail"])+tuple(hw["k2_word"])
            ).is_solved()
        )
        parity=(
            hw is not None
            and int(hw["total"])==gt
            and replay
        )
        hard["parity"]=parity
        hard["replay"]=replay
        if not parity:
            failures.append("hard20")
        print(
            f"GT={gt} B={hw['total'] if hw else None} "
            f"nodes={hard['nodes']} K2={hard['k2_calls']} "
            f"parity={'PASS' if parity else 'FAIL'}"
        )

    good=[r for r in rows if r["parity"]]
    old_good=[
        r for r in rows
        if r["parity"] and r["v3718"]["parity"]
    ]

    old_wall=sum(r["v3718"]["wall"] for r in old_good)
    new_wall_same=sum(
        r["no_closed"]["wall"] for r in old_good
    )
    all_wall=sum(
        r["no_closed"]["wall"] for r in good
    )

    recovered=all(
        rows[i-1]["parity"] for i in failed
    )
    static_clean=all(
        not traces[str(i)]["static_bound_violation"]
        and traces[str(i)]["all_moves_supported"]
        for i in failed
    )

    elapsed=time.perf_counter()-t0

    if recovered and static_clean and not failures:
        conclusion="CLOSED_THRESHOLD_CACHE_CONFIRMED_UNSAFE_REMOVE_IT"
    elif recovered and not failures:
        conclusion="NO_CLOSED_CACHE_RESTORES_PARITY_STATIC_TRACE_NEEDS_REVIEW"
    else:
        conclusion="PARITY_FAILURE_PERSISTS_WITHOUT_CLOSED_CACHE"

    print()
    print("# SUMMARY")
    print("static GT path clean :",static_clean)
    print("failed cases recovered:",recovered)
    print("no-cache parity      :",f"{len(good)}/{len(rows)}")
    print("failures             :",failures or "-")
    if old_wall:
        print(
            "no-cache/old wall on old-PASS cases:",
            f"{new_wall_same/old_wall:.3f}x",
        )
    print("no-cache total wall  :",f"{all_wall:.3f}s")
    print("audit wall           :",f"{elapsed:.3f}s")
    print("conclusion           :",conclusion)

    payload={
        "version":"v37.18.1",
        "source_v3715_json":args.v3715_json,
        "source_v3718_json":args.v3718_json,
        "failed_v3718_cases":failed,
        "ground_truth_traces":traces,
        "rows":rows,
        "hard20_RL":hard,
        "static_gt_path_clean":static_clean,
        "failed_cases_recovered":recovered,
        "no_closed_failures":failures,
        "no_closed_total_wall":all_wall,
        "same_case_wall_ratio_no_closed_over_v3718":(
            None if not old_wall else new_wall_same/old_wall
        ),
        "elapsed":elapsed,
        "conclusion":conclusion,
        "scope_note":(
            "Diagnostic exact arm removes only v37.18 closed_threshold reuse. "
            "All admissible static bounds, branch ordering, bounded K2, "
            "Orientation transitions, and seed policy are unchanged."
        ),
    }

    out=Path(args.output)
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(
        json.dumps(payload,indent=2,ensure_ascii=False),
        encoding="utf-8",
    )
    print("JSON                 :",out)


if __name__=="__main__":
    main()

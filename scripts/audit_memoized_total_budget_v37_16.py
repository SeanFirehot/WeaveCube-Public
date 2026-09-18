#!/usr/bin/env python3
"""CubeLab v37.16 - memoized unified total-budget O/P predicate.

Research-only. Production changes: NONE.

v37.15 proved exact variable-H semantics but iterative total budgets revisit many
of the same full O/P states.

v37.16 memoizes the monotone existence predicate:

    ExistsTotalWithin(state, depth-set, previous-face, parent-goal, B)

For a budget-independent state key K:

SUCCESS cache
    If a witness of local total cost C is known, every future query B>=C is YES.

FAIL-UP-TO cache
    If the complete subtree was proven NO for budget B, every future query
    B'<=B is also NO.

Orientation transition cache
    (q, depth-set, previous-face) -> exact supported move / child-depth-set list
    independent of total budget and full permutation.

K2 threshold cache
    Per (pk, previous-face):
      exact success distance is reusable for every cap >= distance;
      a failed cap closes every smaller cap.

The predicate remains exact.  It returns a witness, not merely a Boolean.
Ground truth and v37.15 baseline metrics are reused from frozen JSON.
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
import search_pdcc_guided as pg
import search_twist_skeleton as ts
from cubelab.pdcc import PDCCState

from p1_orientation_domains_v37_0 import (
    D6MoveDomainOracle,
    MOVE_INDEX,
)
from audit_permutation_secondary_v37_2_1 import (
    HARD20,HARD20_PREFIX12,RL_MAP,map_word,
)


class MonotoneK2:
    def __init__(self,tables,order,probe):
        (
            _co,_eo,_sl,_cos,_eos,
            self.cp,self.up,self.sp,
            self.cpd,self.upd,self.spd,
        )=tables
        self.order=order
        self.probe=int(probe)

        # (pk,prev) -> (exact_dist, word)
        self.success={}
        # (pk,prev) -> highest cap proven impossible
        self.fail_upto={}

        self.calls=0
        self.nodes=0
        self.success_hits=0
        self.fail_hits=0
        self.wall=0.0

    def query(self,pk,prev,cap):
        cap=int(cap)
        if cap<0:
            return None,None

        key=(int(pk),prev)

        hit=self.success.get(key)
        if hit is not None:
            dist,word=hit
            if int(dist)<=cap:
                self.success_hits+=1
                return tuple(word),int(dist)

        failed=self.fail_upto.get(key,-1)
        if cap<=failed:
            self.fail_hits+=1
            return None,None

        stats={}
        t0=time.perf_counter()
        word,dist=p2.shortest_b_compatible(
            int(pk),prev,cap,
            self.cp,self.up,self.sp,
            self.cpd,self.upd,self.spd,
            order=self.order,
            stats=stats,
            auto_probe_nodes=self.probe,
        )
        self.wall+=time.perf_counter()-t0
        self.calls+=1
        self.nodes+=int(stats.get("nodes",0))

        if word is None or dist is None:
            self.fail_upto[key]=max(failed,cap)
            return None,None

        val=(int(dist),tuple(word))
        old=self.success.get(key)
        if old is None or val[0]<old[0]:
            self.success[key]=val
        return tuple(word),int(dist)


class MemoTotalBudgetSearch:
    def __init__(self,tables,oo,*,p2_order,p2_probe):
        self.tables=tables
        self.oo=oo
        (
            self.co,self.eo,self.sl,self.cos,self.eos,
            _cp,_up,_sp,self.cpd,self.upd,self.spd,
        )=tables

        self.k2=MonotoneK2(
            tables,p2_order,p2_probe
        )

        # key=(poses,q,depths,last,parent_was_goal)
        # -> (cost, h_tail, k2_word, k2_exact)
        self.success_cache={}

        # same key -> greatest local budget fully proved impossible
        self.fail_upto={}

        # (q,positive_depths,last) ->
        # tuple[(move, mi, nq, child_depths), ...]
        self.transition_cache={}

        self.reset_metrics()

    def reset_metrics(self):
        self.nodes=0
        self.terminals=0
        self.lb_skips=0
        self.success_cache_hits=0
        self.fail_cache_hits=0
        self.transition_hits=0
        self.transition_builds=0
        self.domain_queries=0

    def combined_terminal_lb(self,pk):
        c,u,s=ts.unpack_p(pk)
        p2lb=max(
            int(self.cpd[c]),
            int(self.upd[u]),
            int(self.spd[s]),
        )
        ds=max((int(x) for x in dstar.profile(pk)),default=0)
        return max(p2lb,ds)

    def transitions(self,q,depths,last):
        pos=tuple(d for d in depths if d>0)
        key=(int(q),pos,last)
        hit=self.transition_cache.get(key)
        if hit is not None:
            self.transition_hits+=1
            return hit

        by_d={}
        for d in pos:
            self.domain_queries+=1
            by_d[d]=self.oo.allowed_domain(
                int(q),int(d),last
            )

        out=[]
        for move in ts.MOVE_ORDER:
            child_depths=tuple(sorted(
                d-1
                for d,dom in by_d.items()
                if dom.contains(move)
            ))
            if not child_depths:
                continue
            mi=MOVE_INDEX[move]
            nq=ts.q_move(
                int(q),mi,self.co,self.eo,self.sl
            )
            out.append((move,mi,nq,child_depths))

        val=tuple(out)
        self.transition_cache[key]=val
        self.transition_builds+=1
        return val

    def _state_key(self,poses,q,depths,last,parent_q):
        return (
            tuple(poses),
            int(q),
            tuple(depths),
            last,
            bool(parent_q==ts.GOAL_Q),
        )

    def _record_success(self,key,witness):
        cost=int(witness["total"])
        old=self.success_cache.get(key)
        if old is None or cost<int(old["total"]):
            self.success_cache[key]={
                "total":cost,
                "h_tail":tuple(witness["h_tail"]),
                "k2_word":tuple(witness["k2_word"]),
                "k2_exact":int(witness["k2_exact"]),
            }

    def find(
        self,
        poses,
        q,
        depths,
        last_face,
        parent_q,
        budget,
    ):
        """Exact witness with LOCAL H+K2 cost <= budget, else None."""
        self.nodes+=1
        budget=int(budget)
        depths=tuple(sorted(set(int(d) for d in depths)))

        if budget<0 or not depths:
            return None
        if min(depths)>budget:
            return None

        key=self._state_key(
            poses,q,depths,last_face,parent_q
        )

        success=self.success_cache.get(key)
        if success is not None and int(success["total"])<=budget:
            self.success_cache_hits+=1
            return dict(success)

        failed=self.fail_upto.get(key,-1)
        if budget<=failed:
            self.fail_cache_hits+=1
            return None

        # Boundary candidate.
        if 0 in depths:
            if (
                int(q)!=int(ts.GOAL_Q)
                or parent_q==ts.GOAL_Q
            ):
                raise RuntimeError(
                    "depth-set contains 0 outside true last-DR boundary"
                )

            self.terminals+=1
            state=PDCCState(poses)
            if not ts.is_dr(state):
                raise RuntimeError(
                    "q=GOAL total-budget terminal failed full DR"
                )
            pk=ts.p_of(state)
            if pk is None:
                raise RuntimeError("DR terminal missing Phase2 coordinate")

            if self.combined_terminal_lb(pk)<=budget:
                bword,bdist=self.k2.query(
                    pk,last_face,budget
                )
                if bword is not None and bdist is not None:
                    witness={
                        "h_tail":(),
                        "k2_word":tuple(bword),
                        "k2_exact":int(bdist),
                        "total":int(bdist),
                    }
                    self._record_success(key,witness)
                    return witness
            else:
                self.lb_skips+=1

        if budget==0:
            self.fail_upto[key]=max(failed,budget)
            return None

        for move,mi,nq,child_depths in self.transitions(
            q,depths,last_face
        ):
            if min(child_depths)>budget-1:
                continue

            nposes=pg.move_poses(poses,mi)
            child=self.find(
                nposes,
                nq,
                child_depths,
                move[0],
                q,
                budget-1,
            )
            if child is None:
                continue

            total=1+int(child["total"])
            if total>budget:
                raise RuntimeError("child witness exceeded parent budget")

            witness={
                "h_tail":(move,)+tuple(child["h_tail"]),
                "k2_word":tuple(child["k2_word"]),
                "k2_exact":int(child["k2_exact"]),
                "total":total,
            }
            self._record_success(key,witness)
            return witness

        # Complete NO proof at this local budget.
        self.fail_upto[key]=max(failed,budget)
        return None

    def exact_min_total(
        self,start,q0,previous_face,depths,max_total
    ):
        depths=tuple(sorted(set(int(d) for d in depths)))
        p1lb=int(ts.phase1_lb(
            int(q0),self.cos,self.eos
        ))
        low=max(min(depths),p1lb)

        attempts=[]
        first=None

        for B in range(low,int(max_total)+1):
            before={
                "nodes":self.nodes,
                "terminals":self.terminals,
                "success_hits":self.success_cache_hits,
                "fail_hits":self.fail_cache_hits,
                "transition_hits":self.transition_hits,
                "transition_builds":self.transition_builds,
                "domain_queries":self.domain_queries,
                "k2_calls":self.k2.calls,
                "k2_success_hits":self.k2.success_hits,
                "k2_fail_hits":self.k2.fail_hits,
            }
            t0=time.perf_counter()
            hit=self.find(
                start.poses,
                int(q0),
                depths,
                previous_face,
                None,
                B,
            )
            wall=time.perf_counter()-t0

            after={
                "nodes":self.nodes,
                "terminals":self.terminals,
                "success_hits":self.success_cache_hits,
                "fail_hits":self.fail_cache_hits,
                "transition_hits":self.transition_hits,
                "transition_builds":self.transition_builds,
                "domain_queries":self.domain_queries,
                "k2_calls":self.k2.calls,
                "k2_success_hits":self.k2.success_hits,
                "k2_fail_hits":self.k2.fail_hits,
            }
            delta={
                k:after[k]-before[k] for k in before
            }
            attempts.append({
                "budget":B,
                "found":hit is not None,
                "wall":wall,
                **delta,
            })

            if hit is not None:
                if int(hit["total"])!=B:
                    raise RuntimeError(
                        f"first budget {B} but witness cost {hit['total']}"
                    )
                first={
                    **hit,
                    "budget":B,
                }
                break

        return {
            "lower_budget":low,
            "attempts":attempts,
            "witness":first,
            "total_nodes":self.nodes,
            "total_terminals":self.terminals,
            "success_cache_hits":self.success_cache_hits,
            "fail_cache_hits":self.fail_cache_hits,
            "transition_hits":self.transition_hits,
            "transition_builds":self.transition_builds,
            "domain_queries":self.domain_queries,
            "success_cache_size":len(self.success_cache),
            "fail_cache_size":len(self.fail_upto),
            "transition_cache_size":len(self.transition_cache),
            "k2_calls":self.k2.calls,
            "k2_nodes":self.k2.nodes,
            "k2_success_hits":self.k2.success_hits,
            "k2_fail_hits":self.k2.fail_hits,
            "k2_wall":self.k2.wall,
            "total_wall":sum(x["wall"] for x in attempts),
        }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument(
        "--v3715-json",
        default="reports/v37/unified_total_budget_v37_15.json",
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
    ap.add_argument("--p2-order",default="auto")
    ap.add_argument("--p2-auto-probe-nodes",type=int,default=128)
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
        default="reports/v37/memoized_total_budget_v37_16.json",
    )
    args=ap.parse_args()

    base=json.loads(Path(args.v3715_json).read_text(encoding="utf-8"))
    base_rows=base.get("rows",[])
    if not base_rows:
        raise SystemExit("v37.15 JSON has no rows")

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
    if not depths:
        raise SystemExit("no valid depths")

    print("# CubeLab v37.16 - MEMOIZED UNIFIED TOTAL-BUDGET")
    print()
    print("baseline             :",args.v3715_json)
    print("cases                :",len(base_rows))
    print("H depth domain       :",depths)
    print("memo                 : monotone success/fail + transition + K2")
    print("production changes   : NONE")
    print()

    rows=[]
    failures=[]
    t0=time.perf_counter()

    for i,b in enumerate(base_rows,1):
        scramble=tuple(b["scramble"])
        start=ts.engine.from_word(" ".join(scramble))
        q0=ts.q_of(start)
        gt=int(b["gt_best_total"])

        solver=MemoTotalBudgetSearch(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
        )
        res=solver.exact_min_total(
            start,q0,None,depths,gt
        )
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

        old=b["unified"]
        old_nodes=int(old["total_nodes"])
        old_wall=float(old["total_wall"])
        old_q=int(old["total_threshold_k2_calls"])

        row={
            "case":i,
            "scramble":list(scramble),
            "gt_best_total":gt,
            "gt_best_h":int(b["gt_best_h"]),
            "gt_best_k2":int(b["gt_best_k2"]),
            "parity":parity,
            "replay":replay,
            "memoized":res,
            "v3715":{
                "nodes":old_nodes,
                "wall":old_wall,
                "k2_calls":old_q,
            },
            "node_ratio":(
                res["total_nodes"]/old_nodes if old_nodes else None
            ),
            "wall_ratio":(
                res["total_wall"]/old_wall if old_wall else None
            ),
            "k2_ratio":(
                res["k2_calls"]/old_q if old_q else None
            ),
        }
        rows.append(row)

        print(
            f"[{i:2d}/{len(base_rows)}] "
            f"GT={b['gt_best_h']}+{b['gt_best_k2']}={gt} "
            f"budgets={len(res['attempts'])} "
            f"nodes={old_nodes:,}->{res['total_nodes']:,} "
            f"memoHit={res['success_cache_hits']+res['fail_cache_hits']:,} "
            f"transHit={res['transition_hits']:,} "
            f"K2={old_q}->{res['k2_calls']} "
            f"wall={old_wall:.3f}->{res['total_wall']:.3f}s "
            f"parity={'PASS' if parity else 'FAIL'}"
        )

    print()
    print("# HARD20 RL PREFIX12")
    hard_base=base.get("hard20_RL_prefix12")
    hard=None
    if hard_base and hard_base.get("witness") is not None:
        mapped_scramble=map_word(HARD20,RL_MAP)
        mapped_prefix=map_word(HARD20_PREFIX12,RL_MAP)
        start=ts.engine.from_word(
            " ".join(mapped_scramble)
        ).apply_word(mapped_prefix)
        q0=ts.q_of(start)
        gt=int(hard_base["witness"]["total"])

        solver=MemoTotalBudgetSearch(
            tables,oo,
            p2_order=args.p2_order,
            p2_probe=args.p2_auto_probe_nodes,
        )
        hard=solver.exact_min_total(
            start,q0,mapped_prefix[-1][0],
            depths,gt
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
        print(
            f"GT={gt} "
            f"U={hw['h_tail'] if hw else None}+"
            f"{hw['k2_exact'] if hw else None} "
            f"nodes={hard['total_nodes']:,} "
            f"K2q={hard['k2_calls']} "
            f"parity={'PASS' if parity else 'FAIL'}"
        )
        if not parity:
            failures.append("hard20")

    good=[r for r in rows if r["parity"]]

    old_nodes=sum(r["v3715"]["nodes"] for r in good)
    new_nodes=sum(r["memoized"]["total_nodes"] for r in good)
    old_wall=sum(r["v3715"]["wall"] for r in good)
    new_wall=sum(r["memoized"]["total_wall"] for r in good)
    old_q=sum(r["v3715"]["k2_calls"] for r in good)
    new_q=sum(r["memoized"]["k2_calls"] for r in good)

    success_hits=sum(
        r["memoized"]["success_cache_hits"] for r in good
    )
    fail_hits=sum(
        r["memoized"]["fail_cache_hits"] for r in good
    )
    trans_hits=sum(
        r["memoized"]["transition_hits"] for r in good
    )
    trans_builds=sum(
        r["memoized"]["transition_builds"] for r in good
    )
    k2_success_hits=sum(
        r["memoized"]["k2_success_hits"] for r in good
    )
    k2_fail_hits=sum(
        r["memoized"]["k2_fail_hits"] for r in good
    )

    elapsed=time.perf_counter()-t0
    overall=(not failures)

    print()
    print("# SUMMARY")
    print("exact parity         :",f"{len(good)}/{len(rows)}")
    print("failures             :",failures or "-")
    print(
        "memo/unified nodes  :",
        f"{new_nodes/old_nodes:.3f}x" if old_nodes else "n/a",
    )
    print(
        "memo/unified wall   :",
        f"{new_wall/old_wall:.3f}x" if old_wall else "n/a",
    )
    print(
        "memo/unified K2 q   :",
        f"{new_q/old_q:.3f}x" if old_q else "n/a",
    )
    print("state success hits   :",f"{success_hits:,}")
    print("state fail hits      :",f"{fail_hits:,}")
    print("transition hit/build :",f"{trans_hits:,}/{trans_builds:,}")
    print("K2 success/fail hits :",f"{k2_success_hits:,}/{k2_fail_hits:,}")
    print("audit wall           :",f"{elapsed:.3f}s")

    if overall and old_wall and new_wall<old_wall:
        conclusion="MEMOIZED_TOTAL_BUDGET_EXACT_AND_FASTER"
    elif overall:
        conclusion="MEMOIZED_TOTAL_BUDGET_EXACT_NO_SPEED_GAIN"
    else:
        conclusion="MEMOIZED_TOTAL_BUDGET_PARITY_FAIL"
    print("conclusion           :",conclusion)

    payload={
        "version":"v37.16",
        "source_v3715_json":args.v3715_json,
        "depths":depths,
        "rows":rows,
        "hard20_RL_prefix12":hard,
        "failures":failures,
        "weighted_memo_over_unified_nodes":(
            None if not old_nodes else new_nodes/old_nodes
        ),
        "weighted_memo_over_unified_wall":(
            None if not old_wall else new_wall/old_wall
        ),
        "weighted_memo_over_unified_k2_calls":(
            None if not old_q else new_q/old_q
        ),
        "state_success_cache_hits":success_hits,
        "state_fail_cache_hits":fail_hits,
        "transition_cache_hits":trans_hits,
        "transition_cache_builds":trans_builds,
        "k2_success_cache_hits":k2_success_hits,
        "k2_fail_cache_hits":k2_fail_hits,
        "elapsed":elapsed,
        "overall_pass":overall,
        "conclusion":conclusion,
        "scope_note":(
            "Exact memoization over hashable full pose tuples and Orientation "
            "depth-set states. Still limited to d6 horizon; no production hook."
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

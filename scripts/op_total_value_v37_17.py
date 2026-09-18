#!/usr/bin/env python3
"""CubeLab v37.17 - direct exact O/P total-value function.

Research-only. Production changes: NONE.

Bellman semantics inside the compiled d6 horizon:

    V(S, D) =
        min(
            exact_K2(S)                         if H-depth 0 is supported,
            1 + V(move(S), child_depth_set)    for supported H moves
        )

where:
    S = full cube state + orientation quotient + previous-face boundary context
    D = exact H lengths still supported by the d6 policy

Unlike v37.15/v37.16 there is:
- no iterative total-budget loop;
- no success/fail budget predicate cache.

Each exact O/P state gets one value:
    minimum remaining H + K2
plus a witness.

The highly successful budget-independent Orientation transition cache from
v37.16 is retained.
"""
from __future__ import annotations

import time

import pdcc_double_star as dstar
import pdcc_phase2 as p2
import search_pdcc_guided as pg
import search_twist_skeleton as ts
from cubelab.pdcc import PDCCState

from p1_orientation_domains_v37_0 import MOVE_INDEX


INF = 10**9


class ExactK2Value:
    def __init__(self,tables,*,order,probe,max_depth):
        (
            _co,_eo,_sl,_cos,_eos,
            self.cp,self.up,self.sp,
            self.cpd,self.upd,self.spd,
        )=tables
        self.order=order
        self.probe=int(probe)
        self.max_depth=int(max_depth)

        self.cache={}
        self.calls=0
        self.cache_hits=0
        self.nodes=0
        self.wall=0.0

    def value(self,pk,previous_face):
        key=(int(pk),previous_face)
        hit=self.cache.get(key)
        if hit is not None:
            self.cache_hits+=1
            return hit

        stats={}
        t0=time.perf_counter()
        word,dist=p2.shortest_b_compatible(
            int(pk),
            previous_face,
            self.max_depth,
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
            val=None
        else:
            val=(int(dist),tuple(word))

        self.cache[key]=val
        return val


class OrientationTransitionCompiler:
    """Budget-independent variable-H Orientation policy transitions."""

    def __init__(self,orientation_oracle):
        self.oo=orientation_oracle
        self.cache={}
        self.hits=0
        self.builds=0
        self.domain_queries=0

    def get(self,q,depths,previous_face):
        pos=tuple(sorted(
            d for d in set(int(x) for x in depths)
            if d>0
        ))
        key=(int(q),pos,previous_face)
        hit=self.cache.get(key)
        if hit is not None:
            self.hits+=1
            return hit

        by_d={}
        for d in pos:
            self.domain_queries+=1
            by_d[d]=self.oo.allowed_domain(
                int(q),int(d),previous_face
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
                int(q),mi,
                self.oo.co,self.oo.eo,self.oo.sl,
            )
            out.append(
                (move,mi,int(nq),child_depths)
            )

        val=tuple(out)
        self.cache[key]=val
        self.builds+=1
        return val


class DirectTotalValue:
    """Exact memoized Bellman value for min(H_remaining + K2)."""

    def __init__(
        self,
        tables,
        orientation_oracle,
        *,
        p2_order,
        p2_probe,
        max_k2,
        shared_transition_compiler=None,
    ):
        self.tables=tables
        self.oo=orientation_oracle
        self.transitions=(
            shared_transition_compiler
            if shared_transition_compiler is not None
            else OrientationTransitionCompiler(orientation_oracle)
        )
        self.k2=ExactK2Value(
            tables,
            order=p2_order,
            probe=p2_probe,
            max_depth=max_k2,
        )

        self.memo={}
        self.nodes=0
        self.memo_hits=0
        self.terminals=0
        self.infeasible_states=0
        self.child_lb_skips=0

    def _key(self,poses,q,depths,last,parent_q):
        return (
            tuple(poses),
            int(q),
            tuple(sorted(set(int(d) for d in depths))),
            last,
            bool(parent_q==ts.GOAL_Q),
        )

    @staticmethod
    def _rank(w):
        if w is None:
            return (INF,INF,INF,())
        return (
            int(w["total"]),
            len(w["h_tail"]),
            int(w["k2_exact"]),
            tuple(w["h_tail"])+tuple(w["k2_word"]),
        )

    def value(
        self,
        poses,
        q,
        depths,
        previous_face,
        parent_q,
    ):
        self.nodes+=1
        depths=tuple(sorted(set(int(d) for d in depths)))
        if not depths:
            self.infeasible_states+=1
            return None

        key=self._key(
            poses,q,depths,previous_face,parent_q
        )
        if key in self.memo:
            self.memo_hits+=1
            return self.memo[key]

        best=None

        # Exact H boundary option.
        if 0 in depths:
            valid_boundary=(
                int(q)==int(ts.GOAL_Q)
                and parent_q!=ts.GOAL_Q
            )
            if valid_boundary:
                self.terminals+=1
                state=PDCCState(poses)
                if not ts.is_dr(state):
                    raise RuntimeError(
                        "Bellman depth-0 state is q=GOAL but not full DR"
                    )
                pk=ts.p_of(state)
                if pk is None:
                    raise RuntimeError(
                        "Bellman DR boundary has no Phase2 coordinate"
                    )

                kval=self.k2.value(
                    pk,previous_face
                )
                if kval is not None:
                    dist,word=kval
                    best={
                        "total":int(dist),
                        "h_tail":(),
                        "k2_word":tuple(word),
                        "k2_exact":int(dist),
                    }

        # Positive H options.
        if any(d>0 for d in depths):
            for move,mi,nq,child_depths in self.transitions.get(
                q,depths,previous_face
            ):
                # Very cheap exact lower bound: this child still needs at least
                # min(child_depths) H moves if 0 is not already supported.
                local_lb=1+min(child_depths)
                if (
                    best is not None
                    and local_lb>=int(best["total"])
                ):
                    self.child_lb_skips+=1
                    continue

                nposes=pg.move_poses(poses,mi)
                child=self.value(
                    nposes,
                    nq,
                    child_depths,
                    move[0],
                    q,
                )
                if child is None:
                    continue

                cand={
                    "total":1+int(child["total"]),
                    "h_tail":(move,)+tuple(child["h_tail"]),
                    "k2_word":tuple(child["k2_word"]),
                    "k2_exact":int(child["k2_exact"]),
                }
                if self._rank(cand)<self._rank(best):
                    best=cand

        if best is None:
            self.infeasible_states+=1

        self.memo[key]=best
        return best

    def solve(self,start,q0,previous_face,depths):
        t0=time.perf_counter()
        witness=self.value(
            start.poses,
            int(q0),
            tuple(depths),
            previous_face,
            None,
        )
        wall=time.perf_counter()-t0

        return {
            "witness":witness,
            "nodes":self.nodes,
            "memo_hits":self.memo_hits,
            "memo_states":len(self.memo),
            "terminals":self.terminals,
            "infeasible_states":self.infeasible_states,
            "child_lb_skips":self.child_lb_skips,
            "transition_hits":self.transitions.hits,
            "transition_builds":self.transitions.builds,
            "domain_queries":self.transitions.domain_queries,
            "transition_cache_size":len(self.transitions.cache),
            "k2_calls":self.k2.calls,
            "k2_cache_hits":self.k2.cache_hits,
            "k2_nodes":self.k2.nodes,
            "k2_wall":self.k2.wall,
            "wall":wall,
        }

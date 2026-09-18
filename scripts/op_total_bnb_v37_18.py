#!/usr/bin/env python3
"""CubeLab v37.18 - branch-and-bound exact total value.

Research-only. Production changes: NONE.

v37.17 direct Bellman evaluation was exact but evaluated exact K2 at nearly
every DR terminal. v37.18 keeps the same total-value objective while using an
incumbent and admissible lower bounds.

Algorithm
=========
1. Seed an incumbent with a deterministic exact Orientation completion at the
   shortest feasible H depth, followed by one exact K2 solve.
2. Search the full variable-H exact Orientation policy.
3. At every O/P state:
       used_H + max(min_supported_H_remaining, corner_star_full_solution_LB)
   is an admissible total lower bound.
   If it cannot beat incumbent, prune.
4. At a true DR boundary:
       used_H + max(P2LB, DStarLB)
   is an admissible total lower bound.
   Only if it can beat incumbent do we ask:
       exact boundary-compatible K2 <= incumbent - used_H - 1 ?
5. Any improving witness lowers the global incumbent immediately.
6. Exhaustion proves the incumbent is the exact min(H+K2).

No iterative total-budget loop and no full exact-K2 value at every terminal.
"""
from __future__ import annotations

import time

import pdcc_double_star as dstar
import pdcc_p1_pose as p1pose
import pdcc_phase2 as p2
import search_pdcc_guided as pg
import search_twist_skeleton as ts
from cubelab.pdcc import PDCCState

from p1_orientation_domains_v37_0 import MOVE_INDEX
from p1_orientation_synthesis_v37_1 import synthesize_orientation_suffix
from op_total_value_v37_17 import OrientationTransitionCompiler


class MonotoneBoundedK2:
    """Exact threshold K2 oracle with monotone success/failure reuse."""

    def __init__(self,tables,*,order,probe,max_depth):
        (
            _co,_eo,_sl,_cos,_eos,
            self.cp,self.up,self.sp,
            self.cpd,self.upd,self.spd,
        )=tables
        self.order=order
        self.probe=int(probe)
        self.max_depth=int(max_depth)

        # (pk,prev) -> best exact success (dist,word)
        self.success={}
        # (pk,prev) -> largest cap exactly proved impossible
        self.fail_upto={}

        self.calls=0
        self.nodes=0
        self.wall=0.0
        self.success_hits=0
        self.fail_hits=0

    def query(self,pk,prev,cap):
        cap=min(int(cap),self.max_depth)
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


class BranchBoundTotalValue:
    def __init__(
        self,
        tables,
        orientation_oracle,
        *,
        p2_order,
        p2_probe,
        max_k2,
        p1_pose_cache,
        shared_transition_compiler=None,
    ):
        self.tables=tables
        self.oo=orientation_oracle
        (
            self.co,self.eo,self.sl,self.cos,self.eos,
            _cp,_up,_sp,self.cpd,self.upd,self.spd,
        )=tables

        self.transitions=(
            shared_transition_compiler
            if shared_transition_compiler is not None
            else OrientationTransitionCompiler(orientation_oracle)
        )

        self.k2=MonotoneBoundedK2(
            tables,
            order=p2_order,
            probe=p2_probe,
            max_depth=max_k2,
        )

        p1pose.configure(p1_pose_cache)
        p1pose.preload()

        self.incumbent=10**9
        self.best=None

        self.nodes=0
        self.terminals=0
        self.pose_bound_calls=0
        self.state_bound_prunes=0
        self.child_bound_prunes=0
        self.terminal_lb_skips=0
        self.improvements=0
        self.seed_k2_calls=0
        self.pose_lb_cache={}
        self.closed_threshold={}
        self.closed_hits=0

    def pose_lb(self,poses):
        key=tuple(poses)
        hit=self.pose_lb_cache.get(key)
        if hit is not None:
            return hit
        self.pose_bound_calls+=1
        val=int(p1pose.lower_bound(poses))
        self.pose_lb_cache[key]=val
        return val

    def terminal_lb(self,pk):
        c,u,s=ts.unpack_p(pk)
        p2lb=max(
            int(self.cpd[c]),int(self.upd[u]),int(self.spd[s])
        )
        ds=max((int(x) for x in dstar.profile(pk)),default=0)
        return max(p2lb,ds)

    def seed_incumbent(self,start,q0,previous_face,depths):
        """Use the shortest exact-H deterministic completion as an upper bound."""
        for h in sorted(set(int(d) for d in depths if int(d)>0)):
            syn=synthesize_orientation_suffix(
                int(q0),
                h,
                previous_face,
                self.oo,
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
                raise RuntimeError("seed DR terminal missing Phase2 coordinate")

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
            self.k2.wall+=time.perf_counter()-t0
            self.k2.calls+=1
            self.seed_k2_calls+=1
            self.k2.nodes+=int(stats.get("nodes",0))

            if bword is None or bdist is None:
                continue

            key=(int(pk),hword[-1][0] if hword else previous_face)
            self.k2.success[key]=(int(bdist),tuple(bword))

            self.incumbent=h+int(bdist)
            self.best={
                "total":self.incumbent,
                "h_tail":hword,
                "k2_word":tuple(bword),
                "k2_exact":int(bdist),
                "source":"SHORTEST_FEASIBLE_H_SEED",
            }
            return

        raise RuntimeError("could not construct any initial H+K2 incumbent")

    def _closed_key(self,poses,q,depths,last,parent_q):
        return (
            tuple(poses),
            int(q),
            tuple(depths),
            last,
            bool(parent_q==ts.GOAL_Q),
        )

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

        # If this exact state was fully closed against an equal-or-looser
        # incumbent earlier, reuse the proof.
        key=self._closed_key(
            poses,q,depths,previous_face,parent_q
        )
        closed_at=self.closed_threshold.get(key)
        if closed_at is not None and self.incumbent<=closed_at:
            self.closed_hits+=1
            return

        min_h=min(depths)
        state_lb=max(
            min_h,
            self.pose_lb(poses),
        )
        if int(used_h)+state_lb>=self.incumbent:
            self.state_bound_prunes+=1
            self.closed_threshold[key]=max(
                self.closed_threshold.get(key,-1),
                self.incumbent,
            )
            return

        start_incumbent=self.incumbent

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
                    "q=GOAL branch-and-bound terminal failed full DR"
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
                        "source":"BRANCH_BOUND_IMPROVEMENT",
                    }
                    self.improvements+=1
            else:
                self.terminal_lb_skips+=1

        # Positive H children. Build full-state child LB once and explore best
        # bound first.
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
            # Incumbent may have tightened since candidate construction.
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

        # If exhaustive work found no solution below the incumbent that existed
        # on entry, this state is closed for that threshold.
        self.closed_threshold[key]=max(
            self.closed_threshold.get(key,-1),
            start_incumbent,
        )

    def solve(self,start,q0,previous_face,depths):
        t0=time.perf_counter()

        self.seed_incumbent(
            start,q0,previous_face,depths
        )
        seed_total=self.incumbent
        seed_best=dict(self.best)

        self.search(
            start.poses,
            int(q0),
            tuple(depths),
            previous_face,
            None,
            0,
            (),
        )
        wall=time.perf_counter()-t0

        return {
            "seed_total":seed_total,
            "seed_witness":seed_best,
            "witness":self.best,
            "nodes":self.nodes,
            "terminals":self.terminals,
            "pose_bound_calls":self.pose_bound_calls,
            "pose_lb_cache_size":len(self.pose_lb_cache),
            "state_bound_prunes":self.state_bound_prunes,
            "child_bound_prunes":self.child_bound_prunes,
            "terminal_lb_skips":self.terminal_lb_skips,
            "improvements":self.improvements,
            "closed_hits":self.closed_hits,
            "closed_states":len(self.closed_threshold),
            "transition_hits":self.transitions.hits,
            "transition_builds":self.transitions.builds,
            "domain_queries":self.transitions.domain_queries,
            "k2_calls":self.k2.calls,
            "seed_k2_calls":self.seed_k2_calls,
            "k2_success_hits":self.k2.success_hits,
            "k2_fail_hits":self.k2.fail_hits,
            "k2_nodes":self.k2.nodes,
            "k2_wall":self.k2.wall,
            "wall":wall,
        }

#!/usr/bin/env python3
"""Exact PDCC-guided last-DR solver with fast integer-pose child ordering.

PDCC structure affects ordering only.  Admissible P1/P2 pruning and exact phase2
completion are unchanged.  The full PDCCState object is constructed only at DR
terminals; intermediate DFS nodes carry a 20-int pose tuple updated through a
precomputed (move,piece,pose)->pose table.
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Sequence

import pdcc_phase2 as p2
import search_twist_skeleton as ts
from cubelab.pdcc import (
    CUBE_ROTATIONS, MOVES, NATIVE_ADJACENCY, PIECE_NAMES, PDCCState,
    slot_for_pose, transition_piece_pose,
)
from cubelab.pdcc.model import PIECE_INDEX

ADJ = tuple((PIECE_INDEX[a], PIECE_INDEX[b]) for a,b in NATIVE_ADJACENCY)
DIFF = tuple(tuple(CUBE_ROTATIONS.compose(CUBE_ROTATIONS.inverse(a),b) for b in range(24)) for a in range(24))
HOME = tuple(tuple(slot_for_pose(piece,p)==piece for p in range(24)) for piece in PIECE_NAMES)

# Compact exact pose transition table: [move index][piece index][pose].
POSE_NEXT = []
for move in ts.MOVE_ORDER:
    spec = MOVES[move]
    by_piece=[]
    for piece in PIECE_NAMES:
        by_piece.append(tuple(transition_piece_pose(piece,p,spec)[0] for p in range(24)))
    POSE_NEXT.append(tuple(by_piece))
POSE_NEXT=tuple(POSE_NEXT)

# One-event quotient lookahead is revisited heavily across transpositions.
# Keep a bounded process-local cache; it is ordering-only and table-independent
# for the fixed CubeLab move/PDB convention loaded in this process.
_EVENT_H_CACHE: dict[tuple[int, str | None], int] = {}
_EVENT_H_CACHE_MAX = 300_000


def move_poses(poses: tuple[int,...], mi: int) -> tuple[int,...]:
    t=POSE_NEXT[mi]
    return tuple(t[i][poses[i]] for i in range(20))

@lru_cache(maxsize=800_000)
def relation_core(poses: tuple[int,...]):
    cnt=[0]*24; moved=0
    for i,p in enumerate(poses):
        cnt[p]+=1
        moved += not HOME[i][p]
    sizes=[n for n in cnt if n]
    bundles=len(sizes); largest=max(sizes,default=0)
    cohesion=sum(n*(n-1)//2 for n in sizes)
    boundary=0; mask=0
    for i,j in ADJ:
        a,b=poses[i],poses[j]
        if a!=b:
            boundary+=1
            mask |= 1 << DIFF[a][b]
    labels=mask.bit_count()
    return boundary,bundles,moved,largest,cohesion,labels

def relation_key(poses):
    b,r,m,l,c,x=relation_core(poses)
    return (b,r,m,-l,-c,x)

@dataclass
class SolveResult:
    seed_total:int; best_total:int; solution:tuple[str,...]; certified:bool; cut:bool
    nodes:int; dr_entries:int; exact_b_calls:int; first_improvement_node:int|None
    elapsed:float; max_completed_depth:int


def next_event_h(q:int, last_face:str|None, co,eo,sl,cos,eos):
    key=(q,last_face)
    hit=_EVENT_H_CACHE.get(key)
    if hit is not None:
        return hit
    best=10**9
    for tm in ts.T_MOVES:
        if not ts.allow(last_face,tm):
            continue
        tq=ts.q_move(q,ts.MI[tm],co,eo,sl)
        h=ts.phase1_lb(tq,cos,eos)
        if h<best: best=h
    if len(_EVENT_H_CACHE) >= _EVENT_H_CACHE_MAX:
        _EVENT_H_CACHE.clear()
    _EVENT_H_CACHE[key]=best
    return best


def ordered(policy,q,poses,remaining,last_face,co,eo,sl,cos,eos):
    out=[]
    for mi,move in enumerate(ts.MOVE_ORDER):
        if not ts.allow(last_face,move): continue
        nq=ts.q_move(q,mi,co,eo,sl)
        h=ts.phase1_lb(nq,cos,eos)
        if h>remaining-1: continue
        if policy=='baseline':
            key=(h,0 if move in ts.T_MOVES else 1,move)
            nposes=None
        else:
            nposes=move_poses(poses,mi)
            rk=relation_key(nposes)
            if policy=='p1_relation':
                key=(h,rk,0 if move in ts.T_MOVES else 1,move)
            elif policy=='relation_first':
                key=(rk,h,0 if move in ts.T_MOVES else 1,move)
            elif policy=='event_relation':
                eh=next_event_h(nq,move[0],co,eo,sl,cos,eos) if remaining>=2 else h
                key=(eh,rk,h,0 if move in ts.T_MOVES else 1,move)
            else: raise ValueError(policy)
        out.append((key,mi,move,nq,nposes))
    out.sort(key=lambda x:x[0])
    return out


def solve(scramble_word:Sequence[str],tables,policy:str,*,time_limit:float,node_cap:int):
    co,eo,sl,cos,eos,cp,up,sp,cpd,upd,spd=tables
    start=ts.engine.from_word(' '.join(scramble_word)); q0=ts.q_of(start)
    seed=ts.normalize(ts.inverse_word(tuple(scramble_word)))
    if not start.apply_word(seed).is_solved(): raise RuntimeError('inverse seed replay failed')
    incumbent=len(seed); best=seed; seed_total=incumbent
    p1lb=ts.phase1_lb(q0,cos,eos)
    nodes=entries=bcalls=0; cut=False; max_completed=p1lb-1; first=None
    t0=time.time(); bcache={}
    depth=p1lb
    while depth<=incumbent-1 and not cut:
        path=[]
        def dfs(q,poses,remaining,last_face,parent_q):
            nonlocal nodes,entries,bcalls,cut,incumbent,best,first
            if cut:return
            nodes+=1
            if nodes>node_cap or time.time()-t0>=time_limit:
                cut=True;return
            if ts.phase1_lb(q,cos,eos)>remaining:return
            if q==ts.GOAL_Q and remaining==0:
                if parent_q==ts.GOAL_Q:return
                entries+=1
                st=PDCCState(poses)
                pk=ts.p_of(st)
                if pk is None: raise RuntimeError('DR terminal missing P2 coordinate')
                p2lb=ts.phase2_lb(pk,cpd,upd,spd)
                if depth+p2lb>incumbent:return
                maxb=incumbent-depth; prev=path[-1][0] if path else None
                ck=(pk,maxb,prev)
                val=bcache.get(ck)
                if val is None:
                    val=p2.shortest_b_compatible(pk,prev,maxb,cp,up,sp,cpd,upd,spd);bcache[ck]=val;bcalls+=1
                bw,bd=val
                if bw is None:return
                sol=tuple(path)+tuple(bw)
                if len(sol)<incumbent:
                    if not start.apply_word(sol).is_solved(): raise RuntimeError('candidate replay failed')
                    incumbent=len(sol);best=sol
                    if first is None:first=nodes
                return
            if remaining==0:return
            for _key,mi,m,nq,np in ordered(policy,q,poses,remaining,last_face,co,eo,sl,cos,eos):
                if np is None: np=move_poses(poses,mi)
                path.append(m);dfs(nq,np,remaining-1,m[0],q);path.pop()
                if cut:return
        dfs(q0,start.poses,depth,None,None)
        if not cut:max_completed=depth
        depth+=1
    certified=(not cut) and max_completed>=incumbent-1
    return SolveResult(seed_total,incumbent,tuple(best),certified,cut,nodes,entries,bcalls,first,time.time()-t0,max_completed)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scramble',required=True)
    ap.add_argument('--p1-order',choices=('baseline','p1_relation','relation_first','event_relation'),default='event_relation')
    ap.add_argument('--time-limit',type=float,default=60);ap.add_argument('--node-cap',type=int,default=3_000_000)
    ap.add_argument('--cache',default='reports/pdcc_cache/twist_skeleton_tables_v1.pkl');a=ap.parse_args()
    print('loading/building pruning tables...',flush=True);tt=time.time();tables,loaded=ts.cache_load_or_build(Path(a.cache));print(f"tables {'loaded from cache' if loaded else 'built and cached'} in {time.time()-tt:.2f}s")
    r=solve(tuple(a.scramble.split()),tables,a.p1_order,time_limit=a.time_limit,node_cap=a.node_cap)
    print('\nPDCC GUIDED EXACT SOLVER v10');print('='*80)
    for k,v in [('scramble',a.scramble),('P1 order',a.p1_order),('seed total',r.seed_total),('best total',r.best_total),('certified',r.certified),('search cut',r.cut),('nodes',f'{r.nodes:,}'),('true DR entries',f'{r.dr_entries:,}'),('exact B calls',r.exact_b_calls),('first improvement',r.first_improvement_node if r.first_improvement_node is not None else '(none)'),('elapsed',f'{r.elapsed:.2f}s'),('solution',' '.join(r.solution))]:print(f'{k:18}: {v}')
    print('full replay       :',ts.engine.from_word(a.scramble).apply_word(r.solution).is_solved())
    print('NOTE: pose/event signals order children only; exact proof pruning is unchanged.')
    return 0
if __name__=='__main__':raise SystemExit(main())

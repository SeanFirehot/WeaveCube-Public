"""CubeLab v36.6i.13 - exact no-suffix membership gate.

This keeps only the strongest/cheapest fact discovered in d5:

    For fixed (remaining P1 length, previous face, packed q),
    does ANY canonical exact-length suffix exist that enters GOAL_Q
    from a non-GOAL parent?

If no:
    exact FALSE -> H child may be pruned.

If yes:
    UNKNOWN -> main DFS continues normally.

No poses, P2 coordinates, P2 LB, K2 table, or suffix replay are used.

Runtime representation is per-bucket frozenset[int], so membership is a
C-level hash lookup. The gate can run BEFORE `pg.move_poses`, avoiding even the
pose update for pruned children.
"""
from __future__ import annotations

from dataclasses import dataclass,asdict
import pickle,time
from pathlib import Path

import search_twist_skeleton as ts

MAGIC="CubeLab-v36.6i.13-no-suffix-membership"
CTX=(None,"U","D","R","L","F","B")
CTX_ID={x:i for i,x in enumerate(CTX)}


@dataclass
class Stats:
    queries:int=0
    false_answers:int=0
    unknown_answers:int=0
    zero_remaining_false:int=0
    membership_false:int=0
    membership_hit_unknown:int=0
    wall:float=0.0

    def row(self): return asdict(self)


class NoSuffixMembershipTable:
    def __init__(self,max_depth,buckets,meta=None):
        self.max_depth=int(max_depth)
        self.buckets=buckets
        self.meta=meta or {}

    @classmethod
    def load(cls,path):
        p=Path(path)
        with p.open("rb") as f:
            x=pickle.load(f)
        if x.get("magic")!=MAGIC:
            raise RuntimeError(f"bad no-suffix table {p}")
        return cls(x["max_depth"],x["buckets"],x.get("meta") or {})


class NoSuffixRefuter:
    def __init__(self,table):
        self.table=table
        self.stats=Stats()

    def query(self,q,remaining,last_ctx,parent_q):
        """False = exact no-suffix proof. None = unresolved."""
        s=self.stats
        s.queries+=1
        t0=time.perf_counter()

        if remaining==0:
            # True DR entry must be GOAL_Q reached from a non-GOAL parent.
            if q!=ts.GOAL_Q or parent_q==ts.GOAL_Q:
                s.false_answers+=1
                s.zero_remaining_false+=1
                s.wall+=time.perf_counter()-t0
                return False
            s.unknown_answers+=1
            s.wall+=time.perf_counter()-t0
            return None

        if remaining>self.table.max_depth:
            s.unknown_answers+=1
            s.wall+=time.perf_counter()-t0
            return None

        if int(q) not in self.table.buckets[remaining][last_ctx]:
            s.false_answers+=1
            s.membership_false+=1
            s.wall+=time.perf_counter()-t0
            return False

        s.unknown_answers+=1
        s.membership_hit_unknown+=1
        s.wall+=time.perf_counter()-t0
        return None

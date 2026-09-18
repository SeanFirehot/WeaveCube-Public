#!/usr/bin/env python3
from __future__ import annotations
from dataclasses import dataclass
import search_twist_skeleton as ts
from p1_orientation_domains_v37_0 import D6MoveDomainOracle, MOVE_INDEX

@dataclass
class OrientationSynthesisResult:
    success: bool
    word: tuple[str, ...]
    domain_masks: tuple[int, ...]
    domain_sizes: tuple[int, ...]
    q_path: tuple[int, ...]
    reason: str

    @property
    def singleton_steps(self):
        return sum(n == 1 for n in self.domain_sizes)

    @property
    def ambiguous_steps(self):
        return sum(n > 1 for n in self.domain_sizes)


def synthesize_orientation_suffix(q0, remaining, previous_face, oracle: D6MoveDomainOracle, *, forced_first_move=None):
    q=int(q0); rem=int(remaining); last=previous_face
    word=[]; masks=[]; sizes=[]; q_path=[q]; parent_q=None
    while rem>0:
        domain=oracle.allowed_domain(q, rem, last)
        masks.append(domain.mask); sizes.append(domain.size())
        if domain.size()==0:
            return OrientationSynthesisResult(False,tuple(word),tuple(masks),tuple(sizes),tuple(q_path),f"empty exact domain at remaining={rem}")
        if forced_first_move is not None and not word:
            if not domain.contains(forced_first_move):
                return OrientationSynthesisResult(False,tuple(word),tuple(masks),tuple(sizes),tuple(q_path),f"unsupported forced first move {forced_first_move}")
            move=forced_first_move
        else:
            move=domain.moves()[0]
        parent_q=q
        q=ts.q_move(q, MOVE_INDEX[move], oracle.co, oracle.eo, oracle.sl)
        word.append(move); q_path.append(q); last=move[0]; rem-=1
    if q!=ts.GOAL_Q:
        return OrientationSynthesisResult(False,tuple(word),tuple(masks),tuple(sizes),tuple(q_path),"final q != GOAL_Q")
    if parent_q==ts.GOAL_Q:
        return OrientationSynthesisResult(False,tuple(word),tuple(masks),tuple(sizes),tuple(q_path),"not a true last DR entry")
    return OrientationSynthesisResult(True,tuple(word),tuple(masks),tuple(sizes),tuple(q_path),"ok")

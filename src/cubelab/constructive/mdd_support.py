from __future__ import annotations

from dataclasses import dataclass

from cubelab.column_families.symbolic_mdd_v2 import SymbolicMDD

from .problem_scope import ProblemScope


@dataclass(
    frozen=True,
    slots=True,
)
class SymbolicMDDSupport:
    """
    Exact correlation-preserving SAME-word support.

    The MDD, not projected column domains, is the authority.
    """

    problem_scope: ProblemScope
    mdd: SymbolicMDD

    def intersect(
        self,
        other: "SymbolicMDDSupport",
        *,
        node_cap: int = 5_000_000,
    ) -> "SymbolicMDDSupport":

        self.problem_scope.require_same(
            other.problem_scope
        )

        return SymbolicMDDSupport(
            problem_scope=
                self.problem_scope,

            mdd=
                self.mdd.intersect(
                    other.mdd,
                    node_cap=node_cap,
                ),
        )

    def count(
        self,
    ) -> int:
        return int(
            self.mdd.path_count()
        )

    def empty_p(
        self,
    ) -> bool:
        return self.count() == 0

    def first_word(
        self,
    ) -> tuple[str, ...] | None:

        sample = self.mdd.sample(
            limit=1
        )

        if not sample:
            return None

        return tuple(
            str(v)
            for v in sample[
                0
            ]
        )

    def column_domain_masks(
        self,
    ) -> tuple[int, ...]:

        return tuple(
            int(v)
            for v in self.mdd.supported_move_masks()
        )

    def structural_hash(
        self,
    ) -> str:

        return str(
            self.mdd.structural_hash()
        )

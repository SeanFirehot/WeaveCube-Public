from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json


@dataclass(
    frozen=True,
    slots=True,
)
class ProblemScope:
    """
    Semantic scope of one constructive SAME-word problem.

    This is deliberately separate from SupportScope, which identifies
    only the underlying explicit word universe.

    Two exact languages may be intersected as one solver problem only
    when this semantic scope is identical.
    """

    schema: str

    state_sha256: str
    remaining: int
    previous_face: str | None

    q_transition_hash: str
    canonical_rule_hash: str

    goal: str

    def to_json(
        self,
    ) -> dict[str, object]:
        return {
            "schema":
                self.schema,

            "state_sha256":
                self.state_sha256,

            "remaining":
                int(
                    self.remaining
                ),

            "previous_face":
                self.previous_face,

            "q_transition_hash":
                self.q_transition_hash,

            "canonical_rule_hash":
                self.canonical_rule_hash,

            "goal":
                self.goal,
        }

    @property
    def digest(
        self,
    ) -> str:

        payload = json.dumps(
            self.to_json(),
            sort_keys=True,
            separators=(
                ",",
                ":",
            ),
        ).encode(
            "utf-8"
        )

        return sha256(
            payload
        ).hexdigest()

    def require_same(
        self,
        other: "ProblemScope",
    ) -> None:

        if self != other:
            raise ValueError(
                "ProblemScope mismatch: "
                f"{self!r} != {other!r}"
            )

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np


class SixCosetATOTerminal:
    """
    Exact universal terminal for physical all-ATO-zero CubeLab Q states.

    Frozen theorem/runtime contract:

        physical ATO
            = disjoint union of six half-turn cosets

    Query:
        Q
        -> source-piece destination permutation
        -> six inverse-representative normalizations
        -> exactly one frozen-H membership hit
        -> exact H parent tail
        -> representative inverse

    No delta/S3 classifier is required in the hot path.
    """

    EXPECTED_H_STATES = 663_552

    def __init__(
        self,
        *,
        codes: np.ndarray,
        distance: np.ndarray,
        parent_index: np.ndarray,
        parent_move_id: np.ndarray,
        dest_from_q: np.ndarray,
        digit_from_dest: np.ndarray,
        inv_rep_perm: np.ndarray,
        inv_rep_word_ids: np.ndarray,
        inv_rep_word_lengths: np.ndarray,
        move_names: Sequence[str],
        native_q_hash: str,
        root_index: int,
        metadata: dict,
    ):
        self.codes = codes
        self.distance = distance
        self.parent_index = parent_index
        self.parent_move_id = parent_move_id
        self.dest_from_q = dest_from_q
        self.digit_from_dest = digit_from_dest
        self.inv_rep_perm = inv_rep_perm
        self.inv_rep_word_ids = inv_rep_word_ids
        self.inv_rep_word_lengths = inv_rep_word_lengths

        self.move_names = tuple(str(x) for x in move_names)
        self.native_q_hash = str(native_q_hash)
        self.root_index = int(root_index)
        self.metadata = dict(metadata)

        self._piece20 = np.arange(20, dtype=np.intp)
        self._piece_axis = self._piece20[None, :]
        self._identity20 = np.arange(20, dtype=np.int16)

        self._weights = (
            np.uint64(1)
            << (
                np.uint64(2)
                * np.arange(20, dtype=np.uint64)
            )
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        *,
        expected_native_q_hash: str | None = None,
        expected_move_names: Sequence[str] | None = None,
        validate: bool = True,
    ) -> "SixCosetATOTerminal":
        path = Path(path)

        if not path.is_file():
            raise FileNotFoundError(path)

        with np.load(path, allow_pickle=False) as x:
            codes = np.asarray(
                x["codes"],
                dtype=np.uint64,
            ).copy()

            distance = np.asarray(
                x["distance"],
                dtype=np.uint8,
            ).copy()

            parent_index = np.asarray(
                x["parent_index"],
                dtype=np.int32,
            ).copy()

            parent_move_id = np.asarray(
                x["parent_move_id"],
                dtype=np.uint8,
            ).copy()

            dest_from_q = np.asarray(
                x["dest_from_q"],
                dtype=np.int16,
            ).copy()

            digit_from_dest = np.asarray(
                x["digit_from_dest"],
                dtype=np.int8,
            ).copy()

            inv_rep_perm = np.asarray(
                x["inv_rep_perm"],
                dtype=np.int16,
            ).copy()

            inv_rep_word_ids = np.asarray(
                x["inv_rep_word_ids"],
                dtype=np.uint8,
            ).copy()

            inv_rep_word_lengths = np.asarray(
                x["inv_rep_word_lengths"],
                dtype=np.uint8,
            ).copy()

            raw = x["metadata_json"]

            if isinstance(raw, np.ndarray):
                raw = raw.item()

            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")

            metadata = (
                json.loads(raw)
                if isinstance(raw, str)
                else dict(raw)
            )

        native_q_hash = str(
            metadata["native_q_hash"]
        )

        move_names = tuple(
            str(x)
            for x in metadata["move_names"]
        )

        if (
            expected_native_q_hash is not None
            and native_q_hash
            != str(expected_native_q_hash)
        ):
            raise RuntimeError(
                "six-coset asset/native Q hash mismatch: "
                f"{native_q_hash} != "
                f"{expected_native_q_hash}"
            )

        if expected_move_names is not None:
            expected = tuple(
                str(x)
                for x in expected_move_names
            )

            if move_names != expected:
                raise RuntimeError(
                    "six-coset asset/runtime move-order mismatch"
                )

        roots = np.flatnonzero(
            distance == 0
        )

        if len(roots) != 1:
            raise RuntimeError(
                f"expected one H root, got {len(roots)}"
            )

        obj = cls(
            codes=codes,
            distance=distance,
            parent_index=parent_index,
            parent_move_id=parent_move_id,
            dest_from_q=dest_from_q,
            digit_from_dest=digit_from_dest,
            inv_rep_perm=inv_rep_perm,
            inv_rep_word_ids=inv_rep_word_ids,
            inv_rep_word_lengths=inv_rep_word_lengths,
            move_names=move_names,
            native_q_hash=native_q_hash,
            root_index=int(roots[0]),
            metadata=metadata,
        )

        if validate:
            obj.validate()

        return obj

    def validate(self) -> None:
        n = self.EXPECTED_H_STATES

        expected_shapes = {
            "codes": (n,),
            "distance": (n,),
            "parent_index": (n,),
            "parent_move_id": (n,),
            "dest_from_q": (20, 24),
            "digit_from_dest": (20, 20),
            "inv_rep_perm": (6, 20),
            "inv_rep_word_lengths": (6,),
        }

        for name, shape in expected_shapes.items():
            arr = getattr(self, name)

            if arr.shape != shape:
                raise RuntimeError(
                    f"{name} shape drift: "
                    f"{arr.shape} != {shape}"
                )

        if (
            self.inv_rep_word_ids.ndim != 2
            or self.inv_rep_word_ids.shape[0] != 6
        ):
            raise RuntimeError(
                "inv_rep_word_ids shape drift"
            )

        if np.any(
            self.codes[1:]
            <= self.codes[:-1]
        ):
            raise RuntimeError(
                "H codes are not strictly sorted"
            )

        if int(self.distance.max()) != 15:
            raise RuntimeError(
                "H diameter drift"
            )

        nonroot = self.distance > 0

        pi = self.parent_index[nonroot]

        if np.any(pi < 0) or np.any(pi >= n):
            raise RuntimeError(
                "parent index outside H bank"
            )

        if not np.array_equal(
            self.distance[
                pi.astype(np.intp)
            ].astype(np.int16),
            self.distance[nonroot].astype(
                np.int16
            ) - 1,
        ):
            raise RuntimeError(
                "sealed parent distance contract failed"
            )

        if self.parent_index[
            self.root_index
        ] != -1:
            raise RuntimeError(
                "H root unexpectedly has parent"
            )

        if len(self.move_names) != 18:
            raise RuntimeError(
                "expected 18 move names"
            )

        if np.any(
            self.inv_rep_word_lengths
            > self.inv_rep_word_ids.shape[1]
        ):
            raise RuntimeError(
                "representative inverse length overflow"
            )

    @property
    def resident_bytes(self) -> int:
        return int(
            self.codes.nbytes
            + self.distance.nbytes
            + self.parent_index.nbytes
            + self.parent_move_id.nbytes
            + self.dest_from_q.nbytes
            + self.digit_from_dest.nbytes
            + self.inv_rep_perm.nbytes
            + self.inv_rep_word_ids.nbytes
            + self.inv_rep_word_lengths.nbytes
        )

    def is_ato(self, q) -> bool:
        qv = np.asarray(q)

        if qv.shape != (20,):
            return False

        if np.any(qv < 0) or np.any(qv >= 24):
            return False

        qv = qv.astype(
            np.intp,
            copy=False,
        )

        dest = self.dest_from_q[
            self._piece20,
            qv,
        ]

        return bool(
            np.all(dest >= 0)
        )

    def _destination_perm(
        self,
        q,
        *,
        allow_non_ato: bool,
    ) -> np.ndarray | None:
        qv = np.asarray(q)

        if qv.shape != (20,):
            raise ValueError(
                f"Q state must have shape (20,), "
                f"got {qv.shape}"
            )

        if np.any(qv < 0) or np.any(qv >= 24):
            if allow_non_ato:
                return None

            raise ValueError(
                "Q code outside native 0..23 range"
            )

        qv = qv.astype(
            np.intp,
            copy=False,
        )

        gp = self.dest_from_q[
            self._piece20,
            qv,
        ]

        if np.any(gp < 0):
            if allow_non_ato:
                return None

            raise ValueError(
                "state is not all-ATO-zero"
            )

        if not np.array_equal(
            np.sort(gp),
            self._identity20,
        ):
            raise RuntimeError(
                "ATO Q state violates physical "
                "piece occupancy"
            )

        return gp

    def _solve_from_perm(
        self,
        gp: np.ndarray,
    ) -> tuple[
        int,
        int,
        int,
        tuple[int, ...],
    ]:
        # Six exact normalizations:
        #
        #     P_h = P_g o P_r_i^-1
        hp6 = gp[
            self.inv_rep_perm
        ]

        digits = self.digit_from_dest[
            self._piece_axis,
            hp6,
        ]

        locally_valid = np.all(
            digits >= 0,
            axis=1,
        )

        valid = np.flatnonzero(
            locally_valid
        )

        packed = np.zeros(
            6,
            dtype=np.uint64,
        )

        if len(valid):
            packed[valid] = np.sum(
                digits[
                    valid
                ].astype(
                    np.uint64,
                    copy=False,
                )
                * self._weights[
                    None, :
                ],
                axis=1,
                dtype=np.uint64,
            )

        hit = np.zeros(
            6,
            dtype=bool,
        )

        index = np.zeros(
            6,
            dtype=np.int64,
        )

        if len(valid):
            ii = np.searchsorted(
                self.codes,
                packed[valid],
            )

            in_range = (
                ii
                < self.EXPECTED_H_STATES
            )

            safe = np.minimum(
                ii,
                self.EXPECTED_H_STATES - 1,
            )

            exact = (
                in_range
                & (
                    self.codes[safe]
                    == packed[valid]
                )
            )

            hit[valid] = exact
            index[valid] = safe

        hits = np.flatnonzero(hit)

        # For a physical ATO state, R5-R11 theorem says
        # this must be exactly one.
        if len(hits) != 1:
            raise RuntimeError(
                "six-coset theorem/runtime contract "
                "failed: expected exactly one H hit, "
                f"got {hits.tolist()}"
            )

        cid = int(hits[0])
        h_index = int(index[cid])
        h_distance = int(
            self.distance[h_index]
        )

        out: list[int] = []

        cur = h_index

        # Follow exactly the certified H distance.
        #
        # A depth-15 state requires exactly 15 parent transitions;
        # checking the root only at the top of range(15) creates an
        # off-by-one failure after the 15th transition reaches root.
        for step in range(h_distance):
            current_distance = int(
                self.distance[cur]
            )

            expected_distance = (
                h_distance - step
            )

            if (
                current_distance
                != expected_distance
            ):
                raise RuntimeError(
                    "sealed H parent-chain distance drift: "
                    f"step={step} "
                    f"expected={expected_distance} "
                    f"actual={current_distance}"
                )

            move_id = int(
                self.parent_move_id[cur]
            )

            if not (
                0 <= move_id
                < len(self.move_names)
            ):
                raise RuntimeError(
                    "sealed parent move ID corrupt"
                )

            out.append(move_id)

            cur = int(
                self.parent_index[cur]
            )

            if cur < 0:
                raise RuntimeError(
                    "sealed H parent chain "
                    "terminated early"
                )

        if cur != self.root_index:
            raise RuntimeError(
                "sealed H tail ended at wrong root"
            )

        if int(self.distance[cur]) != 0:
            raise RuntimeError(
                "sealed H tail root has nonzero distance"
            )

        if len(out) != h_distance:
            raise RuntimeError(
                "sealed H tail length != exact H distance"
            )

        ln = int(
            self.inv_rep_word_lengths[cid]
        )

        out.extend(
            int(x)
            for x in self.inv_rep_word_ids[
                cid,
                :ln,
            ]
        )

        return (
            cid,
            h_index,
            h_distance,
            tuple(out),
        )

    def try_solve_ids(
        self,
        q,
    ) -> tuple[
        int,
        int,
        int,
        tuple[int, ...],
    ] | None:
        """
        Return None only when q is not all-ATO-zero.

        A physical ATO input must have exactly one H coset hit;
        violations of that proven contract raise RuntimeError.
        """
        gp = self._destination_perm(
            q,
            allow_non_ato=True,
        )

        if gp is None:
            return None

        return self._solve_from_perm(gp)

    def solve_ids(
        self,
        q,
    ) -> tuple[
        int,
        int,
        int,
        tuple[int, ...],
    ]:
        gp = self._destination_perm(
            q,
            allow_non_ato=False,
        )

        assert gp is not None

        return self._solve_from_perm(gp)

    def solve_word(
        self,
        q,
    ) -> tuple[str, ...]:
        _, _, _, ids = self.solve_ids(q)

        return tuple(
            self.move_names[mid]
            for mid in ids
        )

    def try_solve_word(
        self,
        q,
    ) -> tuple[str, ...] | None:
        hit = self.try_solve_ids(q)

        if hit is None:
            return None

        return tuple(
            self.move_names[mid]
            for mid in hit[3]
        )


__all__ = [
    "SixCosetATOTerminal",
]

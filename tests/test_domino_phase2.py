from cubelab.domino_phase2 import (
    build_phase2_pruning_tables,
    phase2_coordinates_from_transformation,
    solve_phase2,
    solve_with_dr,
)
from cubelab.domino_reduction import DRAxis
from cubelab.transformations import from_sequence


def test_solved_phase2_coordinates_are_zero():
    coordinates = phase2_coordinates_from_transformation(from_sequence(()))
    assert coordinates.is_solved


def test_phase2_move_table_matches_physical_move():
    tables = build_phase2_pruning_tables()
    state = from_sequence(("U", "R2", "F2", "D'"))
    coordinates = phase2_coordinates_from_transformation(state)
    for move_index, move in enumerate(tables.moves):
        physical = phase2_coordinates_from_transformation(from_sequence(("U", "R2", "F2", "D'", move)))
        assert tables.corner_move[coordinates.corner_permutation_index][move_index] == physical.corner_permutation_index
        assert tables.ud_edge_move[coordinates.ud_edge_permutation_index][move_index] == physical.ud_edge_permutation_index
        assert tables.slice_edge_move[coordinates.slice_edge_permutation_index][move_index] == physical.slice_edge_permutation_index


def test_phase2_solves_known_dr_state():
    state = from_sequence(("U", "R2", "F2", "D'", "L2"))
    solution = solve_phase2(state, axis=DRAxis.UD, max_depth=8)
    assert solution is not None
    assert from_sequence(("U", "R2", "F2", "D'", "L2") + solution.sequence).mapping == from_sequence(()).mapping


def test_full_dr_solver_solves_short_scramble():
    scramble = ("R", "U", "R'", "U'", "F", "D")
    solution = solve_with_dr(scramble, max_dr_depth=8, dr_extra_depth=1, terminals_per_axis=8, max_phase2_depth=14)
    assert solution is not None
    assert from_sequence(scramble + solution.full_sequence).mapping == from_sequence(()).mapping


def test_phase2_solves_fb_axis_state():
    state = from_sequence(("F", "U2", "R2", "B'", "D2"))
    solution = solve_phase2(state, axis=DRAxis.FB, max_depth=8)
    assert solution is not None
    assert from_sequence(("F", "U2", "R2", "B'", "D2") + solution.sequence).mapping == from_sequence(()).mapping


def test_phase2_solves_rl_axis_state():
    state = from_sequence(("R", "U2", "F2", "L'", "D2"))
    solution = solve_phase2(state, axis=DRAxis.RL, max_depth=8)
    assert solution is not None
    assert from_sequence(("R", "U2", "F2", "L'", "D2") + solution.sequence).mapping == from_sequence(()).mapping


def test_phase2_lower_bound_is_admissible_for_known_state():
    from cubelab.domino_phase2 import phase2_lower_bound
    state = from_sequence(("U", "R2", "F2", "D'", "L2"))
    lower = phase2_lower_bound(state, axis=DRAxis.UD)
    solution = solve_phase2(state, axis=DRAxis.UD, max_depth=8)
    assert solution is not None
    assert lower <= solution.depth


def test_selector_limited_full_solver_still_solves():
    scramble = ("R", "U", "R'", "U'", "F", "D")
    solution = solve_with_dr(
        scramble,
        max_dr_depth=8,
        dr_extra_depth=1,
        terminals_per_axis=8,
        max_phase2_depth=14,
        phase2_exact_candidates=4,
    )
    assert solution is not None
    assert from_sequence(scramble + solution.full_sequence).mapping == from_sequence(()).mapping


def test_selector_can_include_shortest_terminal_per_axis():
    scramble = ("R", "U", "R'", "U'", "F", "D")
    solution = solve_with_dr(
        scramble,
        max_dr_depth=8,
        dr_extra_depth=1,
        terminals_per_axis=8,
        max_phase2_depth=14,
        phase2_exact_candidates=1,
        include_shortest_per_axis=True,
    )
    assert solution is not None
    assert from_sequence(scramble + solution.full_sequence).mapping == from_sequence(()).mapping


def test_selector_safeguard_contains_lb_top_k_and_shortest_axes():
    from cubelab.domino_phase2 import select_dr_terminals_for_exact_evaluation
    from cubelab.domino_reduction import find_dr_portfolio_ida_from_sequence

    portfolio = find_dr_portfolio_ida_from_sequence(
        ("R", "U", "R'", "U'", "F", "D"),
        max_depth=8,
        extra_depth=1,
        max_terminals_per_axis=8,
    )
    top4 = select_dr_terminals_for_exact_evaluation(
        portfolio.terminals, lower_bound_top_k=4
    )
    safe = select_dr_terminals_for_exact_evaluation(
        portfolio.terminals, lower_bound_top_k=4, include_shortest_per_axis=True
    )
    safe_keys = {(t.axis, t.sequence) for t in safe}
    assert {(t.axis, t.sequence) for t in top4} <= safe_keys
    for axis in {t.axis for t in portfolio.terminals}:
        shortest = min(
            (t for t in portfolio.terminals if t.axis == axis),
            key=lambda t: (len(t.sequence), t.sequence),
        )
        assert (shortest.axis, shortest.sequence) in safe_keys

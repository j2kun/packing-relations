from quasiaffine import bind_dims
from relation import IndexSpace
from hoist import is_valid_layout, apply_layout, hoist_through_matvec, Layout

import numpy as np


def test_hoist_cyclic_rotation_through_halevi_shoup():
    mat_rows = 4
    mat_cols = 4
    num_cts = 4
    num_slots = 8

    data = np.arange(mat_rows * mat_cols, dtype=np.int32).reshape((mat_rows, mat_cols))

    row, col, ct, slot = bind_dims("row", "col", "ct", "slot")
    domain = IndexSpace([mat_rows, mat_cols], dims=[row, col])
    codomain = IndexSpace([num_cts, num_slots], dims=[ct, slot])

    # Halevi-Shoup diagonal matrix layout
    constraints = [(slot % mat_rows) - row, (ct + slot) % mat_cols - col]
    diagonal_matrix_layout = Layout(
        domain=domain, codomain=codomain, constraints=constraints
    )
    assert is_valid_layout(diagonal_matrix_layout)
    halevi_shoup_applied = apply_layout(diagonal_matrix_layout, data)
    expected_halevi_shoup = np.array(
        [
            [0, 5, 10, 15, 0, 5, 10, 15],
            [1, 6, 11, 12, 1, 6, 11, 12],
            [2, 7, 8, 13, 2, 7, 8, 13],
            [3, 4, 9, 14, 3, 4, 9, 14],
        ]
    )
    assert np.array_equal(halevi_shoup_applied, expected_halevi_shoup)

    # Vector layout and post-matvec conversion
    v, w, x = bind_dims("v", "w", "x")
    vec_domain1 = IndexSpace([mat_cols], dims=[v])
    vec_domain2 = IndexSpace([mat_cols], dims=[w])
    ciphertext_domain = IndexSpace([num_slots], dims=[x])
    vec_layout1 = Layout(vec_domain1, ciphertext_domain, [(x % mat_cols) - v])
    vec_layout2 = Layout(vec_domain2, ciphertext_domain, [((x + 1) % mat_cols) - w])

    assert np.array_equal(
        apply_layout(vec_layout1, np.arange(mat_cols)),
        np.array([0, 1, 2, 3, 0, 1, 2, 3]),
    )
    assert np.array_equal(
        apply_layout(vec_layout2, np.arange(mat_cols)),
        np.array([1, 2, 3, 0, 1, 2, 3, 0]),
    )

    convert_1_to_2 = vec_layout2.inverse().compose(vec_layout1)
    assert is_valid_layout(convert_1_to_2)

    hoisted = hoist_through_matvec(diagonal_matrix_layout, convert_1_to_2)
    assert is_valid_layout(hoisted)

    hoisted_applied = apply_layout(hoisted, data)
    expected_hoisted = np.array(
        [
            [1, 6, 11, 12, 1, 6, 11, 12],
            [2, 7, 8, 13, 2, 7, 8, 13],
            [3, 4, 9, 14, 3, 4, 9, 14],
            [0, 5, 10, 15, 0, 5, 10, 15],
        ]
    )
    assert np.array_equal(hoisted_applied, expected_hoisted)

from quasiaffine import bind_dims, bind_unique_dims
from relation import IndexSpace, IntegerRelation


# A layout is an integer relation where the domain is the data index space and
# the codomain is the ciphertext slot index space. A tuple (I1, I2) is in the
# relation if the data element at index I1 is stored in the ciphertext slot at
# I2. Note that this relation is invalid if two data elements map to the same
# ciphertext slot, but the mapping may be partial (leaving some slots unused)
# or replicate data among slots.
Layout = IntegerRelation


def is_valid_layout(layout: Layout) -> bool:
    # no data collides in a slot
    visited_slots = set()
    visited_data = set()

    for data_index, slot_index in layout.enumerate_relation():
        if slot_index in visited_slots:
            raise ValueError(
                f"Invalid layout: data {data_index} collides with another in slot {slot_index}"
            )
        visited_slots.add(slot_index)
        visited_data.add(data_index)

    # all data is mapped somewhere
    if len(visited_data) != layout.domain.size():
        raise ValueError(
            f"Invalid layout: {layout.domain.size() - len(visited_data)} data elements are not mapped to any slot"
        )

    return True


def hoist_through_matvec(mat_layout: Layout, conversion: IntegerRelation):
    # Each constraint is of the form:
    #
    #   f(d0, d1, c0, c1, l0, l1, ...) = 0
    #
    # and we need to add a new dimension corresponding to the domain of the
    # conversion relation, as well as replace d1 with the new dimension.
    #
    # Can't use compose because we're working on a subset of the domain's index
    # space.
    assert len(conversion.domain.dims) == 1
    assert conversion.domain.dim_sizes[0] == mat_layout.domain.dim_sizes[1]

    # If the integer relation has dims (i, j), we want j to map to d1
    mat_layout_dim = mat_layout.domain.dims[1]
    (new_dim,) = bind_unique_dims(1)

    new_domain = mat_layout.domain
    new_codomain = mat_layout.codomain
    # Add a new existential quantifier for the conversion relation domain.
    new_locals = IndexSpace(
        dim_sizes=mat_layout.locals.dim_sizes
        + conversion.locals.dim_sizes
        + [conversion.codomain.dim_sizes[0]],
        dims=mat_layout.locals.dims + conversion.locals.dims + [new_dim],
    )

    new_constraints = []
    # Replace d1 with the new dimension in the existing constraints.
    for constraint in mat_layout.constraints:
        new_constraints.append(constraint.replace_dims({mat_layout_dim: new_dim}))

    # Make new constriants using the conversion relation and the existential
    # quantifier.
    for constraint in conversion.constraints:
        # Conv(i, j) = 0 becomes Conv(new_local, d1) = 0
        new_constraints.append(
            constraint.replace_dims(
                {
                    conversion.domain.dims[0]: new_dim,
                    conversion.codomain.dims[0]: mat_layout_dim,
                }
            )
        )

    return Layout(
        domain=new_domain,
        codomain=new_codomain,
        locals=new_locals,
        constraints=new_constraints,
    )


if __name__ == "__main__":
    mat_dim = 4
    num_cts = 4
    num_slots = 8
    # Halevi-Shoup diagonal matrix layout
    row, col, ct, slot = bind_dims("row", "col", "ct", "slot")
    domain = IndexSpace([mat_dim, mat_dim], dims=[row, col])
    # 32 ciphertexts each with 128 slots
    codomain = IndexSpace([mat_dim, num_slots], dims=[ct, slot])

    constraints = [(slot % mat_dim) - row, (row + col) % mat_dim - ct]
    diagonal_matrix_layout = Layout(
        domain=domain, codomain=codomain, constraints=constraints
    )

    print(diagonal_matrix_layout)
    # print("Is valid?", is_valid_layout(diagonal_matrix_layout))

    # Vector layout and post-matvec conversion
    v, w, x = bind_dims("v", "w", "x")
    vec_domain1 = IndexSpace([mat_dim], dims=[v])
    vec_domain2 = IndexSpace([mat_dim], dims=[w])
    ciphertext_domain = IndexSpace([num_slots], dims=[x])
    vec_layout1 = Layout(vec_domain1, ciphertext_domain, [(x % num_slots) - v])
    vec_layout2 = Layout(vec_domain2, ciphertext_domain, [((x + 2) % mat_dim) - w])

    # Hoist the layout conversion up through the matvec mul, by analyzing the
    # composition of the converted layouts to get the transformation required
    # on axis 1 of the matrix.
    convert_1_to_2 = vec_layout2.inverse().compose(vec_layout1)

    # It should only make sense to hoist if the conversion is a permutation.
    # print("conversion is valid?", is_valid_layout(convert_1_to_2))

    hoisted = hoist_through_matvec(diagonal_matrix_layout, convert_1_to_2)
    print(f"hoisted: {hoisted}")

    all_pts = list(hoisted.enumerate_relation())
    all_pts.sort()

    for pt in all_pts:
        print(pt)

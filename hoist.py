from quasiaffine import bind_dims
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


if __name__ == "__main__":
    # Halevi-Shoup diagonal matrix layout
    row, col, ct, slot = bind_dims("row", "col", "ct", "slot")
    domain = IndexSpace([32, 32], dims=[row, col])
    # 32 ciphertexts each with 128 slots
    codomain = IndexSpace([32, 128], dims=[ct, slot])

    constraints = [(slot % 32) - row, (row + col) % 32 - ct]
    diagonal_matrix_layout = Layout(
        domain=domain, codomain=codomain, constraints=constraints
    )

    print(diagonal_matrix_layout)
    # print("Is valid?", is_valid_layout(diagonal_matrix_layout))

    # Vector layout and post-matvec conversion
    v, w, x = bind_dims("v", "w", "x")
    vec_domain1 = IndexSpace([32], dims=[v])
    vec_domain2 = IndexSpace([32], dims=[w])
    ciphertext_domain = IndexSpace([128], dims=[x])
    vec_layout1 = Layout(vec_domain1, ciphertext_domain, [(x % 32) - v])
    vec_layout2 = Layout(vec_domain2, ciphertext_domain, [((x + 5) % 32) - w])

    # Hoist the layout conversion up through the matvec mul, by analyzing the
    # composition of the converted layouts to get the transformation required
    # on axis 1 of the matrix.
    convert_1_to_2 = vec_layout2.inverse().compose(vec_layout1)

    # It should only make sense to hoist if the conversion is a permutation.
    # print("conversion is valid?", is_valid_layout(convert_1_to_2))

    # TODO: apply the hoisting


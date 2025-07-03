import islpy as isl


def enumerate(set_):
    L = []
    set_.foreach_point(lambda x: L.append(x))
    return L


def evaluate_domain(map, domain_values):
    eval_map_vals = ",".join(f"{k}={v}" for (k, v) in domain_values.items())
    eval_map_str = f"{{ [{eval_map_vals}] }}"
    eval_map = isl.Map(s=eval_map_str, context=map.get_ctx())
    return eval_map.reverse().apply_domain(map)


def evaluate_codomain(map, codomain_values):
    eval_map_vals = ",".join(f"{k}={v}" for (k, v) in codomain_values.items())
    eval_map_str = f"{{ [{eval_map_vals}] }}"
    eval_map = isl.Map(s=eval_map_str, context=map.get_ctx())
    return map.apply_range(eval_map.reverse())


def apply_to_domain_dimension(original_map, transform_relation, dim_index):
    """
    Apply a transformation relation to a specific dimension of a map's domain.
    """
    domain_space = original_map.domain().get_space()
    n_dims = domain_space.dim(isl.dim_type.set)

    identity_maps = []
    for i in range(n_dims):
        if i == dim_index:
            identity_maps.append(transform_relation)
        else:
            identity_maps.append(isl.Map.identity(isl.Space.alloc(original_map.get_ctx(), 0, 1, 1)))

    domain_transform = identity_maps[0]
    for i in range(1, n_dims):
        domain_transform = domain_transform.product(identity_maps[i])

    return original_map.apply_domain(domain_transform)


if __name__ == "__main__":
    mat_rows = 4
    mat_cols = 4
    num_cts = 4
    num_slots = 8

    print("Matrix layout")
    ctx = isl.Context()
    layout_map_str = f"""
        {{ [row, col] -> [ct, slot] :
            0 <= row < {mat_rows} and
            0 <= col < {mat_cols} and
            0 <= ct < {num_cts} and
            0 <= slot < {num_slots} and
            (slot - row) % {mat_rows} = 0 and
            (ct + slot - col) % {mat_cols} = 0
        }}
    """

    diagonal_matrix_layout = isl.Map(layout_map_str, ctx)
    print(diagonal_matrix_layout)

    print("Enumerating: ")
    for point in enumerate(diagonal_matrix_layout.range().to_union_set()):
        print(point)

    print("Evaluating at row=1, col=2: ")
    values = {"row": 1, "col": 2}
    sub_map = evaluate_domain(diagonal_matrix_layout, values)
    print(f"sub map is {sub_map}")
    # should be [1, 1], [1, 5]
    for point in enumerate(sub_map.domain().to_union_set()):
        print(point)

    print("Evaluating at ct=0, slot=1: ")
    values = {"ct": 0, "slot": 1}
    sub_map = evaluate_codomain(diagonal_matrix_layout, values)
    print(f"sub map is {sub_map}")
    # should be [1, 1]
    for point in enumerate(sub_map.domain().to_union_set()):
        print(point)

    print("Vector 1 layout")
    vec_layout1_str = f"""
        {{ [v] -> [slot] :
            0 <= v < {mat_cols} and
            0 <= slot < {num_slots} and
            slot % {mat_cols} = v
        }}
    """
    vec_layout1 = isl.Map(vec_layout1_str, ctx)
    print(vec_layout1)

    print("Vector 2 layout")
    vec_layout2_str = f"""
        {{ [v] -> [slot] :
            0 <= v < {mat_cols} and
            0 <= slot < {num_slots} and
            (slot + 1) % {mat_cols} = v
        }}
    """
    vec_layout2 = isl.Map(vec_layout2_str, ctx)
    print(vec_layout2)

    print("vec conversion layout")
    # both are v -> slot, and we want v->v
    convert_1_to_2 = vec_layout2.apply_range(vec_layout1.reverse())
    print(convert_1_to_2)

    # sub_map = evaluate_domain(convert_1_to_2, {"slot": 1})
    # for point in enumerate(sub_map.domain().to_union_set()):
    #    print(point)

    hoisted = apply_to_domain_dimension(diagonal_matrix_layout, convert_1_to_2, 1)

    print("Hoisted layout:")
    print(hoisted)

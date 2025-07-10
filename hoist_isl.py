import islpy as isl
import numpy as np


def point_to_python(point):
    # Alternatively, you have to iterate over the dims and use
    # point.get_coordinate_val(isl.dim_type.set_, 0).to_python()
    return eval(point.to_str().strip("{}"))


def enumerate(set_):
    L = []
    set_.foreach_point(lambda x: L.append(point_to_python(x)))
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


def apply_layout(map, data, ciphertexts_shape):
    dims = ["ct", "slot"]
    def pt_to_values(pt):
        return dict(zip(dims, pt))

    ciphertexts = np.zeros(ciphertexts_shape, dtype=data.dtype)
    for codomain_pt in enumerate(map.range().to_union_set()):
        sub_map = evaluate_codomain(map, pt_to_values(codomain_pt))
        for domain_pt in enumerate(sub_map.domain().to_union_set()):
            ciphertexts[*codomain_pt] = data[*domain_pt]

    return ciphertexts


def apply_to_domain_dimension(original_map, transform_relation, dim_index):
    """
    Apply a transformation relation to a specific dimension of a map's domain.
    """
    n_dims = original_map.dim(isl.dim_type.in_)
    dims = [
        isl.Aff.var_on_domain(original_map.domain().get_space(), isl.dim_type.set, i)
        for i in range(n_dims)
    ]
    id_maps = [isl.Map.from_aff(dim) for dim in dims]
    id_maps[dim_index] = id_maps[dim_index].apply_range(transform_relation)

    product = id_maps[0]
    for i in range(1, n_dims):
        product = product.flat_range_product(id_maps[i])

    return original_map.apply_domain(product.reverse())


if __name__ == "__main__":
    mat_rows = 4
    mat_cols = 4
    num_cts = 4
    num_slots = 8

    data = np.arange(mat_rows * mat_cols).reshape((mat_rows, mat_cols))
    ct_shape = (num_cts, num_slots)

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

    print("Applied: ")
    print(apply_layout(diagonal_matrix_layout, data, ct_shape))

    vec_data = np.arange(mat_cols)
    vec_ct_shape = (num_slots,)
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
    print("Vector 1 layout applied:")
    print(apply_layout(vec_layout1, vec_data, vec_ct_shape))

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
    print("Vector 2 layout applied:")
    print(apply_layout(vec_layout2, vec_data, vec_ct_shape))


    print("vec conversion layout")
    # both are v -> slot, and we want v->v
    convert_1_to_2 = vec_layout2.apply_range(vec_layout1.reverse())
    print(convert_1_to_2)

    print("Vector conversion layout applied:")
    print(apply_layout(convert_1_to_2, vec_data, (mat_cols,)))

    hoisted = apply_to_domain_dimension(diagonal_matrix_layout, convert_1_to_2, 1)

    print("Hoisted layout:")
    print(hoisted)
    print("Applied:")
    print(apply_layout(hoisted, data, ct_shape))

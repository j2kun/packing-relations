import pytest
from quasiaffine import (
    Dim,
    Constant,
    AffineExprKind,
    bind_dims,
    get_constants,
)


class TestAffineExprBasics:

    def test_dim_expr_creation(self):
        d0 = Dim(0)
        assert d0.get_kind() == AffineExprKind.DIM_ID
        assert d0.get_position() == 0
        assert str(d0) == "d0"

    def test_constant_expr_creation(self):
        c42 = Constant(42)
        assert c42.get_kind() == AffineExprKind.CONSTANT
        assert c42.get_value() == 42
        assert str(c42) == "42"

    def test_equality(self):
        d0a = Dim(0)
        d0b = Dim(0)
        d1 = Dim(1)

        assert d0a == d0b
        assert d0a != d1

        # Test equality with int
        c5 = Constant(5)
        assert c5 == 5
        assert c5 != 3


class TestArithmeticOperations:

    def test_addition(self):
        d0 = Dim(0)
        d1 = Dim(1)

        expr = d0 + d1
        assert expr.get_kind() == AffineExprKind.ADD
        assert expr.get_lhs() == d0
        assert expr.get_rhs() == d1

    def test_addition_with_int(self):
        d0 = Dim(0)
        expr = d0 + 5

        assert expr.get_kind() == AffineExprKind.ADD
        assert expr.get_lhs() == d0
        assert expr.get_rhs() == Constant(5)

    def test_multiplication(self):
        d0 = Dim(0)
        expr = d0 * 3

        assert expr.get_kind() == AffineExprKind.MUL
        assert expr.get_lhs() == d0
        assert expr.get_rhs() == Constant(3)

    def test_subtraction(self):
        d0 = Dim(0)
        d1 = Dim(1)

        expr = d0 - d1
        # Subtraction becomes addition with negation
        assert expr.get_kind() == AffineExprKind.ADD

        # Check that d1 was negated (multiplied by -1)
        rhs = expr.get_rhs()
        assert rhs.get_kind() == AffineExprKind.MUL

    def test_floor_div(self):
        d0 = Dim(0)
        expr = d0.floor_div(4)

        assert expr.get_kind() == AffineExprKind.FLOOR_DIV
        assert expr.get_lhs() == d0
        assert expr.get_rhs() == Constant(4)

    def test_ceil_div(self):
        d0 = Dim(0)
        expr = d0.ceil_div(3)

        assert expr.get_kind() == AffineExprKind.CEIL_DIV
        assert expr.get_lhs() == d0
        assert expr.get_rhs() == Constant(3)

    def test_modulo(self):
        d0 = Dim(0)
        expr = d0 % 8

        assert expr.get_kind() == AffineExprKind.MOD
        assert expr.get_lhs() == d0
        assert expr.get_rhs() == Constant(8)


class TestSimplification:

    def test_constant_addition(self):
        # 3 + 5 should simplify to 8
        expr = Constant(3) + Constant(5)
        assert isinstance(expr, Constant)
        assert expr.get_value() == 8

    def test_constant_multiplication(self):
        # 4 * 6 should simplify to 24
        expr = Constant(4) * Constant(6)
        assert isinstance(expr, Constant)
        assert expr.get_value() == 24

    def test_addition_with_zero(self):
        d0 = Dim(0)
        expr = d0 + 0

        # Should simplify to just d0
        assert expr == d0

    def test_multiplication_by_one(self):
        d0 = Dim(0)
        expr = d0 * 1

        # Should simplify to just d0
        assert expr == d0

    def test_multiplication_by_zero(self):
        d0 = Dim(0)
        expr = d0 * 0

        # Should simplify to constant 0
        assert isinstance(expr, Constant)
        assert expr.get_value() == 0

    def test_canonicalization_constant_left(self):
        # 5 + d0 should become d0 + 5
        d0 = Dim(0)
        expr = Constant(5) + d0

        assert expr.get_lhs() == d0
        assert expr.get_rhs() == Constant(5)

    def test_successive_additions(self):
        # (d0 + 2) + 3 should become d0 + 5
        d0 = Dim(0)
        expr1 = d0 + 2
        expr2 = expr1 + 3

        assert expr2.get_lhs() == d0
        assert isinstance(expr2.get_rhs(), Constant)
        assert expr2.get_rhs().get_value() == 5

    def test_successive_multiplications(self):
        # (d0 * 2) * 3 should become d0 * 6
        d0 = Dim(0)
        expr1 = d0 * 2
        expr2 = expr1 * 3

        assert expr2.get_lhs() == d0
        assert isinstance(expr2.get_rhs(), Constant)
        assert expr2.get_rhs().get_value() == 6

    def test_floor_div_by_one(self):
        d0 = Dim(0)
        expr = d0.floor_div(1)

        # Should simplify to just d0
        assert expr == d0

    def test_constant_floor_div(self):
        # 7 floordiv 3 should be 2
        expr = Constant(7).floor_div(3)
        assert isinstance(expr, Constant)
        assert expr.get_value() == 2

    def test_constant_mod(self):
        # 17 mod 5 should be 2
        expr = Constant(17) % 5
        assert isinstance(expr, Constant)
        assert expr.get_value() == 2

    def test_mod_of_multiple(self):
        # (d0 * 8) mod 4 should be 0 since 8 is multiple of 4
        d0 = Dim(0)
        expr = (d0 * 8) % 4
        assert isinstance(expr, Constant)
        assert expr.get_value() == 0


class TestProperties:

    def test_is_constant(self):
        d0 = Dim(0)
        c5 = Constant(5)
        assert not d0.is_constant()
        assert c5.is_constant()

    def test_is_pure_affine(self):
        d0 = Dim(0)

        # Basic expressions are pure affine
        assert d0.is_pure_affine()
        assert Constant(5).is_pure_affine()

        # Addition is pure affine
        assert (d0 + d0).is_pure_affine()

        # Multiplication with constant is pure affine
        assert (d0 * 3).is_pure_affine()

        # Multiplication without constant is not pure affine
        assert not (d0 * d0).is_pure_affine()

        # Division by constant is pure affine
        assert (d0.floor_div(2)).is_pure_affine()

    def test_get_largest_known_divisor(self):
        d0 = Dim(0)

        # Dimensions have divisor 1
        assert d0.get_largest_known_divisor() == 1

        # get_constants have their absolute value as divisor
        assert Constant(12).get_largest_known_divisor() == 12
        assert Constant(-8).get_largest_known_divisor() == 8

        # Multiplication multiplies divisors
        expr = (d0 * 6) * 4  # Should have divisor 24
        assert expr.get_largest_known_divisor() == 24

    def test_is_multiple_of(self):
        d0 = Dim(0)

        # Only factor 1 works for dimensions (factor * factor == 1)
        assert d0.is_multiple_of(1)
        assert not d0.is_multiple_of(2)

        # get_constants work normally
        c12 = Constant(12)
        assert c12.is_multiple_of(3)
        assert c12.is_multiple_of(4)
        assert not c12.is_multiple_of(5)

    def test_is_function_of_dim(self):
        d0 = Dim(0)
        d1 = Dim(1)

        assert d0.is_function_of_dim(0)
        assert not d0.is_function_of_dim(1)

        # Binary expression
        expr = d0 + d1
        assert expr.is_function_of_dim(0)
        assert expr.is_function_of_dim(1)
        assert not expr.is_function_of_dim(2)


class TestReplacement:

    def test_replace_dims(self):
        d0 = Dim(0)
        d1 = Dim(1)

        expr = d0 + d1 * 2
        new_expr = expr.replace_dims([d0, Constant(5)])
        expected = d0 + 10
        assert str(new_expr) == str(expected)

        new_expr = expr.replace({d0: Constant(5)})
        expected = 5 + d1 * 2
        assert str(new_expr) == str(expected)


class TestUtilityFunctions:

    def test_bind_dims(self):
        dims = bind_dims("i", "j", "k")

        assert len(dims) == 3
        assert all(isinstance(d, Dim) for d in dims)
        assert dims[0].get_position() == 0
        assert dims[1].get_position() == 1
        assert dims[2].get_position() == 2

    def test_get_constants(self):
        constants = get_constants([1, 2, 3])

        assert len(constants) == 3
        assert all(isinstance(c, Constant) for c in constants)
        assert constants[0].get_value() == 1
        assert constants[1].get_value() == 2
        assert constants[2].get_value() == 3


class TestComplexExpressions:

    def test_mixed_expression(self):
        # Build: 2*d0 + 3*d1 - 5
        d0, d1 = bind_dims("i", "j")

        expr = 2 * d0 + 3 * d1 - 5

        # Should be pure affine
        assert expr.is_pure_affine()

        # Should be function of both dimensions
        assert expr.is_function_of_dim(0)
        assert expr.is_function_of_dim(1)

    def test_division_expression(self):
        d0 = Dim(0)

        # (d0 + 7).floor_div(3)
        expr = (d0 + 7).floor_div(3)

        assert expr.get_kind() == AffineExprKind.FLOOR_DIV
        assert expr.is_pure_affine()

    def test_nested_operations(self):
        d0 = Dim(0)

        # ((d0 * 2) + 1) % 5
        expr = ((d0 * 2) + 1) % 5

        assert expr.get_kind() == AffineExprKind.MOD
        assert expr.is_pure_affine()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

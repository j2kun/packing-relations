import pytest
from quasiaffine import (
    NamedDim,
    Dim,
    Constant,
    AffineExprKind,
    bind_dims,
    get_constants,
)


class TestAffineExprBasics:

    def test_dim_expr_creation(self):
        d0 = NamedDim("d0")
        assert d0.kind() == AffineExprKind.DIM_ID
        assert str(d0) == "d0"

    def test_constant_expr_creation(self):
        c42 = Constant(42)
        assert c42.kind() == AffineExprKind.CONSTANT
        assert c42.value == 42
        assert str(c42) == "42"
        assert c42 == Constant(42)

    def test_equality(self):
        d0a = NamedDim("d0")
        d0b = NamedDim("d0")
        d1 = NamedDim("d1")
        assert d0a == d0b
        assert d0a != d1


class TestArithmeticOperations:

    def test_addition(self):
        d0 = NamedDim("d0")
        d1 = NamedDim("d1")

        expr = d0 + d1
        assert expr.kind() == AffineExprKind.ADD
        assert expr.lhs == d0
        assert expr.rhs == d1

    def test_addition_with_int(self):
        d0 = NamedDim("d0")
        expr = d0 + 5

        assert expr.kind() == AffineExprKind.ADD
        assert expr.lhs == d0
        assert expr.rhs == Constant(5)

    def test_multiplication(self):
        d0 = NamedDim("d0")
        expr = d0 * 3

        assert expr.kind() == AffineExprKind.MUL
        assert expr.lhs == d0
        assert expr.rhs == Constant(3)

    def test_subtraction(self):
        d0 = NamedDim("d0")
        d1 = NamedDim("d1")

        expr = d0 - d1
        # Subtraction becomes addition with negation
        assert expr.kind() == AffineExprKind.ADD

        # Check that d1 was negated (multiplied by -1)
        rhs = expr.rhs
        assert rhs.kind() == AffineExprKind.MUL

    def test_floor_div(self):
        d0 = NamedDim("d0")
        expr = d0.floor_div(4)

        assert expr.kind() == AffineExprKind.FLOOR_DIV
        assert expr.lhs == d0
        assert expr.rhs == Constant(4)

    def test_ceil_div(self):
        d0 = NamedDim("d0")
        expr = d0.ceil_div(3)

        assert expr.kind() == AffineExprKind.CEIL_DIV
        assert expr.lhs == d0
        assert expr.rhs == Constant(3)

    def test_modulo(self):
        d0 = NamedDim("d0")
        expr = d0 % 8

        assert expr.kind() == AffineExprKind.MOD
        assert expr.lhs == d0
        assert expr.rhs == Constant(8)


class TestSimplification:

    def test_constant_addition(self):
        # 3 + 5 should simplify to 8
        expr = Constant(3) + Constant(5)
        assert isinstance(expr, Constant)
        assert expr.value == 8

    def test_constant_multiplication(self):
        # 4 * 6 should simplify to 24
        expr = Constant(4) * Constant(6)
        assert isinstance(expr, Constant)
        assert expr.value == 24

    def test_addition_with_zero(self):
        d0 = NamedDim("d0")
        expr = d0 + 0

        # Should simplify to just d0
        assert expr == d0

    def test_multiplication_by_one(self):
        d0 = NamedDim("d0")
        expr = d0 * 1

        # Should simplify to just d0
        assert expr == d0

    def test_multiplication_by_zero(self):
        d0 = NamedDim("d0")
        expr = d0 * 0

        # Should simplify to constant 0
        assert isinstance(expr, Constant)
        assert expr.value == 0

    def test_canonicalization_constant_left(self):
        # 5 + d0 should become d0 + 5
        d0 = NamedDim("d0")
        expr = Constant(5) + d0

        assert expr.lhs == d0
        assert expr.rhs == Constant(5)

    def test_successive_additions(self):
        # (d0 + 2) + 3 should become d0 + 5
        d0 = NamedDim("d0")
        expr1 = d0 + 2
        expr2 = expr1 + 3

        assert expr2.lhs == d0
        assert isinstance(expr2.rhs, Constant)
        assert expr2.rhs.value == 5

    def test_successive_multiplications(self):
        # (d0 * 2) * 3 should become d0 * 6
        d0 = NamedDim("d0")
        expr1 = d0 * 2
        expr2 = expr1 * 3

        assert expr2.lhs == d0
        assert isinstance(expr2.rhs, Constant)
        assert expr2.rhs.value == 6

    def test_floor_div_by_one(self):
        d0 = NamedDim("d0")
        expr = d0.floor_div(1)

        # Should simplify to just d0
        assert expr == d0

    def test_constant_floor_div(self):
        # 7 floordiv 3 should be 2
        expr = Constant(7).floor_div(3)
        assert isinstance(expr, Constant)
        assert expr.value == 2

    def test_constant_mod(self):
        # 17 mod 5 should be 2
        expr = Constant(17) % 5
        assert isinstance(expr, Constant)
        assert expr.value == 2


class TestProperties:

    def test_is_constant(self):
        d0 = NamedDim("d0")
        c5 = Constant(5)
        assert not d0.is_constant()
        assert c5.is_constant()

    def test_is_pure_affine(self):
        d0 = NamedDim("d0")

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


class TestReplacement:

    def test_replace_dims(self):
        d0 = NamedDim("d0")
        d1 = NamedDim("d1")

        expr = d0 + d1 * 2
        new_expr = expr.replace_dims({d0: Constant(5)})
        expected = 5 + d1 * 2
        assert str(new_expr) == str(expected)


class TestUtilityFunctions:

    def test_bind_dims(self):
        dims = bind_dims("i", "j", "k")

        assert len(dims) == 3
        assert all(isinstance(d, Dim) for d in dims)
        assert dims[0].name == "i"
        assert dims[1].name == "j"
        assert dims[2].name == "k"

    def test_get_constants(self):
        constants = get_constants([1, 2, 3])

        assert len(constants) == 3
        assert all(isinstance(c, Constant) for c in constants)
        assert constants[0].value == 1
        assert constants[1].value == 2
        assert constants[2].value == 3


class TestComplexExpressions:

    def test_division_expression(self):
        d0 = NamedDim("d0")
        expr = (d0 + 7).floor_div(3)
        assert expr.kind() == AffineExprKind.FLOOR_DIV
        assert expr.is_pure_affine()

    def test_nested_operations(self):
        d0 = NamedDim("d0")
        expr = ((d0 * 2) + 1) % 5
        assert expr.kind() == AffineExprKind.MOD
        assert expr.is_pure_affine()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

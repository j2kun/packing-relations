"A data structure mirroring AffineExpr in MLIR." ""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Optional
import math


class AffineExprKind(Enum):
    ADD = "Add"
    MUL = "Mul"
    MOD = "Mod"
    FLOOR_DIV = "FloorDiv"
    CEIL_DIV = "CeilDiv"
    CONSTANT = "Constant"
    DIM_ID = "DimId"


class AffineExpr(ABC):

    @abstractmethod
    def get_kind(self) -> AffineExprKind:
        pass

    def __eq__(self, other) -> bool:
        if isinstance(other, int):
            return isinstance(self, Constant) and self.get_value() == other
        return (
            isinstance(other, AffineExpr)
            and self.get_kind() == other.get_kind()
            and self._equals_impl(other)
        )

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return True

    @abstractmethod
    def _equals_impl(self, other) -> bool:
        pass

    def is_constant(self) -> bool:
        kind = self.get_kind()
        if kind == AffineExprKind.CONSTANT:
            return True
        elif kind == AffineExprKind.DIM_ID:
            return False
        elif kind in [
            AffineExprKind.ADD,
            AffineExprKind.MUL,
            AffineExprKind.FLOOR_DIV,
            AffineExprKind.CEIL_DIV,
            AffineExprKind.MOD,
        ]:
            bin_expr = self
            return bin_expr.get_lhs().is_constant() and bin_expr.get_rhs().is_constant()
        return False

    def is_pure_affine(self) -> bool:
        kind = self.get_kind()
        if kind in [
            AffineExprKind.DIM_ID,
            AffineExprKind.CONSTANT,
        ]:
            return True
        elif kind == AffineExprKind.ADD:
            return self.get_lhs().is_pure_affine() and self.get_rhs().is_pure_affine()
        elif kind == AffineExprKind.MUL:
            return (
                self.get_lhs().is_pure_affine()
                and self.get_rhs().is_pure_affine()
                and (
                    isinstance(self.get_lhs(), Constant)
                    or isinstance(self.get_rhs(), Constant)
                )
            )
        elif kind in [
            AffineExprKind.FLOOR_DIV,
            AffineExprKind.CEIL_DIV,
            AffineExprKind.MOD,
        ]:
            return self.get_lhs().is_pure_affine() and isinstance(
                self.get_rhs(), Constant
            )
        return False

    def get_largest_known_divisor(self) -> int:
        kind = self.get_kind()
        if kind in [AffineExprKind.DIM_ID]:
            return 1
        elif kind == AffineExprKind.CONSTANT:
            return abs(self.get_value())
        elif kind == AffineExprKind.MUL:
            return (
                self.get_lhs().get_largest_known_divisor()
                * self.get_rhs().get_largest_known_divisor()
            )
        elif kind in [AffineExprKind.ADD, AffineExprKind.MOD]:
            return math.gcd(
                self.get_lhs().get_largest_known_divisor(),
                self.get_rhs().get_largest_known_divisor(),
            )
        elif kind in [AffineExprKind.FLOOR_DIV, AffineExprKind.CEIL_DIV]:
            if isinstance(self.get_rhs(), Constant):
                rhs_val = self.get_rhs().get_value()
                if rhs_val != 0:
                    lhs_div = self.get_lhs().get_largest_known_divisor()
                    if lhs_div % rhs_val == 0:
                        return abs(lhs_div // rhs_val)
            return 1
        return 1

    def is_multiple_of(self, factor: int) -> bool:
        kind = self.get_kind()
        if kind in [AffineExprKind.DIM_ID]:
            return factor * factor == 1
        elif kind == AffineExprKind.CONSTANT:
            return self.get_value() % factor == 0
        elif kind == AffineExprKind.MUL:
            l = self.get_lhs().get_largest_known_divisor()
            r = self.get_rhs().get_largest_known_divisor()
            return l % factor == 0 or r % factor == 0 or (l * r) % factor == 0
        else:
            return (
                math.gcd(
                    self.get_lhs().get_largest_known_divisor(),
                    self.get_rhs().get_largest_known_divisor(),
                )
                % factor
                == 0
            )

    def is_function_of_dim(self, position: int) -> bool:
        if self.get_kind() == AffineExprKind.DIM_ID:
            return self.get_position() == position
        elif isinstance(self, AffineBinaryOpExpr):
            return self.get_lhs().is_function_of_dim(
                position
            ) or self.get_rhs().is_function_of_dim(position)
        return False

    def __add__(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_add(self, other) or AffineBinaryOpExpr(
            AffineExprKind.ADD, self, other
        )

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_mul(self, other) or AffineBinaryOpExpr(
            AffineExprKind.MUL, self, other
        )

    def __rmul__(self, other):
        return self.__mul__(other)

    def __neg__(self):
        return self * Constant(-1)

    def __sub__(self, other):
        if isinstance(other, int):
            return self + (-other)
        return self + (-other)

    def __rsub__(self, other):
        if isinstance(other, int):
            return Constant(other) - self
        return other - self

    def floor_div(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_floor_div(self, other) or AffineBinaryOpExpr(
            AffineExprKind.FLOOR_DIV, self, other
        )

    def ceil_div(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_ceil_div(self, other) or AffineBinaryOpExpr(
            AffineExprKind.CEIL_DIV, self, other
        )

    def __mod__(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_mod(self, other) or AffineBinaryOpExpr(
            AffineExprKind.MOD, self, other
        )

    def replace_dims(self, dim_replacements: List["AffineExpr"]) -> "AffineExpr":
        kind = self.get_kind()
        if kind == AffineExprKind.CONSTANT:
            return self
        elif kind == AffineExprKind.DIM_ID:
            pos = self.get_position()
            if pos < len(dim_replacements):
                return dim_replacements[pos]
            return self
        elif isinstance(self, AffineBinaryOpExpr):
            new_lhs = self.get_lhs().replace_dims(dim_replacements)
            new_rhs = self.get_rhs().replace_dims(dim_replacements)
            if new_lhs == self.get_lhs() and new_rhs == self.get_rhs():
                return self
            return get_affine_binary_op_expr(kind, new_lhs, new_rhs)
        return self

    def replace(self, expr_map: Dict["AffineExpr", "AffineExpr"]) -> "AffineExpr":
        if isinstance(self, Dim) and self in expr_map:
            return expr_map[self]

        if isinstance(self, AffineBinaryOpExpr):
            new_lhs = self.get_lhs().replace(expr_map)
            new_rhs = self.get_rhs().replace(expr_map)
            if new_lhs == self.get_lhs() and new_rhs == self.get_rhs():
                return self
            return get_affine_binary_op_expr(self.get_kind(), new_lhs, new_rhs)

        return self

    def __str__(self) -> str:
        return self._to_string()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._to_string()})"

    @abstractmethod
    def _to_string(self) -> str:
        pass


class AffineBinaryOpExpr(AffineExpr):

    def __init__(self, kind: AffineExprKind, lhs: AffineExpr, rhs: AffineExpr):
        self._kind = kind
        self._lhs = lhs
        self._rhs = rhs

    def get_kind(self) -> AffineExprKind:
        return self._kind

    def get_lhs(self) -> AffineExpr:
        return self._lhs

    def get_rhs(self) -> AffineExpr:
        return self._rhs

    def _equals_impl(self, other) -> bool:
        if not isinstance(other, AffineBinaryOpExpr):
            return False
        return self._lhs == other._lhs and self._rhs == other._rhs

    def _to_string(self) -> str:
        op_str = {
            AffineExprKind.ADD: " + ",
            AffineExprKind.MUL: " * ",
            AffineExprKind.FLOOR_DIV: " floordiv ",
            AffineExprKind.CEIL_DIV: " ceildiv ",
            AffineExprKind.MOD: " mod ",
        }
        return f"({self._lhs._to_string()}{op_str[self._kind]}{self._rhs._to_string()})"


class Dim(AffineExpr):

    def __init__(self, position: int):
        self._position = position

    def get_kind(self) -> AffineExprKind:
        return AffineExprKind.DIM_ID

    def get_position(self) -> int:
        return self._position

    def _equals_impl(self, other) -> bool:
        if not isinstance(other, Dim):
            return False
        return self._position == other._position

    def _to_string(self) -> str:
        return f"d{self._position}"

    def __hash__(self):
        return hash(self._to_string())


class Constant(AffineExpr):

    def __init__(self, value: int):
        self._value = value

    def get_kind(self) -> AffineExprKind:
        return AffineExprKind.CONSTANT

    def get_value(self) -> int:
        return self._value

    def _equals_impl(self, other) -> bool:
        if not isinstance(other, Constant):
            return False
        return self._value == other._value

    def _to_string(self) -> str:
        return str(self._value)

    def __hash__(self):
        return hash(self._value)


def get_affine_binary_op_expr(
    kind: AffineExprKind, lhs: AffineExpr, rhs: AffineExpr
) -> AffineBinaryOpExpr:
    if kind == AffineExprKind.ADD:
        return lhs + rhs
    elif kind == AffineExprKind.MUL:
        return lhs * rhs
    elif kind == AffineExprKind.FLOOR_DIV:
        return lhs.floor_div(rhs)
    elif kind == AffineExprKind.CEIL_DIV:
        return lhs.ceil_div(rhs)
    elif kind == AffineExprKind.MOD:
        return lhs % rhs
    else:
        raise ValueError(f"Unknown binary operation: {kind}")


def _simplify_add(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    # Both constants
    if isinstance(lhs, Constant) and isinstance(rhs, Constant):
        return Constant(lhs.get_value() + rhs.get_value())

    # Canonicalize: constant on right (4 + d0 becomes d0 + 4)
    if lhs.is_constant() and not rhs.is_constant():
        return _simplify_add(rhs, lhs) or AffineBinaryOpExpr(
            AffineExprKind.ADD, rhs, lhs
        )

    # Addition with zero
    if isinstance(rhs, Constant) and rhs.get_value() == 0:
        return lhs

    # Fold successive additions: (d0 + 2) + 3 = d0 + 5
    if isinstance(lhs, AffineBinaryOpExpr) and lhs.get_kind() == AffineExprKind.ADD:
        if isinstance(rhs, Constant) and isinstance(lhs.get_rhs(), Constant):
            return lhs.get_lhs() + (lhs.get_rhs().get_value() + rhs.get_value())

    return None


def _simplify_mul(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if isinstance(lhs, Constant) and isinstance(rhs, Constant):
        return Constant(lhs.get_value() * rhs.get_value())

    # Canonicalize: constant on right
    if lhs.is_constant() and not rhs.is_constant():
        return _simplify_mul(rhs, lhs) or AffineBinaryOpExpr(
            AffineExprKind.Mul, rhs, lhs
        )

    if isinstance(rhs, Constant):
        if rhs.get_value() == 1:
            return lhs
        if rhs.get_value() == 0:
            return rhs

    # Fold successive multiplications: (d0 * 2) * 3 = d0 * 6
    if isinstance(lhs, AffineBinaryOpExpr) and lhs.get_kind() == AffineExprKind.MUL:
        if isinstance(rhs, Constant) and isinstance(lhs.get_rhs(), Constant):
            return lhs.get_lhs() * (lhs.get_rhs().get_value() * rhs.get_value())

    return None


def _simplify_floor_div(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if not isinstance(rhs, Constant) or rhs.get_value() == 0:
        return None

    rhs_val = rhs.get_value()

    if isinstance(lhs, Constant):
        return Constant(lhs.get_value() // rhs_val)

    if rhs_val == 1:
        return lhs

    # Simplify (expr * c) floordiv d when c is multiple of d
    if isinstance(lhs, AffineBinaryOpExpr) and lhs.get_kind() == AffineExprKind.MUL:
        if isinstance(lhs.get_rhs(), Constant):
            lhs_val = lhs.get_rhs().get_value()
            if lhs_val % rhs_val == 0:
                return lhs.get_lhs() * (lhs_val // rhs_val)

    return None


def _simplify_ceil_div(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if not isinstance(rhs, Constant) or rhs.get_value() == 0:
        return None

    rhs_val = rhs.get_value()

    if isinstance(lhs, Constant):
        lhs_val = lhs.get_value()
        # Ceiling division: (a + b - 1) // b for positive numbers
        result = (
            (lhs_val + rhs_val - 1) // rhs_val if lhs_val >= 0 else lhs_val // rhs_val
        )
        return Constant(result)

    if rhs_val == 1:
        return lhs

    if isinstance(lhs, AffineBinaryOpExpr) and lhs.get_kind() == AffineExprKind.MUL:
        if isinstance(lhs.get_rhs(), Constant):
            lhs_val = lhs.get_rhs().get_value()
            if lhs_val % rhs_val == 0:
                return lhs.get_lhs() * (lhs_val // rhs_val)

    return None


def _simplify_mod(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if not isinstance(rhs, Constant) or rhs.get_value() < 1:
        return None

    rhs_val = rhs.get_value()

    if isinstance(lhs, Constant):
        return Constant(lhs.get_value() % rhs_val)

    # Expression is multiple of modulo factor
    if lhs.get_largest_known_divisor() % rhs_val == 0:
        return Constant(0)

    return None


def bind_dims(*names) -> List[Dim]:
    return [Dim(i) for i in range(len(names))]


def get_constants(constants: List[int]) -> List[Constant]:
    return [Constant(c) for c in constants]


if __name__ == "__main__":
    d0 = Dim(0)
    d1 = Dim(1)
    c5 = Constant(5)

    expr1 = d0 + d1
    print(f"d0 + d1 = {expr1}")

    expr3 = (d0 + 4).floor_div(2)
    print(f"(d0 + 4) floordiv 2 = {expr3}")

    expr4 = d0 % 8
    print(f"d0 mod 8 = {expr4}")

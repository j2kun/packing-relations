"A data structure mirroring AffineExpr in MLIR." ""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, TypeVar, Callable


T = TypeVar("T")


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
    def kind(self) -> AffineExprKind:
        pass

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __bool__(self) -> bool:
        return True

    def reduce(self, fn: Callable[["AffineExpr", list[T], T], T], accum: T) -> T:
        match self:
            case Constant() | Dim():
                return fn(self, [], accum)
            case _:
                ts = [self.lhs.reduce(fn, accum), self.rhs.reduce(fn, accum)]
                return fn(self, ts, accum)

    def is_constant(self) -> bool:
        def reducer(expr, ts, accum):
            match expr:
                case Constant():
                    return True
                case Dim():
                    return False
                case _:
                    return accum and all(ts)

        return self.reduce(reducer, True)

    def is_pure_affine(self) -> bool:
        def reducer(expr, ts, accum):
            match expr:
                case Constant() | Dim():
                    return True
                case Add() | Binary(rhs=Constant()):
                    return accum and all(ts)
                case _:
                    return False

        return self.reduce(reducer, True)

    def replace_dims(self, dim_replacements: dict["Dim", "AffineExpr"]) -> "AffineExpr":
        def reducer(expr, ts, accum):
            match expr:
                case Constant():
                    return expr
                case Dim():
                    return dim_replacements.get(expr, expr)
                case Binary():
                    return type(expr)(*ts)
                case _:
                    raise ValueError(f"Unknown expression type: {expr}")

        return self.reduce(reducer, True).simplify()

    def is_function_of_dims(self, dims: set["Dim"]) -> "AffineExpr":
        def reducer(expr, ts, accum):
            match expr:
                case Constant():
                    return True
                case Dim():
                    return expr in dims
                case _:
                    return accum and all(ts)

        return self.reduce(reducer, True)

    def simplify(self) -> "AffineExpr":
        def reducer(expr, ts, accum):
            match expr:
                case Constant() | Dim():
                    return expr
                case Add():
                    return _simplify_add(*ts) or Add(*ts)
                case Mul():
                    return _simplify_mul(*ts) or Mul(*ts)
                case CeilDiv():
                    return _simplify_ceil_div(*ts) or CeilDiv(*ts)
                case FloorDiv():
                    return _simplify_floor_div(*ts) or FloorDiv(*ts)
                case Mod():
                    return _simplify_mod(*ts) or Mod(*ts)
                case _:
                    raise ValueError(f"Unknown expression type: {expr}")

        return self.reduce(reducer, True)

    def __add__(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_add(self, other) or Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_mul(self, other) or Mul(self, other)

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
        return _simplify_floor_div(self, other) or FloorDiv(self, other)

    def ceil_div(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_ceil_div(self, other) or CeilDiv(self, other)

    def __mod__(self, other):
        if isinstance(other, int):
            other = Constant(other)
        return _simplify_mod(self, other) or Mod(self, other)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self})"


class Dim(AffineExpr):
    def kind(self) -> AffineExprKind:
        return AffineExprKind.DIM_ID


class NamedDim(Dim):
    def __init__(self, name: str):
        self.name = name

    def __eq__(self, other) -> bool:
        return isinstance(other, NamedDim) and self.name == other.name

    def __str__(self) -> str:
        return f"{self.name}"

    def __hash__(self):
        return hash(self.name)


class UniqueDim(Dim):
    # static id for all UniqueDims
    _id_counter = 0

    def __init__(self):
        self._id = UniqueDim._id_counter
        UniqueDim._id_counter += 1

    def __eq__(self, other) -> bool:
        return isinstance(other, UniqueDim) and self._id == other._id

    def __str__(self) -> str:
        return f"d{self._id}"

    def __hash__(self):
        return hash(self._id)


class Constant(AffineExpr):

    def __init__(self, value: int):
        self.value = value

    def kind(self) -> AffineExprKind:
        return AffineExprKind.CONSTANT

    def __eq__(self, other) -> bool:
        return isinstance(other, Constant) and self.value == other.value

    def __str__(self) -> str:
        return str(self.value)

    def __hash__(self):
        return hash(self.value)


class Binary(AffineExpr):

    def __init__(self, lhs: AffineExpr, rhs: AffineExpr):
        self.lhs = lhs
        self.rhs = rhs

    def __eq__(self, other) -> bool:
        return (
            isinstance(other, Binary)
            and self.kind() == other.kind()
            and self.lhs == other.lhs
            and self.rhs == other.rhs
        )

    def __str__(self) -> str:
        op_str = {
            AffineExprKind.ADD: " + ",
            AffineExprKind.MUL: " * ",
            AffineExprKind.FLOOR_DIV: " floordiv ",
            AffineExprKind.CEIL_DIV: " ceildiv ",
            AffineExprKind.MOD: " mod ",
        }
        return f"({self.lhs}{op_str[self.kind()]}{self.rhs})"


class Add(Binary):
    def kind(self) -> AffineExprKind:
        return AffineExprKind.ADD


class Mul(Binary):
    def kind(self) -> AffineExprKind:
        return AffineExprKind.MUL


class Mod(Binary):
    def kind(self) -> AffineExprKind:
        return AffineExprKind.MOD


class CeilDiv(Binary):
    def kind(self) -> AffineExprKind:
        return AffineExprKind.CEIL_DIV


class FloorDiv(Binary):
    def kind(self) -> AffineExprKind:
        return AffineExprKind.FLOOR_DIV


def _simplify_add(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    # Both constants
    if isinstance(lhs, Constant) and isinstance(rhs, Constant):
        return Constant(lhs.value + rhs.value)

    # Canonicalize: constant on right (4 + d0 becomes d0 + 4)
    if lhs.is_constant() and not rhs.is_constant():
        return _simplify_add(rhs, lhs) or Add(rhs, lhs)

    # Addition with zero
    if isinstance(rhs, Constant) and rhs.value == 0:
        return lhs

    # Fold successive additions: (d0 + 2) + 3 = d0 + 5
    if isinstance(lhs, Binary) and lhs.kind() == AffineExprKind.ADD:
        if isinstance(rhs, Constant) and isinstance(lhs.rhs, Constant):
            return lhs.lhs + (lhs.rhs.value + rhs.value)

    return None


def _simplify_mul(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if isinstance(lhs, Constant) and isinstance(rhs, Constant):
        return Constant(lhs.value * rhs.value)

    # Canonicalize: constant on right
    if lhs.is_constant() and not rhs.is_constant():
        return _simplify_mul(rhs, lhs) or Binary(AffineExprKind.Mul, rhs, lhs)

    if isinstance(rhs, Constant):
        if rhs.value == 1:
            return lhs
        if rhs.value == 0:
            return rhs

    # Fold successive multiplications: (d0 * 2) * 3 = d0 * 6
    if isinstance(lhs, Binary) and lhs.kind() == AffineExprKind.MUL:
        if isinstance(rhs, Constant) and isinstance(lhs.rhs, Constant):
            return lhs.lhs * (lhs.rhs.value * rhs.value)

    return None


def _simplify_floor_div(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if not isinstance(rhs, Constant) or rhs.value == 0:
        return None

    rhs_val = rhs.value

    if isinstance(lhs, Constant):
        return Constant(lhs.value // rhs_val)

    if rhs_val == 1:
        return lhs

    # Simplify (expr * c) floordiv d when c is multiple of d
    if isinstance(lhs, Binary) and lhs.kind() == AffineExprKind.MUL:
        if isinstance(lhs.rhs, Constant):
            lhs_val = lhs.rhs.value
            if lhs_val % rhs_val == 0:
                return lhs.lhs * (lhs_val // rhs_val)

    return None


def _simplify_ceil_div(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if not isinstance(rhs, Constant) or rhs.value == 0:
        return None

    rhs_val = rhs.value

    if isinstance(lhs, Constant):
        lhs_val = lhs.value
        # Ceiling division: (a + b - 1) // b for positive numbers
        result = (
            (lhs_val + rhs_val - 1) // rhs_val if lhs_val >= 0 else lhs_val // rhs_val
        )
        return Constant(result)

    if rhs_val == 1:
        return lhs

    if isinstance(lhs, Binary) and lhs.kind() == AffineExprKind.MUL:
        if isinstance(lhs.rhs, Constant):
            lhs_val = lhs.rhs.value
            if lhs_val % rhs_val == 0:
                return lhs.lhs * (lhs_val // rhs_val)

    return None


def _simplify_mod(lhs: AffineExpr, rhs: AffineExpr) -> Optional[AffineExpr]:
    if not isinstance(rhs, Constant) or rhs.value < 1:
        return None

    rhs_val = rhs.value

    if isinstance(lhs, Constant):
        return Constant(lhs.value % rhs_val)

    return None


def bind_dims(*names) -> list[Dim]:
    return [NamedDim(name) for name in names]


def bind_unique_dims(count) -> list[Dim]:
    return [UniqueDim() for _ in range(count)]


def get_constants(constants: list[int]) -> list[Constant]:
    return [Constant(c) for c in constants]


if __name__ == "__main__":
    d0, d1 = bind_unique_dims(2)
    c5 = Constant(5)

    expr1 = d0 + d1
    print(f"d0 + d1 = {expr1}")

    expr3 = (d0 + 4).floor_div(2)
    print(f"(d0 + 4) floordiv 2 = {expr3}")

    expr4 = d0 % 8
    print(f"d0 mod 8 = {expr4}")

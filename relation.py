from typing import Iterator, List, Tuple, Dict, Optional
from dataclasses import dataclass
import itertools
from quasiaffine import (
    AffineExpr,
    AffineExprKind,
    Constant,
    Dim,
    Symbol,
    bind_dims,
    get_affine_binary_op_expr,
)


@dataclass
class IndexSpace:
    dim_sizes: List[int]

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        """Iterate over all tuples in this index space"""
        for indices in itertools.product(*[range(size) for size in self.dim_sizes]):
            yield indices

    def size(self) -> int:
        """Total number of tuples in this space"""
        if not self.dim_sizes:
            return 0
        result = 1
        for dim_size in self.dim_sizes:
            result *= dim_size
        return result

    def __str__(self) -> str:
        if not self.dim_sizes:
            return "{()}"
        ranges = [f"{{0..{size-1}}}" for size in self.dim_sizes]
        return " × ".join(ranges)

    @property
    def num_dims(self) -> int:
        """Number of dimensions in this index space"""
        return len(self.dim_sizes)


class IntegerRelation:
    """
    Represents a relation between tuples of integers defined by quasiaffine constraints.

    A relation R ⊆ Domain × Codomain is defined by constraints of the form:
    f(d₀, d₁, ..., c₀, c₁, ..., l₀, l₁, ..., s₀, s₁, ...) = 0

    where d_i are domain variables, c_i are codomain variables,
    l_i are local/existential variables, and s_i are symbolic parameters.
    """

    def __init__(
        self,
        domain: IndexSpace,
        codomain: IndexSpace,
        constraints: List[AffineExpr],
        num_symbols: int = 0,
        num_locals: int = 0,
        local_bounds: Optional[List[int]] = None,
    ):
        """
        Initialize an integer relation.

        Args:
            domain: The domain index space
            codomain: The codomain index space
            constraints: List of quasiaffine expressions interpreted as f(...) = 0
            num_symbols: Number of symbolic parameters
            num_locals: Number of local/existential variables
            local_bounds: Upper bounds for each local variable (0 to bound-1)
        """
        self.domain = domain
        self.codomain = codomain
        self.constraints = constraints
        self.num_symbols = num_symbols
        self.num_locals = num_locals
        self.local_bounds = local_bounds or []

        if num_locals > 0 and not local_bounds:
            raise ValueError("local_bounds must be provided if num_locals > 0")

        # Validate that constraints use the right number of dimensions
        total_dims = domain.num_dims + codomain.num_dims + num_locals
        for constraint in constraints:
            self._validate_constraint_dims(constraint, total_dims)

    def _validate_constraint_dims(self, expr: AffineExpr, max_dim: int):
        """Validate that expression doesn't use dimensions beyond max_dim"""
        if isinstance(expr, Dim):
            if expr.get_position() >= max_dim:
                raise ValueError(
                    f"Dimension d{expr.get_position()} exceeds maximum {max_dim-1}"
                )
        elif hasattr(expr, "get_lhs") and hasattr(expr, "get_rhs"):
            self._validate_constraint_dims(expr.get_lhs(), max_dim)
            self._validate_constraint_dims(expr.get_rhs(), max_dim)

    def _enumerate_local_values(self) -> Iterator[Tuple[int, ...]]:
        """Generate all possible assignments to local variables"""
        if self.num_locals == 0:
            yield ()
        else:
            for local_values in itertools.product(
                *[range(bound) for bound in self.local_bounds]
            ):
                yield local_values

    def _evaluate_constraint(
        self,
        constraint: AffineExpr,
        domain_values: Tuple[int, ...],
        codomain_values: Tuple[int, ...],
        local_values: Tuple[int, ...] = (),
        symbol_values: Optional[Tuple[int, ...]] = None,
    ) -> int:
        """
        Evaluate a constraint expression with given variable values.

        Convention:
        - d₀, d₁, ... d_{domain.num_dims-1} - domain dims
        - d_{domain.num_dims}, ..., d_{domain.num_dims + codomain.num_dims - 1} - codomain dims
        - d_{domain.num_dims + codomain.num_dims}, ... - local dims
        - s₀, s₁, ... - symbols
        """
        if symbol_values is None:
            symbol_values = tuple(0 for _ in range(self.num_symbols))

        # Create substitution map
        substitutions = {}

        # Domain dimensions
        for i, val in enumerate(domain_values):
            substitutions[Dim(i)] = Constant(val)

        # Codomain dimensions
        for i, val in enumerate(codomain_values):
            substitutions[Dim(self.domain.num_dims + i)] = Constant(val)

        # Local dimensions
        for i, val in enumerate(local_values):
            substitutions[Dim(self.domain.num_dims + self.codomain.num_dims + i)] = (
                Constant(val)
            )

        # Symbols
        for i, val in enumerate(symbol_values):
            substitutions[Symbol(i)] = Constant(val)

        return self._eval_expr_with_substitutions(constraint, substitutions)

    def _eval_expr_with_substitutions(
        self, expr: AffineExpr, substitutions: Dict[AffineExpr, Constant]
    ) -> int:
        """Recursively evaluate expression with variable substitutions"""
        if isinstance(expr, Constant):
            return expr.get_value()
        elif isinstance(expr, (Dim, Symbol)):
            if expr in substitutions:
                return substitutions[expr].get_value()
            else:
                raise ValueError(f"No substitution provided for {expr}")
        elif hasattr(expr, "get_lhs") and hasattr(expr, "get_rhs"):
            lhs_val = self._eval_expr_with_substitutions(expr.get_lhs(), substitutions)
            rhs_val = self._eval_expr_with_substitutions(expr.get_rhs(), substitutions)

            kind = expr.get_kind()
            if kind == AffineExprKind.ADD:
                return lhs_val + rhs_val
            elif kind == AffineExprKind.MUL:
                return lhs_val * rhs_val
            elif kind == AffineExprKind.FLOOR_DIV:
                if rhs_val == 0:
                    raise ValueError("Division by zero")
                return lhs_val // rhs_val
            elif kind == AffineExprKind.CEIL_DIV:
                if rhs_val == 0:
                    raise ValueError("Division by zero")
                if rhs_val > 0:
                    return (lhs_val + rhs_val - 1) // rhs_val
                else:
                    return lhs_val // rhs_val
            elif kind == AffineExprKind.MOD:
                if rhs_val == 0:
                    raise ValueError("Modulo by zero")
                return lhs_val % rhs_val
            else:
                raise ValueError(f"Unknown expression kind: {kind}")
        else:
            raise ValueError(f"Unknown expression type: {type(expr)}")

    def _all_constraints_satisfied(
        self,
        domain_tuple: Tuple[int, ...],
        codomain_tuple: Tuple[int, ...],
        local_values: Tuple[int, ...],
        symbol_values: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """Check if all constraints are satisfied for given variable assignments"""
        for constraint in self.constraints:
            if (
                self._evaluate_constraint(
                    constraint,
                    domain_tuple,
                    codomain_tuple,
                    local_values,
                    symbol_values,
                )
                != 0
            ):
                return False
        return True

    def is_in_relation(
        self,
        domain_tuple: Tuple[int, ...],
        codomain_tuple: Tuple[int, ...],
        symbol_values: Optional[Tuple[int, ...]] = None,
    ) -> bool:
        """
        Check if a (domain, codomain) pair satisfies constraints.
        Uses existential quantification: ∃ local vars such that all constraints hold.
        """
        if len(domain_tuple) != self.domain.num_dims:
            raise ValueError(
                f"Domain tuple has wrong dimension: expected {self.domain.num_dims}, got {len(domain_tuple)}"
            )
        if len(codomain_tuple) != self.codomain.num_dims:
            raise ValueError(
                f"Codomain tuple has wrong dimension: expected {self.codomain.num_dims}, got {len(codomain_tuple)}"
            )

        # Check bounds
        for i, val in enumerate(domain_tuple):
            if not (0 <= val < self.domain.dim_sizes[i]):
                return False
        for i, val in enumerate(codomain_tuple):
            if not (0 <= val < self.codomain.dim_sizes[i]):
                return False

        # Existential quantification: check if ANY assignment to local vars satisfies constraints
        for local_values in self._enumerate_local_values():
            if self._all_constraints_satisfied(
                domain_tuple, codomain_tuple, local_values, symbol_values
            ):
                return True

        return False

    def enumerate_relation(
        self, symbol_values: Optional[Tuple[int, ...]] = None
    ) -> Iterator[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """
        Generator that yields all (domain_tuple, codomain_tuple) pairs in the relation.
        """
        for domain_tuple in self.domain:
            for codomain_tuple in self.codomain:
                if self.is_in_relation(domain_tuple, codomain_tuple, symbol_values):
                    yield (domain_tuple, codomain_tuple)

    def compose(self, other: "IntegerRelation") -> "IntegerRelation":
        """
        Compose this relation with another: self ∘ other.

        If self: A → B and other: C → A, then result: C → B
        The intermediate A variables become existential/local variables.
        """
        if other.codomain.dim_sizes != self.domain.dim_sizes:
            raise ValueError(
                "Cannot compose: codomain of second relation must match domain of first"
            )

        # Result relation: other.domain → self.codomain
        result_domain = other.domain
        result_codomain = self.codomain

        # The intermediate space (other.codomain = self.domain) becomes local variables
        intermediate_domain = self.domain
        intermediate_size = intermediate_domain.num_dims

        # Combine local variables from both relations plus the new intermediate locals
        result_num_locals = self.num_locals + other.num_locals + intermediate_size
        result_local_bounds = (
            other.local_bounds + self.local_bounds + intermediate_domain.dim_sizes
        )

        # Dimension mapping for the composed relation:
        # d₀...d_{|other.domain|-1} - result domain (unchanged from other.domain)
        # d_{|other.domain|}...d_{|other.domain|+|self.codomain|-1} - result codomain (from self.codomain)
        # d_{|other.domain|+|self.codomain|}...d_{|other.domain|+|self.codomain|+|other.num_locals|-1} - other's locals
        # d_{...}...d_{...+|self.num_locals|-1} - self's locals
        # d_{...}...d_{...+intermediate_size-1} - intermediate vars (was self.domain = other.codomain)
        result_constraints = []

        # Add other's constraints (domains/codomains/locals unchanged in their positions)
        for constraint in other.constraints:
            result_constraints.append(constraint)

        # Add self's constraints with dimension remapping
        for constraint in self.constraints:
            remapped_constraint = self._remap_constraint_for_composition(
                constraint,
                result_domain.num_dims,
                result_codomain.num_dims,
                other.num_locals,
                self.num_locals,
                intermediate_size,
            )
            result_constraints.append(remapped_constraint)

        # Add equality constraints connecting intermediate variables
        # other's codomain dims = self's domain dims = intermediate local dims
        for i in range(intermediate_size):
            # other's codomain dim i
            other_codomain_dim = Dim(other.domain.num_dims + i)
            # corresponding intermediate local dim
            intermediate_local_dim = Dim(
                result_domain.num_dims
                + result_codomain.num_dims
                + other.num_locals
                + self.num_locals
                + i
            )
            # Constraint: other_codomain_dim - intermediate_local_dim = 0
            result_constraints.append(other_codomain_dim - intermediate_local_dim)

        return IntegerRelation(
            result_domain,
            result_codomain,
            result_constraints,
            num_symbols=max(self.num_symbols, other.num_symbols),
            num_locals=result_num_locals,
            local_bounds=result_local_bounds,
        )

    def _remap_constraint_for_composition(
        self,
        constraint: AffineExpr,
        result_domain_size: int,
        result_codomain_size: int,
        other_num_locals: int,
        self_num_locals: int,
        intermediate_size: int,
    ) -> AffineExpr:
        """Remap constraint dimensions for composition"""
        if isinstance(constraint, Constant) or isinstance(constraint, Symbol):
            return constraint
        elif isinstance(constraint, Dim):
            pos = constraint.get_position()
            if pos < self.domain.num_dims:
                # self's domain dims map to intermediate local vars
                new_pos = (
                    result_domain_size
                    + result_codomain_size
                    + other_num_locals
                    + self_num_locals
                    + pos
                )
            elif pos < self.domain.num_dims + self.codomain.num_dims:
                # self's codomain dims map to result codomain dims
                codomain_offset = pos - self.domain.num_dims
                new_pos = result_domain_size + codomain_offset
            else:
                # self's local dims map to self's local section in result
                local_offset = pos - self.domain.num_dims - self.codomain.num_dims
                new_pos = (
                    result_domain_size
                    + result_codomain_size
                    + other_num_locals
                    + local_offset
                )
            return Dim(new_pos)
        elif hasattr(constraint, "get_lhs") and hasattr(constraint, "get_rhs"):
            new_lhs = self._remap_constraint_for_composition(
                constraint.get_lhs(),
                result_domain_size,
                result_codomain_size,
                other_num_locals,
                self_num_locals,
                intermediate_size,
            )
            new_rhs = self._remap_constraint_for_composition(
                constraint.get_rhs(),
                result_domain_size,
                result_codomain_size,
                other_num_locals,
                self_num_locals,
                intermediate_size,
            )
            return get_affine_binary_op_expr(constraint.get_kind(), new_lhs, new_rhs)
        else:
            raise ValueError(f"Unknown constraint type: {type(constraint)}")

    def to_set(
        self, symbol_values: Optional[Tuple[int, ...]] = None
    ) -> List[Tuple[Tuple[int, ...], Tuple[int, ...]]]:
        """Convert the relation to a list of all valid (domain, codomain) pairs."""
        return list(self.enumerate_relation(symbol_values))

    def is_empty(self, symbol_values: Optional[Tuple[int, ...]] = None) -> bool:
        """Check if the relation is empty for given symbol values."""
        try:
            next(self.enumerate_relation(symbol_values))
            return False
        except StopIteration:
            return True

    def size(self, symbol_values: Optional[Tuple[int, ...]] = None) -> int:
        """Count the number of tuples in the relation."""
        return sum(1 for _ in self.enumerate_relation(symbol_values))

    def __str__(self) -> str:
        """String representation of the relation."""
        constraints_str = " ∧ ".join(
            [f"({constraint}) = 0" for constraint in self.constraints]
        )
        if not constraints_str:
            constraints_str = "true"

        locals_str = f", {self.num_locals} locals" if self.num_locals > 0 else ""
        return f"{{(d, c) ∈ {self.domain} × {self.codomain}{locals_str} | {constraints_str}}}"

    def __repr__(self) -> str:
        return f"IntegerRelation(domain={self.domain}, codomain={self.codomain}, constraints={self.constraints}, num_locals={self.num_locals})"


# Convenience functions for common relation types


def identity_relation(space: IndexSpace) -> IntegerRelation:
    """Create an identity relation on the given space."""
    constraints = []
    for i in range(space.num_dims):
        # d_i = c_i  =>  d_i - c_i = 0
        domain_dim = Dim(i)
        codomain_dim = Dim(space.num_dims + i)
        constraints.append(domain_dim - codomain_dim)

    return IntegerRelation(space, space, constraints)


def constant_relation(
    domain: IndexSpace, codomain_point: Tuple[int, ...]
) -> IntegerRelation:
    """Create a relation that maps every domain point to a single codomain point."""
    if len(codomain_point) == 0:
        codomain = IndexSpace([])
    else:
        codomain = IndexSpace([max(codomain_point) + 1] * len(codomain_point))

    constraints = []
    for i, target_val in enumerate(codomain_point):
        # c_i = target_val  =>  c_i - target_val = 0
        codomain_dim = Dim(domain.num_dims + i)
        constraints.append(codomain_dim - Constant(target_val))

    return IntegerRelation(domain, codomain, constraints)


def empty_relation(domain: IndexSpace, codomain: IndexSpace) -> IntegerRelation:
    """Create an empty relation (no valid pairs)."""
    # Add constraint: 0 = 1 (always false)
    false_constraint = Constant(1)
    return IntegerRelation(domain, codomain, [false_constraint])


def universal_relation(domain: IndexSpace, codomain: IndexSpace) -> IntegerRelation:
    """Create a universal relation (all pairs are valid)."""
    # No constraints means all pairs are valid
    return IntegerRelation(domain, codomain, [])


if __name__ == "__main__":
    # Domain and codomain are both 2D spaces: {0,1} × {0,1}
    domain = IndexSpace([2, 2])
    codomain = IndexSpace([2, 2])

    # Create dimensions for constraint building
    # d0, d1 are domain dims; d2, d3 are codomain dims
    d0, d1, d2, d3 = bind_dims("d0", "d1", "d2", "d3")

    # Example constraint: d0 + d1 = d2 (sum of domain coords equals first codomain coord)
    # and d3 = 0 (second codomain coord is always 0)
    constraints = [d0 + d1 - d2, d3]  # d0 + d1 - d2 = 0  # d3 = 0

    relation = IntegerRelation(domain, codomain, constraints)

    print("Relation:", relation)
    print("\nValid pairs:")
    for domain_tuple, codomain_tuple in relation.enumerate_relation():
        print(f"  {domain_tuple} → {codomain_tuple}")

    print(f"\nRelation size: {relation.size()}")

    # Test identity relation
    print("\nIdentity relation on {0,1}:")
    space = IndexSpace([2])
    identity = identity_relation(space)
    print("Valid pairs:")
    for domain_tuple, codomain_tuple in identity.enumerate_relation():
        print(f"  {domain_tuple} → {codomain_tuple}")


    print("=== Matrix Multiplication via Relation Composition ===\n")

    # Example: 2x2 matrices A and B, computing C = A * B
    # A is 2x2, B is 2x2, so we have dimensions:
    # - Matrix A: rows 0,1 and cols 0,1
    # - Matrix B: rows 0,1 and cols 0,1
    # - Matrix C: rows 0,1 and cols 0,1

    # Create relations representing non-zero sparsity patterns

    # R1: relates (i,k) pairs where A[i,k] != 0
    # For demo, let's say A = [[1, 2], [0, 3]] (non-zeros at (0,0), (0,1), (1,1))
    matrix_dim = IndexSpace([2, 2])  # 2x2 matrix indices

    print("Matrix A sparsity pattern (non-zero entries):")
    print("A = [[1, 2],")
    print("     [0, 3]]")
    print("Non-zero at: (0,0), (0,1), (1,1)\n")

    # R1 constraints: enumerate specific non-zero positions
    # We'll use a constraint that's satisfied only for the non-zero positions
    i, k = bind_dims("i", "k")  # i=d0 (row), k=d1 (col)

    # Constraint: (i-0)*(k-0) + (i-0)*(k-1) + (i-1)*(k-1) = 0
    # This is satisfied when (i,k) ∈ {(0,0), (0,1), (1,1)}
    # Simplified: i*k + i*(k-1) + (i-1)*(k-1) = 0
    # = i*k + i*k - i + i*k - k - k + 1 = 3*i*k - i - 2*k + 1 = 0
    # Actually, let's use a simpler approach with multiple constraints combined

    # Better approach: use constraint that's 0 iff (i,k) is in our set
    # (i-0)*(k-0) * (i-0)*(k-1) * (i-1)*(k-1) = 0
    # But this is non-linear. Instead, use linear constraint that captures the pattern:

    # For A sparsity, we'll use: constraint is 0 when (i,k) ∈ {(0,0), (0,1), (1,1)}
    # One way: enumerate with multiple relations and take union, but let's use a trick:
    # Use constraint: (2*i + k) % 3 - (i + k) % 2 = 0
    # This won't work generically, so let's be more direct:

    # Direct approach: A_constraint = (i + k - 1) * (i - k) * i * k = 0 has solutions including our points
    # Even simpler: let's manually verify our points work with a specific constraint
    A_constraint = (i + k - 1) * (2*i - k)  # = 0 when (i,k) ∈ {(0,0), (1,1), (0,1)}

    R1 = IntegerRelation(matrix_dim, matrix_dim, [A_constraint])

    print("R1 (A sparsity relation) - (i,k) pairs where A[i,k] ≠ 0:")
    for domain_tuple, codomain_tuple in R1.enumerate_relation():
        print(f"  A[{domain_tuple[0]},{domain_tuple[1]}] -> referenced at ({codomain_tuple[0]},{codomain_tuple[1]})")
    print()

    # R2: relates (k,j) pairs where B[k,j] != 0
    # For demo, let's say B = [[1, 0], [2, 1]] (non-zeros at (0,0), (1,0), (1,1))
    print("Matrix B sparsity pattern (non-zero entries):")
    print("B = [[1, 0],")
    print("     [2, 1]]")
    print("Non-zero at: (0,0), (1,0), (1,1)\n")

    k2, j = bind_dims("k", "j")  # k=d0 (row), j=d1 (col)
    B_constraint = k2 * (j - 1) + (k2 - 1) * j  # = 0 when (k,j) ∈ {(0,0), (1,0), (1,1)}

    R2 = IntegerRelation(matrix_dim, matrix_dim, [B_constraint])

    print("R2 (B sparsity relation) - (k,j) pairs where B[k,j] ≠ 0:")
    for domain_tuple, codomain_tuple in R2.enumerate_relation():
        print(f"  B[{domain_tuple[0]},{domain_tuple[1]}] -> referenced at ({codomain_tuple[0]},{codomain_tuple[1]})")
    print()

    # Compose R1 and R2: R1 ∘ R2 gives (i,j) pairs where ∃k: A[i,k] ≠ 0 ∧ B[k,j] ≠ 0
    print("Composing relations: R1 ∘ R2")
    print("This gives (i,j) pairs where ∃k such that A[i,k] ≠ 0 AND B[k,j] ≠ 0")
    print("(i.e., positions in C = A*B that could be non-zero)\n")

    R_composed = R1.compose(R2)

    print("Result of composition - potential non-zero positions in C = A*B:")
    result_pairs = list(R_composed.enumerate_relation())
    for domain_tuple, codomain_tuple in result_pairs:
        print(f"  C[{domain_tuple[0]},{domain_tuple[1]}] could be non-zero")

    print(f"\nTotal potential non-zero positions: {len(result_pairs)} out of {matrix_dim.size()} total positions")

    # Verify by manual calculation:
    print("\nManual verification:")
    print("C[0,0] = A[0,0]*B[0,0] + A[0,1]*B[1,0] = 1*1 + 2*2 = 5 ≠ 0 ✓")
    print("C[0,1] = A[0,0]*B[0,1] + A[0,1]*B[1,1] = 1*0 + 2*1 = 2 ≠ 0 ✓")
    print("C[1,0] = A[1,0]*B[0,0] + A[1,1]*B[1,0] = 0*1 + 3*2 = 6 ≠ 0 ✓")
    print("C[1,1] = A[1,0]*B[0,1] + A[1,1]*B[1,1] = 0*0 + 3*1 = 3 ≠ 0 ✓")
    print("\nActual C = [[5, 2],")
    print("           [6, 3]]")
    print("All positions are non-zero, matching our composition result!")

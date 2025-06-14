from typing import Iterator, Optional
from dataclasses import dataclass
import itertools
from quasiaffine import (
    AffineExpr,
    Constant,
    Dim,
    bind_dims,
    bind_unique_dims,
)


@dataclass
class IndexSpace:
    dim_sizes: list[int]
    dims: list[Dim]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
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
        ranges = [f"{dim}{{0..{size-1}}}" for (dim, size) in zip(self.dims, self.dim_sizes)]
        return " × ".join(ranges)

    @property
    def num_dims(self) -> int:
        """Number of dimensions in this index space"""
        return len(self.dim_sizes)


class IntegerRelation:
    """
    Represents a relation between tuples of integers defined by quasiaffine constraints.

    A relation R ⊆ Domain × Codomain is defined by constraints of the form:
    f(d₀, d₁, ..., c₀, c₁, ..., l₀, l₁, ...) = 0

    where d_i are domain variables, c_i are codomain variables,
    l_i are local/existential variables
    """

    def __init__(
        self,
        domain: IndexSpace,
        codomain: IndexSpace,
        constraints: list[AffineExpr],
        locals: Optional[IndexSpace] = None,
    ):
        """
        Initialize an integer relation.

        Args:
            domain: The domain index space
            codomain: The codomain index space
            constraints: List of quasiaffine expressions interpreted as f(...) = 0
            num_locals: Number of local/existential variables
            local_bounds: Upper bounds for each local variable (0 to bound-1)
        """
        self.domain = domain
        self.codomain = codomain
        self.constraints = constraints
        self.locals = locals or IndexSpace([], [])

        for constraint in constraints:
            self._validate_constraint_dims(constraint)

    def _validate_constraint_dims(self, expr: AffineExpr):
        """Validate that expression doesn't use dimensions beyond max_dim"""
        all_dims = set(self.domain.dims + self.codomain.dims + self.locals.dims)
        if not expr.is_function_of_dims(all_dims):
            raise ValueError(
                f"Constraint {expr} uses dimensions outside the dims {all_dims}"
                " provided to the domain/codomain/locals."
            )

    def _evaluate_constraint(
        self,
        constraint: AffineExpr,
        domain_values: tuple[int, ...],
        codomain_values: tuple[int, ...],
        local_values: tuple[int, ...] = (),
    ) -> int:
        """
        Evaluate a constraint expression with given variable values.

        Convention:
        - d₀, d₁, ... d_{domain.num_dims-1} - domain dims
        - d_{domain.num_dims}, ..., d_{domain.num_dims + codomain.num_dims - 1} - codomain dims
        - d_{domain.num_dims + codomain.num_dims}, ... - local dims
        """
        substitutions = {}

        for dim, val in zip(self.domain.dims, domain_values):
            substitutions[dim] = Constant(val)

        for dim, val in zip(self.codomain.dims, codomain_values):
            substitutions[dim] = Constant(val)

        for dim, val in zip(self.locals.dims, local_values):
            substitutions[dim] = Constant(val)

        result = constraint.replace_dims(substitutions)
        if isinstance(result, Constant):
            return result.value
        else:
            raise ValueError(
                f"Constraint {constraint} did not evaluate to a constant with substitutions {substitutions}"
            )

    def _all_constraints_satisfied(
        self,
        domain_tuple: tuple[int, ...],
        codomain_tuple: tuple[int, ...],
        local_values: tuple[int, ...],
    ) -> bool:
        """Check if all constraints are satisfied for given variable assignments"""
        return all(
            self._evaluate_constraint(
                constraint,
                domain_tuple,
                codomain_tuple,
                local_values,
            )
            == 0
            for constraint in self.constraints
        )

    def is_in_relation(
        self,
        domain_tuple: tuple[int, ...],
        codomain_tuple: tuple[int, ...],
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

        return any(
            self._all_constraints_satisfied(domain_tuple, codomain_tuple, local_values)
            for local_values in self.locals
        )

    def enumerate_relation(self) -> Iterator[tuple[tuple[int, ...], tuple[int, ...]]]:
        """
        Generator that yields all (domain_tuple, codomain_tuple) pairs in the relation.
        """
        for domain_tuple in self.domain:
            for codomain_tuple in self.codomain:
                if self.is_in_relation(domain_tuple, codomain_tuple):
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
        new_locals = bind_unique_dims(self.domain.num_dims)
        new_locals_domain = IndexSpace(
            self.locals.dim_sizes + other.locals.dim_sizes + self.domain.dim_sizes,
            self.locals.dims + other.locals.dims + new_locals,
        )

        result_constraints = []

        # If self = R2 subset B x C, and other = R1 subset A x B, then a
        # constraint in other needs to have its codomain dims remapped to the
        # intermediate local dims.
        for constraint in other.constraints:
            remapped_constraint = constraint.replace_dims(
                dict(zip(other.codomain.dims, new_locals))
            )
            result_constraints.append(remapped_constraint)

        # If self = R2 subset B x C, and other = R1 subset A x B, then a
        # constraint in self needs to have its domain dims remapped to the
        # intermediate local dims.
        for constraint in self.constraints:
            remapped_constraint = constraint.replace_dims(
                dict(zip(self.domain.dims, new_locals))
            )
            result_constraints.append(remapped_constraint)

        return IntegerRelation(
            result_domain,
            result_codomain,
            result_constraints,
            new_locals_domain,
        )

    def to_set(self) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
        """Convert the relation to a list of all valid (domain, codomain) pairs."""
        return set(self.enumerate_relation())

    def size(self) -> int:
        """Count the number of tuples in the relation."""
        return sum(1 for _ in self.enumerate_relation())

    def __str__(self) -> str:
        """String representation of the relation."""
        constraints_str = " ∧ ".join(
            [f"({constraint}) = 0" for constraint in self.constraints]
        )
        if not constraints_str:
            constraints_str = "true"

        locals_str = f", (locals {self.locals})" if self.locals.num_dims else ""
        return f"{{(d, c) ∈ {self.domain} × {self.codomain}{locals_str} | {constraints_str}}}"

    def __repr__(self) -> str:
        return (
            f"IntegerRelation(domain={self.domain}, codomain={self.codomain}, "
            f"constraints={self.constraints}, locals={self.locals})"
        )


if __name__ == "__main__":
    # Domain and codomain are both 2D spaces: {0,1} × {0,1}
    d0, d1, d2, d3 = bind_dims("d0", "d1", "d2", "d3")
    domain = IndexSpace([2, 2], dims=[d0, d1])
    codomain = IndexSpace([2, 2], dims=[d2, d3])

    # Example constraint: d0 + d1 = d2 (sum of domain coords equals first codomain coord)
    # and d3 = 0 (second codomain coord is always 0)
    constraints = [d0 + d1 - d2, d3]  # d0 + d1 - d2 = 0  # d3 = 0
    relation = IntegerRelation(domain, codomain, constraints)

    print("Relation:", relation)
    print("\nValid pairs:")
    for domain_tuple, codomain_tuple in relation.enumerate_relation():
        print(f"  ({domain_tuple}, {codomain_tuple})")

    print(f"\nRelation size: {relation.size()}")

    # 1D Composition example
    a, b = bind_dims("a", "b")
    b1, c = bind_dims("b1", "c")
    A = IndexSpace([4], [a])
    B = IndexSpace([4], [b])
    B1 = IndexSpace([4], [b1])
    C = IndexSpace([4], [c])

    r1 = IntegerRelation(A, B, [2 * a + b - 4])
    r2 = IntegerRelation(B1, C, [b1 + c - 3])

    print("\n\n")
    print("R1:")
    print(r1)
    for domain_tuple, codomain_tuple in r1.enumerate_relation():
        print(f"  ({domain_tuple}, {codomain_tuple})")
    print("size: " + str(r1.size()))
    print("R2:")
    print(r2)
    for domain_tuple, codomain_tuple in r2.enumerate_relation():
        print(f"  ({domain_tuple}, {codomain_tuple})")
    print("size: " + str(r2.size()))

    composed = r2.compose(r1)
    print("\nComposed relation:")
    print(composed)
    print("size: " + str(composed.size()))

    for domain_tuple, codomain_tuple in composed.enumerate_relation():
        print(f"  ({domain_tuple}, {codomain_tuple})")

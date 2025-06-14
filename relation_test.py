from quasiaffine import bind_dims
from relation import IndexSpace, IntegerRelation


def test_relation():
    a, b = bind_dims("a", "b")
    A = IndexSpace([4], [a])
    B = IndexSpace([4], [b])
    r1 = IntegerRelation(A, B, [2 * a + b - 4])
    actual = r1.to_set()
    expected = {
        ((1,), (2,)),
        ((2,), (0,)),
    }
    assert actual == expected


def test_compose():
    a, b = bind_dims("a", "b")
    b1, c = bind_dims("b1", "c")
    A = IndexSpace([4], [a])
    B = IndexSpace([4], [b])
    B1 = IndexSpace([4], [b1])
    C = IndexSpace([4], [c])

    # 2a + b = 4
    r1 = IntegerRelation(A, B, [2 * a + b - 4])
    # b1 + c = 3
    r2 = IntegerRelation(B1, C, [b1 + c - 3])

    # 2a + b = 4
    # b1 + c = 3
    # b = b1
    #
    # simplifies to
    #
    # 2a - c = 1
    composed = r2.compose(r1)
    expected = {
        ((1,), (1,)),
        ((2,), (3,)),
    }
    actual = composed.to_set()
    assert actual == expected

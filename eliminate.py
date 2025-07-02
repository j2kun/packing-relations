import numpy
import numpy as np
import numpy.linalg


def rowSwap(A, i, j):
    temp = numpy.copy(A[i, :])
    A[i, :] = A[j, :]
    A[j, :] = temp


def colSwap(A, i, j):
    temp = numpy.copy(A[:, i])
    A[:, i] = A[:, j]
    A[:, j] = temp


def scaleCol(A, i, c):
    A[:, i] *= int(c) * numpy.ones(A.shape[0], dtype=np.int64)


def scaleRow(A, i, c):
    A[i, :] = (
        np.array(A[i, :], dtype=numpy.float64)
        * c
        * numpy.ones(A.shape[1], dtype=numpy.float64)
    )


def colCombine(A, addTo, scaleCol, scaleAmt):
    A[:, addTo] += scaleAmt * A[:, scaleCol]


def rowCombine(A, addTo, scaleRow, scaleAmt):
    A[addTo, :] += scaleAmt * A[scaleRow, :]


if __name__ == "__main__":
    # row, col, ct, slot, d0, d1, c1, c2, c3, c4

    c = numpy.array([1] * 10)
    A = numpy.array(
        [
            # slot % 4 - row = 0
            # i.e.
            # slot + 4 c1 - 4 nc1 - row = 0
            [-1, 0, 0, 1, 0, 0, 4, 0, 0, 0],
            # (ct + slot) % 4 - d1 = 0
            # i.e.
            # ct + slot + 4 c2 - 4nc2 - d1 = 0
            [0, 0, 1, 1, 0, -1, 0, 4, 0, 0],
            # d0 % 4 - d1 = 0
            # i.e.
            # d0 + 4 c3 - 4 nc3 - d1 = 0
            [0, 0, 0, 0, 1, -1, 0, 0, 4, 0],
            # (d0 + 1) % 4 - col = 0
            # i.e.
            # d0 + 4 c4 - 4 nc4 - col = -1
            [0, -1, 0, 0, 1, 0, 0, 0, 0, 4],
        ]
    )
    b = numpy.array([[0], [0], [0], [-1]])

    tableau = numpy.column_stack([A, b])
    print(f"Input tableau:\n{tableau}")

    rowSwap(tableau, 1, 3)
    rowSwap(tableau, 2, 3)

    # columns vars are now
    # row, col, ct, d0, slot, d1, c1, c2, c3, c4
    colSwap(tableau, 3, 4)

    scaleRow(tableau, 0, -1)
    scaleRow(tableau, 1, -1)

    rowCombine(tableau, 1, 3, 1)

    print(f"\nReduced tableau:\n{tableau}")

    # reading off the tableau (manually) we get
    """
    row, col, ct, d0, slot, d1, c1, c2, c3, c4

      row - slot - 4c1 = 0
    [[ 1  0  0  0 -1  0 -4  0  0  0  0]

      col - d1 + 4c3 - 4c4 = 1
     [ 0  1  0  0  0 -1  0  0  4 -4  1]

      ct + slot - d1 + 4c2 = 0
     [ 0  0  1  0  1 -1  0  4  0  0  0]

      d0 - d1 + 4c3 = 0
     [ 0  0  0  1  0 -1  0  0  4  0  0]]
    """

    # The constraints are now:
    #
    # row = slot % 4
    # col = (d1 + 1) % 4
    # ct = (-slot + d1) % 4
    # d0 = d1 % 4

    # Now iterating over slot and d1, we can recover all the remaining
    # variables

    entries = []
    for d1 in range(4):
        for slot in range(8):
            row = slot % 4
            col = (d1 + 1) % 4
            ct = (-slot + d1) % 4
            d0 = d1 % 4

            entries.append(((ct, slot), (row, col)))

    entries.sort()
    for entry in entries:
        print(entry)

    mat_rows = 4
    mat_cols = 4
    num_cts = 4
    num_slots = 8
    data = np.arange(mat_rows * mat_cols, dtype=np.int32).reshape((mat_rows, mat_cols))

    result = np.zeros((num_cts, num_slots), dtype=np.int32)
    for (ct, slot), (row, col) in entries:
        result[ct, slot] = data[row, col]

    print(result)

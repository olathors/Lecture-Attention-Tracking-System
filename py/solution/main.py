import numpy as np


def lab00():
    # 1. Test the Getting Started code here:
    # --------------------------------------
    # https://numpy.org/doc/stable/user/quickstart.html

    # 2. Solve the tutorial problems here:
    # a) Create a few vectors and matrices.
    # --------------------------------------
    # Todo: Create vector t.
    # https://numpy.org/doc/stable/user/quickstart.html#array-creation
    t = np.array([1, 0, 3])  # .reshape(3,1)

    # Todo: Create matrix A.
    A = np.array([[1, 0, 3],
                  [4, 5, 6],
                  [7, 8, 9]])

    # Todo: Create identity matrix I.
    # https://numpy.org/doc/stable/user/quickstart.html#functions-and-methods-overview
    I = np.eye(3)

    # Todo: Create matrix T.
    # https://numpy.org/doc/stable/user/quickstart.html#stacking-together-different-arrays
    T = np.row_stack((np.c_[A, t], [0, 0, 0, 1]))

    # Todo: Create matrix B.
    # https://numpy.org/doc/stable/user/quickstart.html#changing-the-shape-of-an-array
    B = A.T

    # Print the results (uncomment code below).
    print("a) Create a few vectors and matrices:")
    print("-------------------------------------")
    print(f"t = \n{t}")
    print(f"A = \n{A}")
    print(f"I = \n{I}")
    print(f"T = \n{T}")
    print(f"B = \n{B}")

    # b) Coefficients.
    # -----------------
    # Todo: Follow lab notes.
    t[1] = 2
    A[0, 1] = 2

    # https://numpy.org/doc/stable/user/quickstart.html#indexing-slicing-and-iterating
    # T[0, 1] = 2
    # T[1, 3] = 2

    # https://numpy.org/doc/stable/user/quickstart.html#indexing-with-arrays-of-indices
    # T[(0, 1), (1, 3)] = 2
    T[[0, 1], [1, 3]] = 2
    print("b) Coefficients:")
    print("-------------------------------------")
    print(f"t = \n{t}")
    print(f"A = \n{A}")
    print(f"T = \n{T}")

    # c) Block operations.
    # ---------------------
    # Todo: Follow lab notes.

    print("c) Block operations:")
    print("-------------------------------------")
    r_2 = A[1, :]
    c_2 = A[:, 1]
    T_3x4 = T[0:3]

    print(f"r_2 = \n{r_2}")
    print(f"c_2 = \n{c_2}")
    print(f"T_3x4 = \n{T_3x4}")

    # https://numpy.org/doc/stable/user/quickstart.html#copies-and-views
    r_2[:] = 0
    c_2[:] = 0
    T_3x4[:] = 0
    print(f"A = \n{A}")
    print(f"T = \n{T}")

    # d) Matrix and vector arithmetic.
    # ---------------------------------
    # Todo: Follow lab notes.
    # https://numpy.org/doc/stable/user/quickstart.html#basic-operations

    v1 = np.array([1, 2, 3])
    v2 = np.array([3, 2, 1])
    print("d) Matrix and vector arithmetic:")
    print("--------------------------------")
    print(f"v1: {v1}\nv2: {v2}")
    print(f"v1 + v2 = {v1 + v2}")
    print(f"A + I = \n{A + I}")
    print(f"(A+I) * T_3x4 =\n{(A+I) @ T_3x4}")
    print(f"v1.transpose() * v2 = {v1.transpose() * v2}")
    print(f"v1.transpose() @ v2 = {v1.transpose() @ v2}")
    print(f"v1.dot(v2) = {v1.dot(v2)}")
    print(f"Element-wise B * I =\n{B * I}")

    # e) Reductions.
    # ---------------
    # Todo: Follow lab notes.
    A = np.arange(10, 19).reshape(3, 3)
    A[2, 2] = 5
    print("e) Reductions:")
    print("--------------------------------")
    print(f"A:\n{A}")

    # Take the sum of all elements in a matrix
    # https://numpy.org/doc/stable/user/quickstart.html#basic-operations
    print(f"sum of A: {A.sum()}")

    # Compute the minimum value in a matrix
    # Also, find its position in the matrix.
    print(f"minimum of A: {A.min()} at index {A.argmin()}, or {np.unravel_index(A.argmin(), A.shape)}")

    # Create a vector that is the maximum of each column in a matrix.
    print(f"maximum of each column in A: {A.max(0)}")

    # Find the L2-norm of a vector.
    v = np.array([1, 1])
    print(f"L2 norm of {v} is {np.linalg.norm(v)}")

    # Find the number of elements in a vector that is greater than a given value.
    # https://numpy.org/doc/stable/user/quickstart.html#changing-the-shape-of-an-array
    v = np.array([0, 10, 2, 13, 4, 15, 6, 17, 8, 19])
    print(f"v: {v}, greater than 10: {(v > 10).sum()}")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    lab00()

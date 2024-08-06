import numpy as np


M = [[4, 6, 0],
     [6, 2, 9],
     [0, 9, 0]]


def gaussian_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    augmented_matrix = np.hstack([A, b.reshape(-1, 1)])

    for i in range(n):
        max_row = i + np.argmax(np.abs(augmented_matrix[i:, i]))
        augmented_matrix[[i, max_row]] = augmented_matrix[[max_row, i]]

        pivot = augmented_matrix[i, i]
        augmented_matrix[i] = augmented_matrix[i] / pivot

        for k in range(i + 1, n):
            coefficient = augmented_matrix[k, i]
            augmented_matrix[k] -= coefficient * augmented_matrix[i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = augmented_matrix[i, -1] - np.sum(augmented_matrix[i, i + 1:n] * x[i + 1:n])

    return x


def polynomial_value(coefficients, x):
    result = 0
    for coefficient in coefficients:
        result = result * x + coefficient
    return result


def secant_method(coefficients, x0, x1, epsilon):
    x2 = 0
    while True:
        f_x0 = polynomial_value(coefficients, x0)
        f_x1 = polynomial_value(coefficients, x1)
        if abs(x1 - x0) < epsilon:
            break
        x2 = (f_x1 * x0 - f_x0 * x1)/(f_x1 - f_x0)
        x0, x1 = x1, x2
    return x2


def horners_scheme(coefficients, root):
    n = len(coefficients)
    quotient = [0] * (n - 1)
    remainder = coefficients[0]

    for i in range(1, n):
        quotient[i - 1] =\
            remainder
        remainder = coefficients[i] + remainder * root

    return quotient


def krylov_method(M, b):
    n = len(M)
    krylov_vectors = []
    for i in range(n):
        if i != 0:
            B = np.linalg.matrix_power(M, i)
            krylov_vectors.append(np.dot(B, b))
        else:
            krylov_vectors.append(b)
    for i in range(len(krylov_vectors)):
        krylov_vectors[i] = np.ravel(krylov_vectors[i])
    krylov_vectors = krylov_vectors[::-1]
    A = np.column_stack(krylov_vectors)
    coefficients = gaussian_elimination(A, -np.dot(np.linalg.matrix_power(M, n), b))
    coefficients = np.insert(coefficients, 0, 1)
    eigenvalues = []
    while len(coefficients) >= 2:
        eigenvalues.append(secant_method(coefficients, 1, 10, 0.0001))
        coefficients = horners_scheme(coefficients, eigenvalues[len(eigenvalues) - 1])
    return eigenvalues


def Sznejder_Kacper_(A):
    b = [1] + [0] * (len(A) - 1)
    return krylov_method(A, b)


print("Wartości własne: " + str(Sznejder_Kacper_(M)))

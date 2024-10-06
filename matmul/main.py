import csv
import timeit
from numba import cuda, jit
import numpy as np


@cuda.jit
def cuda_matmul(matrix_a, matrix_b, matrix_c) -> None:
    """
    Функция ядра, выполняющая произведение матриц.

    Args:
        matrix_a: Первая матрица.
        matrix_b: Вторая матрица.
        matrix_c: Матрица-произведение.
    """
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    while index < matrix_a.size:
        i, j = index // matrix_a.shape[0], index % matrix_a.shape[0]
        matrix_c[i, j] = 0
        for k in range(matrix_a.shape[1]):
            matrix_c[i, j] += matrix_a[i, k] * matrix_b[k, j]
        index += cuda.blockDim.x * cuda.gridDim.x


@jit(nopython=True, cache=True)
def cpu_jit_matmul(matrix_a: np.ndarray, matrix_b: np.ndarray, matrix_c: np.ndarray) -> np.ndarray:
    """
    Произведение матриц с использованием JIT.

    Args:
        matrix_a: Первая матрица.
        matrix_b: Вторая матрица.
        matrix_c: Матрица-произведение.

    Returns:
        Произведение матриц A, B.
    """
    for i in range(matrix_a.shape[1]):
        for j in range(matrix_b.shape[0]):
            matrix_c[i, j] = 0
            for k in range(matrix_a.shape[1]):
                matrix_c[i, j] += matrix_a[i, k] * matrix_b[k, j]
    return matrix_c


def cpu_matmul(matrix_a: np.ndarray, matrix_b: np.ndarray, matrix_c: np.ndarray) -> np.ndarray:
    """
    Произведение матриц с использованием JIT.

    Args:
        matrix_a: Первая матрица.
        matrix_b: Вторая матрица.
        matrix_c: Матрица-произведение.

    Returns:
        Произведение матриц A, B.
    """
    for i in range(matrix_a.shape[1]):
        for j in range(matrix_b.shape[0]):
            matrix_c[i, j] = 0
            for k in range(matrix_a.shape[1]):
                matrix_c[i, j] += matrix_a[i, k] * matrix_b[k, j]
    return matrix_c


def run_cuda_matmul(matrix_a: np.ndarray, matrix_b: np.ndarray) -> np.ndarray:
    """
    Произведение матриц с использованием CUDA.

    Args:
        matrix_a: Первая матрица.
        matrix_b: Вторая матрица.
    """
    # Перенос данных в глобальную память GPU
    d_A = cuda.to_device(matrix_a)
    d_B = cuda.to_device(matrix_b)
    d_C = cuda.device_array_like(matrix_a)
    # blocksize или количество потоков на блок, стандартное значение = 32
    tpb = device.WARP_SIZE
    # блоков на грид
    bpg = 512
    # вызов ядра
    cuda_matmul[bpg, tpb](d_A, d_B, d_C)
    return d_C.copy_to_host()


n = [100, 250, 500, 750, 1000]
number_of_tests = 12
device = cuda.get_current_device()
results = np.zeros((len(n), 8))
results[:, 0] = n
for i, matr_size in enumerate(n):
    A = np.random.randint(1, 10, (matr_size, matr_size))
    B = np.random.randint(1, 10, (matr_size, matr_size))
    C = np.random.randint(1, 10, (matr_size, matr_size))
    results[i, 1] = timeit.timeit(lambda: cpu_matmul(A, B, C), number=1) / 1
    results[i, 2] = timeit.timeit(lambda: cpu_jit_matmul(A, B, C), number=number_of_tests) / number_of_tests
    results[i, 3] = timeit.timeit(lambda: run_cuda_matmul(A, B),
                                  number=number_of_tests) / number_of_tests
    results[i, 4] = timeit.timeit(lambda: np.dot(A, B), number=number_of_tests) / number_of_tests
    print(f"Cpu matmul: {results[i, 1]}")
    print(f"Cpu jit matmul: {results[i, 2]}")
    print(f"Cuda matmul: {results[i, 3]}")
    print(f"Numpy dot: {results[i, 4]}")

results[:, 5] = results[:, 1] / results[:, 2]
results[:, 6] = results[:, 1] / results[:, 3]
results[:, 7] = results[:, 1] / results[:, 4]

with open('results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["Matrix size", "CPU time (s)", "CPU JIT time(s)", "CUDA time(s)", "Numpy dot time(s)",
                         "CPU JIT speed up", "CUDA speed up", "Numpy dot speed up"])
    spamwriter.writerows(results)

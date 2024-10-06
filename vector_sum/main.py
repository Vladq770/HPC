import csv
import timeit
from numba import cuda, jit
import numpy as np


@cuda.jit
def cuda_vector_sum(vector, vec_sum) -> None:
    """
    Функция ядра, выполняющая сложение элементов вектора.

    Args:
        vector: Bектор, представляет собой одномерный np.ndarray.
        vec_sum: Сумма элементов вектора.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    if i < vector.size:
        cuda.atomic.add(vec_sum, 0, vector[i])


@jit(nopython=True, cache=True)
def cpu_jit_vector_sum(vector: np.ndarray) -> float:
    """
    Сложение элементов вектора с использованием JIT.

    Args:
        vector: Bектор, представляет собой одномерный np.ndarray.
    
    Returns:
        Сумма элементов вектора.
    """
    res = 0
    for element in vector:
        res += element
    return res


def cpu_vector_sum(vector: np.ndarray) -> float:
    """
    Сложение элементов вектора.

    Args:
        vector: Bектор, представляет собой одномерный np.ndarray.
    
    Returns:
        Сумма элементов вектора.
    """
    res = 0
    for element in vector:
        res += element
    return res


def run_cuda_vector_sum(vector: np.ndarray, vec_sum: np.ndarray) -> np.ndarray:
    """
    Сложение элементов вектора с использованием CUDA.

    Args:
        vector: Bектор, представляет собой одномерный np.ndarray.
        vec_sum: Сумма элементов вектора.
    """
    # Перенос данных в глобальную память GPU
    d_vector = cuda.to_device(vector)
    d_vec_sum = cuda.device_array_like(vec_sum)
    # blocksize или количество потоков на блок, стандартное значение = 32
    tpb = device.WARP_SIZE
    # блоков на грид
    bpg = int(np.ceil((n) / tpb))
    # вызов ядра
    cuda_vector_sum[bpg, tpb](d_vector, d_vec_sum)
    return d_vec_sum.copy_to_host()


n = 10 ** 2
number_of_tests = 12
vector = np.random.rand(n)
vec_sum = np.random.rand(1)
device = cuda.get_current_device()
k = 8
results = np.zeros((k, 8))
results[:, 0] = [10 ** (4 + i) for i in range(k)]
for i in range(k):
    vector = np.random.rand(int(results[i, 0]))
    results[i, 1] = timeit.timeit(lambda: cpu_vector_sum(vector), number=number_of_tests) / number_of_tests
    results[i, 2] = timeit.timeit(lambda: cpu_jit_vector_sum(vector), number=number_of_tests) / number_of_tests
    results[i, 3] = timeit.timeit(lambda: run_cuda_vector_sum(vector, vec_sum), number=number_of_tests) / number_of_tests
    results[i, 4] = timeit.timeit(lambda: vector.sum(), number=number_of_tests) / number_of_tests
    print(f"Cpu vector sum: {results[i, 1]}")
    print(f"Cpu jit vector sum: {results[i, 2]}")
    print(f"Cuda vector sum: {results[i, 3]}")
    print(f"Numpy sum: {results[i, 4]}")

results[:, 5] = results[:, 1] / results[:, 2]
results[:, 6] = results[:, 1] / results[:, 3]
results[:, 7] = results[:, 1] / results[:, 4]

with open('results.csv', 'w', newline='') as csvfile:
    spamwriter = csv.writer(csvfile)
    spamwriter.writerow(["Vector size", "CPU time (s)", "CPU JIT time(s)", "CUDA time(s)", "Numpy sum time(s)",
                         "CPU JIT speed up", "CUDA speed up", "Numpy sum speed up"])
    spamwriter.writerows(results)

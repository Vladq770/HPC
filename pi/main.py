import timeit
from numba import cuda
import numpy as np


@cuda.jit
def cuda_points_count(points, count) -> None:
    """
    Функция ядра, выполняющая вычисление количества точек в окружности.

    Args:
        points: Массив точек.
        count: Переменная, для хранения количества точек в окружности.
    """
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    while index < points.shape[0]:
        if points[index][0] ** 2 + points[index][1] ** 2 <= 1:
            cuda.atomic.add(count, 0, 1)
        index += cuda.blockDim.x * cuda.gridDim.x


def cpu_points_count(points: np.ndarray) -> int:
    """
    Количество точек в окружности.

    Args:
        points: Массив точек.
    
    Returns:
        Количество точек внутри окружности.
    """
    res = 0
    for point in points:
        if point[0] ** 2 + point[1] ** 2 <= 1:
            res += 1
    return res


def run_cuda_points_count(points: np.ndarray, count: np.ndarray) -> np.ndarray:
    """
    Количество точек в окружности с использованием CUDA.

    Args:
        points: Массив точек.
        count: Переменная, для хранения количества точек в окружности.

    Returns:
        Количество точек внутри окружности.
    """
    # Перенос данных в глобальную память GPU
    d_points = cuda.to_device(points)
    d_count = cuda.device_array_like(count)
    # blocksize или количество потоков на блок, стандартное значение = 32
    tpb = device.WARP_SIZE
    # блоков на грид
    bpg = 512
    # вызов ядра
    cuda_points_count[bpg, tpb](d_points, d_count)
    return d_count.copy_to_host()


n = 10 ** 6
number_of_tests = 12
points = (np.random.rand(n, 2) - 0.5) * 2
count = np.zeros(1)
device = cuda.get_current_device()
cuda_res = 4 * run_cuda_points_count(points, count)[0] / n
cpu_res = 4 * cpu_points_count(points) / n
time_cuda = timeit.timeit(lambda: run_cuda_points_count(points, count), number=number_of_tests) / number_of_tests
time_cpu = timeit.timeit(lambda: cpu_points_count(points), number=number_of_tests) / number_of_tests
print(f"TIME CUDA = {time_cuda}")
print(f"TIME CPU = {time_cpu}")
print(f"SPEED UP = {time_cpu / time_cuda}")
print(f"GPU: pi = {cuda_res}")
print(f"CPU: pi = {cpu_res}")

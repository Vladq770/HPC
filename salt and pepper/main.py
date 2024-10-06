import timeit
import cv2
import numba
from numba import cuda
import numpy as np


@cuda.jit
def cuda_salt_and_pepper(image, res_image) -> None:
    """
    Функция ядра, выполняющая обработку изображения медианным фильтром (3x3).

    Args:
        image: Входное изображение.
        res_image: Обработанное изображение.
    """
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    rows, cols = image.shape[0] - 2, image.shape[1] - 2
    while index < rows * cols:
        window = cuda.local.array(shape=9, dtype=numba.float64)
        i, j = index // cols, index % cols
        for k in range(9):
            window[k] = image[i + k // 3, j + k % 3]
        for k in range(8):
            for l in range(8-k):
                if window[l] > window[l+1]:
                    window[l], window[l + 1] = window[l + 1], window[l]
        res_image[i, j] = window[4]
        index += cuda.blockDim.x * cuda.gridDim.x


def cpu_salt_and_pepper(image: np.ndarray) -> np.ndarray:
    """
    Обработка изображения медианным фильтром (3x3).

    Args:
        image: Входное изображение.
    
    Returns:
        Обработанное изображение.
    """
    rows, cols = image.shape[0] - 2, image.shape[1] - 2
    res_image = np.zeros((rows, cols))
    for i in range(rows):
        for j in range(cols):
            window = image[i:i+3, j:j+3].flatten()
            window.sort()
            res_image[i, j] = window[4]
    return res_image


def run_cuda_salt_and_pepper(image: np.ndarray, res_image: np.ndarray) -> np.ndarray:
    """
    Обработка изображения медианным фильтром (3x3) с использованием CUDA.

    Args:
        image: Входное изображение.
        res_image: Обработанное изображение.

    Returns:
        Обработанное изображение.
    """
    # Перенос данных в глобальную память GPU
    d_image = cuda.to_device(image)
    d_res_image = cuda.device_array_like(res_image)
    # blocksize или количество потоков на блок, стандартное значение = 32
    tpb = device.WARP_SIZE
    # блоков на грид
    bpg = 1024
    # вызов ядра
    cuda_salt_and_pepper[bpg, tpb](d_image, d_res_image)
    return d_res_image.copy_to_host()


probability = 0.3
img = cv2.imread('original.jpg', cv2.IMREAD_GRAYSCALE)
salt_pepper_img = np.where(np.random.rand(*img.shape) < probability, np.random.randint(0, 255, img.shape), img)
cv2.imwrite('salt_pepper.jpg', salt_pepper_img)
number_of_tests = 12
image = np.pad(salt_pepper_img, 1, mode="symmetric")
res_image = np.zeros(salt_pepper_img.shape)
device = cuda.get_current_device()
res_image_gpu = run_cuda_salt_and_pepper(image, res_image)
res_image_cpu = cpu_salt_and_pepper(image)
print(np.allclose(res_image_cpu, res_image_gpu))
cv2.imwrite('res_image_gpu.jpg', res_image_gpu)
cv2.imwrite('res_image_cpu.jpg', res_image_cpu)
time_cuda = timeit.timeit(lambda: run_cuda_salt_and_pepper(image, res_image), number=number_of_tests) / number_of_tests
time_cpu = timeit.timeit(lambda: cpu_salt_and_pepper(image), number=number_of_tests) / number_of_tests
print(f"TIME CUDA = {time_cuda}")
print(f"TIME CPU = {time_cpu}")
print(f"SPEED UP = {time_cpu / time_cuda}")

import time
from numba import cuda
from numba.cuda.random import xoroshiro128p_uniform_float32, create_xoroshiro128p_states, xoroshiro128p_normal_float64
import numpy as np
import matplotlib.pyplot as plt


@cuda.jit
def cuda_get_fitnesses(population, points, fitnesses) -> None:
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    while index < population.shape[0]:
        fitness = 0
        for point in points:
            value = 0
            for j in range(population.shape[1] - 1):
                value += population[index, j] * point[0] ** j
            fitness += (point[1] - value) ** 2
        fitnesses[index] = fitness
        index += cuda.blockDim.x * cuda.gridDim.x


@cuda.jit
def cuda_get_next_population(rng_states, population, new_population) -> None:
    index = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    current_size = population.shape[0]
    while index < current_size:
        i = int(xoroshiro128p_uniform_float32(rng_states, index) * current_size)
        j = int(xoroshiro128p_uniform_float32(rng_states, index) * current_size)
        while i == j:
            j = int(xoroshiro128p_uniform_float32(rng_states, index) * current_size)
        shift = int(xoroshiro128p_uniform_float32(rng_states, index) * gens_count)
        if not shift:
            shift = 1
        for k in range(shift):
            new_population[index, k] = population[i, k]
            new_population[index + current_size, k] = population[j, k]
        for k in range(gens_count - shift):
            new_population[index, shift + k] = population[j, shift + k]
            new_population[index + current_size, shift + k] = population[i, shift + k]
        for k in range(gens_count):
            if xoroshiro128p_uniform_float32(rng_states, index) < 0.02:
                new_population[index, k] = E + D * xoroshiro128p_normal_float64(rng_states, index)
            if xoroshiro128p_uniform_float32(rng_states, index) < 0.02:
                new_population[index + current_size, k] = E + D * xoroshiro128p_normal_float64(rng_states, index)

        index += cuda.blockDim.x * cuda.gridDim.x


def get_points(coefficients, points_count) -> np.ndarray:
    points = np.zeros((points_count, 2))
    for point in points:
        point[0] = -5 + np.random.rand() * 10
        point[1] = get_y(coefficients, point[0]) + np.random.rand()
    return points


def get_y(coefficients, x):
    y = 0
    for i, coef in enumerate(coefficients):
        y += coef * x ** i
    return y


def run_cuda_get_fitnesses(population: np.ndarray, points: np.ndarray) -> np.ndarray:
    d_points = cuda.to_device(points)
    d_population = cuda.to_device(population)
    d_fitnesses = cuda.device_array_like(population[:, 0])
    # blocksize или количество потоков на блок, стандартное значение = 32
    tpb = device.WARP_SIZE
    # блоков на грид
    bpg = 1024
    # вызов ядра
    cuda_get_fitnesses[bpg, tpb](d_population, d_points, d_fitnesses)
    return d_fitnesses.copy_to_host()


def run_cuda_get_next_population(population: np.ndarray) -> np.ndarray:
    d_population = cuda.to_device(population)
    d_new_population = cuda.device_array((population_size, gens_count + 1))
    # blocksize или количество потоков на блок, стандартное значение = 32
    tpb = device.WARP_SIZE
    # блоков на грид
    bpg = 1024
    # вызов ядра
    cuda_get_next_population[bpg, tpb](rng_states, d_population, d_new_population)
    return d_new_population.copy_to_host()


points_count = 10 ** 3
iter_count = 10 ** 3
gens_count = 6
E = 0
D = 1
error = 0.1
population_size = 10 ** 3
coefficients = np.random.rand(gens_count)
coefficients[-1] += 1
points = get_points(coefficients, points_count)
threads_per_block = 32
blocks = 1024
rng_states = create_xoroshiro128p_states(threads_per_block * blocks, seed=1)
population = np.random.rand(population_size, gens_count + 1)
number_of_tests = 12
device = cuda.get_current_device()
current_iter = 0
best_fitness = np.inf
best_solution = None
start_time = time.time()
while current_iter < iter_count:
    fitnesses = run_cuda_get_fitnesses(population, points)
    population[:, -1] = fitnesses
    population = population[np.argsort(population[:, -1])]
    if population[0, -1] < best_fitness:
        best_fitness = population[0, -1]
        best_solution = population[0,:-1]
        if best_fitness < error:
            break
    population = run_cuda_get_next_population(population[:population_size // 2])
    current_iter += 1
print(f"Time: {time.time() - start_time}")
n = 100
y_pred = np.zeros((n, 1))
x_lin = np.linspace(points[:, 0].min(), points[:, 0].max(), n)
for i, x in enumerate(x_lin):
    y_pred[i] = get_y(best_solution, x)

print(f"Коэффициенты: {coefficients}")
print(f"Найденные коэффициенты: {best_solution}")
print(f"Количество итераций: {current_iter}")
plt.scatter(points[:, 0], points[:, 1])
plt.plot(x_lin, y_pred, color="red")
plt.savefig("res.png")
plt.show()



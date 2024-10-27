#include <iostream>
#include <locale>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <fstream> 


void multiplyMatricesCPU(const int* A, const int* B, int* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] += A[i * n + k] * B[k * n + j];
            }
        }
    }
}

void generateRandomMatrix(int* matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = rand() % 10;
    }
}


int main() {
    setlocale(LC_ALL, "Russian");

    const int num_tests = 5; 
    int n_values[num_tests] = { 100, 250, 500, 750, 1000 };

    srand(static_cast<unsigned>(time(0)));

    for (int i = 0; i < num_tests; i++) {
        int n = n_values[i];
        int* A = new int[n * n];
        int* B = new int[n * n];
        int* C_CPU_Seq = new int[n * n];
        int* C_GPU = new int[n * n];

        generateRandomMatrix(A, n);
        generateRandomMatrix(B, n);

 
        auto start_cpu_seq = std::chrono::high_resolution_clock::now();
        multiplyMatricesCPU(A, B, C_CPU_Seq, n);
        auto end_cpu_seq = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration_cpu_seq = end_cpu_seq - start_cpu_seq;
        std::cout << n << ": " << duration_cpu_seq.count() << '\n';

        delete[] A;
        delete[] B;
        delete[] C_CPU_Seq;
        delete[] C_GPU;
    }
    return 0;
}
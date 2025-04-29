#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>
#include <iostream>
#include <vector>

using namespace sycl;
using data_type = double;

void print_matrix(int rows, int cols, const data_type* mat, int ld) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j)
            std::printf("%6.1f ", mat[i * ld + j]);
        std::printf("\n");
    }
}

int main() {
    const int m = 2, n = 2, k = 2;
    const int lda = 2, ldb = 2, ldc = 2;

    std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};  // 2x2
    std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};  // 2x2
    std::vector<data_type> C(m * n, 0.0);

    const data_type alpha = 1.0;
    const data_type beta  = 0.0;

    std::cout << "A\n";
    print_matrix(m, k, A.data(), lda);
    std::cout << "=====\nB\n";
    print_matrix(k, n, B.data(), ldb);
    std::cout << "=====\n";

    // SYCL queue and context
    queue q{gpu_selector_v};

    // USM device allocations (like cudaMalloc)
    data_type* d_A = malloc_device<data_type>(A.size(), q);
    data_type* d_B = malloc_device<data_type>(B.size(), q);
    data_type* d_C = malloc_device<data_type>(C.size(), q);

    // Copy to device
    q.memcpy(d_A, A.data(), sizeof(data_type) * A.size()).wait();
    q.memcpy(d_B, B.data(), sizeof(data_type) * B.size()).wait();

    // Call DPC++ oneMKL GEMM (double precision GEMM)
    oneapi::mkl::blas::column_major::gemm(
        q, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        m, n, k,
        alpha,
        d_A, lda,
        d_B, ldb,
        beta,
        d_C, ldc
    ).wait();

    // Copy back result
    q.memcpy(C.data(), d_C, sizeof(data_type) * C.size()).wait();

    // Display result
    std::cout << "C = A * B\n";
    print_matrix(m, n, C.data(), ldc);
    std::cout << "=====\n";

    // Cleanup
    free(d_A, q);
    free(d_B, q);
    free(d_C, q);

    return 0;
}

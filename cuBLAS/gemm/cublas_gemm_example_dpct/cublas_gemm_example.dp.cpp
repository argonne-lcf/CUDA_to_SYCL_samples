/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <dpct/blas_utils.hpp>

#include "cublas_utils.h"

using data_type = double;

int main(int argc, char *argv[]) try {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    dpct::blas::descriptor_ptr cublasH = NULL;
    dpct::queue_ptr stream = &q_ct1;

    const int m = 2;
    const int n = 2;
    const int k = 2;
    const int lda = 2;
    const int ldb = 2;
    const int ldc = 2;
    /*
     *   A = | 1.0 | 2.0 |
     *       | 3.0 | 4.0 |
     *
     *   B = | 5.0 | 6.0 |
     *       | 7.0 | 8.0 |
     */

    const std::vector<data_type> A = {1.0, 2.0, 3.0, 4.0};
    const std::vector<data_type> B = {5.0, 6.0, 7.0, 8.0};
    std::vector<data_type> C(m * n);
    const data_type alpha = 1.0;
    const data_type beta = 0.0;

    data_type *d_A = nullptr;
    data_type *d_B = nullptr;
    data_type *d_C = nullptr;

    oneapi::mkl::transpose transa = oneapi::mkl::transpose::nontrans;
    oneapi::mkl::transpose transb = oneapi::mkl::transpose::nontrans;

    printf("A\n");
    print_matrix(m, k, A.data(), lda);
    printf("=====\n");

    printf("B\n");
    print_matrix(k, n, B.data(), ldb);
    printf("=====\n");

    /* step 1: create cublas handle, bind a stream */
    CUBLAS_CHECK(DPCT_CHECK_ERROR(cublasH = new dpct::blas::descriptor()));

    /*
    DPCT1025:2: The SYCL queue is created ignoring the flag and priority
    options.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(stream = dev_ct1.create_queue()));
    CUBLAS_CHECK(DPCT_CHECK_ERROR(cublasH->set_queue(stream)));

    /* step 2: copy data to device */
    CUDA_CHECK(DPCT_CHECK_ERROR(d_A = (data_type *)sycl::malloc_device(
                                    sizeof(data_type) * A.size(), q_ct1)));
    CUDA_CHECK(DPCT_CHECK_ERROR(d_B = (data_type *)sycl::malloc_device(
                                    sizeof(data_type) * B.size(), q_ct1)));
    CUDA_CHECK(DPCT_CHECK_ERROR(d_C = (data_type *)sycl::malloc_device(
                                    sizeof(data_type) * C.size(), q_ct1)));

    /*
    DPCT1124:3: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(
        stream->memcpy(d_A, A.data(), sizeof(data_type) * A.size())));
    /*
    DPCT1124:4: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(
        stream->memcpy(d_B, B.data(), sizeof(data_type) * B.size())));

    /* step 3: compute */
    CUBLAS_CHECK(DPCT_CHECK_ERROR(oneapi::mkl::blas::column_major::gemm(
        cublasH->get_queue(), transa, transb, m, n, k, alpha, d_A, lda, d_B,
        ldb, beta, d_C, ldc)));

    /* step 4: copy data to host */
    /*
    DPCT1124:5: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    CUDA_CHECK(DPCT_CHECK_ERROR(
        stream->memcpy(C.data(), d_C, sizeof(data_type) * C.size())));

    CUDA_CHECK(DPCT_CHECK_ERROR(stream->wait()));

    /*
     *   C = | 23.0 | 31.0 |
     *       | 34.0 | 46.0 |
     */

    printf("C\n");
    print_matrix(m, n, C.data(), ldc);
    printf("=====\n");

    /* free resources */
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_A, q_ct1)));
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_B, q_ct1)));
    CUDA_CHECK(DPCT_CHECK_ERROR(dpct::dpct_free(d_C, q_ct1)));

    CUBLAS_CHECK(DPCT_CHECK_ERROR(delete (cublasH)));

    CUDA_CHECK(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(stream)));

    CUDA_CHECK(DPCT_CHECK_ERROR(dev_ct1.reset()));

    return EXIT_SUCCESS;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

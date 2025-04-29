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
#include <complex>
#include <iostream>
#include <vector>
#include <dpct/fft_utils.hpp>

#include "cufft_utils.h"

int main(int argc, char *argv[]) {
    dpct::device_ext &dev_ct1 = dpct::get_current_device();
    sycl::queue &q_ct1 = dev_ct1.in_order_queue();
    dpct::fft::fft_engine_ptr plan;
    dpct::queue_ptr stream = &q_ct1;

    int fft_size = 8;
    int batch_size = 2;
    int element_count = batch_size * fft_size;

    using scalar_type = float;
    using data_type = std::complex<scalar_type>;

    std::vector<data_type> data(element_count, 0);

    for (int i = 0; i < element_count; i++) {
        data[i] = data_type(i, -i);
    }

    std::printf("Input array:\n");
    for (auto &i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    sycl::float2 *d_data = nullptr;

    CUFFT_CALL(DPCT_CHECK_ERROR(plan = dpct::fft::fft_engine::create()));
    CUFFT_CALL(DPCT_CHECK_ERROR(
        plan = dpct::fft::fft_engine::create(
            &q_ct1, fft_size,
            dpct::fft::fft_type::complex_float_to_complex_float, batch_size)));

    /*
    DPCT1025:0: The SYCL queue is created ignoring the flag and priority
    options.
    */
    CUDA_RT_CALL(DPCT_CHECK_ERROR(stream = dev_ct1.create_queue()));
    CUFFT_CALL(DPCT_CHECK_ERROR(plan->set_queue(stream)));

    // Create device data arrays
    CUDA_RT_CALL(DPCT_CHECK_ERROR(d_data = (sycl::float2 *)sycl::malloc_device(
                                      sizeof(data_type) * data.size(), q_ct1)));
    /*
    DPCT1124:2: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    CUDA_RT_CALL(DPCT_CHECK_ERROR(
        stream->memcpy(d_data, data.data(), sizeof(data_type) * data.size())));

    /*
     * Note:
     *  Identical pointers to data and output arrays implies in-place transformation
     */
    CUFFT_CALL(DPCT_CHECK_ERROR((plan->compute<sycl::float2, sycl::float2>(
        d_data, d_data, dpct::fft::fft_direction::forward))));

    // Normalize the data
    stream->submit([&](sycl::handler &cgh) {
        auto fft_size_ct2 = 1.f / fft_size;

        cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, 128),
                                           sycl::range<3>(1, 1, 128)),
                         [=](sycl::nd_item<3> item_ct1) {
                             scaling_kernel(d_data, element_count, fft_size_ct2,
                                            item_ct1);
                         });
    });

    // The original data should be recovered after Forward FFT, normalization and inverse FFT
    CUFFT_CALL(DPCT_CHECK_ERROR((plan->compute<sycl::float2, sycl::float2>(
        d_data, d_data, dpct::fft::fft_direction::backward))));

    /*
    DPCT1124:3: cudaMemcpyAsync is migrated to asynchronous memcpy API. While
    the origin API might be synchronous, it depends on the type of operand
    memory, so you may need to call wait() on event return by memcpy API to
    ensure synchronization behavior.
    */
    CUDA_RT_CALL(DPCT_CHECK_ERROR(
        stream->memcpy(data.data(), d_data, sizeof(data_type) * data.size())));

    CUDA_RT_CALL(DPCT_CHECK_ERROR(stream->wait()));

    std::printf("Output array after Forward FFT, Normalization, and Inverse FFT :\n");
    for (auto &i : data) {
        std::printf("%f + %fj\n", i.real(), i.imag());
    }
    std::printf("=====\n");

    /* free resources */
    CUDA_RT_CALL(DPCT_CHECK_ERROR(dpct::dpct_free(d_data, q_ct1)))

    CUFFT_CALL(DPCT_CHECK_ERROR(dpct::fft::fft_engine::destroy(plan)));

    CUDA_RT_CALL(DPCT_CHECK_ERROR(dev_ct1.destroy_queue(stream)));

    CUDA_RT_CALL(DPCT_CHECK_ERROR(dev_ct1.reset()));

    return EXIT_SUCCESS;
}

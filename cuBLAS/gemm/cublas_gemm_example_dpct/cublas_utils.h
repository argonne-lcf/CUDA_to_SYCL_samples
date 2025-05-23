/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#pragma once

#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <cmath>
#include <functional>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <dpct/lib_common_utils.hpp>

#include <complex>

// CUDA API error checking
/*
DPCT1001:0: The statement could not be removed.
*/
/*
DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
*/
#define CUDA_CHECK(err)                                                        \
    do {                                                                       \
        dpct::err0 err_ = (err);                                               \
        if (err_ != 0) {                                                       \
            std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
            throw std::runtime_error("CUDA error");                            \
        }                                                                      \
    } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                      \
    do {                                                                       \
        int err_ = (err);                                                      \
        if (err_ != 0) {                                                       \
            std::printf("cublas error %d at %s:%d\n", err_, __FILE__,          \
                        __LINE__);                                             \
            throw std::runtime_error("cublas error");                          \
        }                                                                      \
    } while (0)

// memory alignment
#define ALIGN_TO(A, B) (((A + B - 1) / B) * B)

// device memory pitch alignment
static const size_t device_alignment = 32;

// type traits
template <typename T> struct traits;

template <> struct traits<float> {
    // scalar type
    typedef float T;
    typedef T S;

    static constexpr T zero = 0.f;
    static constexpr dpct::library_data_t cuda_data_type =
        dpct::library_data_t::real_float;

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <> struct traits<double> {
    // scalar type
    typedef double T;
    typedef T S;

    static constexpr T zero = 0.;
    static constexpr dpct::library_data_t cuda_data_type =
        dpct::library_data_t::real_double;

    inline static S abs(T val) { return fabs(val); }

    template <typename RNG> inline static T rand(RNG &gen) { return (S)gen(); }

    inline static T add(T a, T b) { return a + b; }

    inline static T mul(T v, double f) { return v * f; }
};

template <> struct traits<sycl::float2> {
    // scalar type
    typedef float S;
    typedef sycl::float2 T;

    static constexpr T zero = {0.f, 0.f};
    static constexpr dpct::library_data_t cuda_data_type =
        dpct::library_data_t::complex_float;

    inline static S abs(T val) { return dpct::cabs<float>(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return sycl::float2((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return a + b; }
    inline static T add(T a, S b) { return a + sycl::float2(b, 0.f); }

    inline static T mul(T v, double f) {
        return sycl::float2(v.x() * f, v.y() * f);
    }
};

template <> struct traits<sycl::double2> {
    // scalar type
    typedef double S;
    typedef sycl::double2 T;

    static constexpr T zero = {0., 0.};
    static constexpr dpct::library_data_t cuda_data_type =
        dpct::library_data_t::complex_double;

    inline static S abs(T val) { return dpct::cabs<double>(val); }

    template <typename RNG> inline static T rand(RNG &gen) {
        return sycl::double2((S)gen(), (S)gen());
    }

    inline static T add(T a, T b) { return a + b; }
    inline static T add(T a, S b) { return a + sycl::double2(b, 0.); }

    inline static T mul(T v, double f) {
        return sycl::double2(v.x() * f, v.y() * f);
    }
};

template <typename T> void print_matrix(const int &m, const int &n, const T *A, const int &lda);

template <> void print_matrix(const int &m, const int &n, const float *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <> void print_matrix(const int &m, const int &n, const double *A, const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f ", A[j * lda + i]);
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const sycl::float2 *A,
                  const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x(),
                        A[j * lda + i].y());
        }
        std::printf("\n");
    }
}

template <>
void print_matrix(const int &m, const int &n, const sycl::double2 *A,
                  const int &lda) {
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            std::printf("%0.2f + %0.2fj ", A[j * lda + i].x(),
                        A[j * lda + i].y());
        }
        std::printf("\n");
    }
}

template <typename T>
void print_packed_matrix(oneapi::mkl::uplo uplo, const int &n, const T *A);

template <>
void print_packed_matrix(oneapi::mkl::uplo uplo, const int &n, const float *A) {
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((uplo == oneapi::mkl::uplo::upper && j >= i) ||
                (uplo == oneapi::mkl::uplo::lower && j <= i)) {
                std::printf("%6.2f ", A[off++]);
            } else if (uplo == oneapi::mkl::uplo::upper) {
                std::printf("       ");
            }
        }
        std::printf("\n");
    }
}

template <>
void print_packed_matrix(oneapi::mkl::uplo uplo, const int &n,
                         const double *A) {
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((uplo == oneapi::mkl::uplo::upper && j >= i) ||
                (uplo == oneapi::mkl::uplo::lower && j <= i)) {
                std::printf("%6.2f ", A[off++]);
            } else if (uplo == oneapi::mkl::uplo::upper) {
                std::printf("       ");
            }
        }
        std::printf("\n");
    }
}

template <>
void print_packed_matrix(oneapi::mkl::uplo uplo, const int &n,
                         const sycl::float2 *A) {
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((uplo == oneapi::mkl::uplo::upper && j >= i) ||
                (uplo == oneapi::mkl::uplo::lower && j <= i)) {
                std::printf("%6.2f + %6.2fj ", A[off].x(), A[off].y());
                off++;
            } else if (uplo == oneapi::mkl::uplo::upper) {
                std::printf("                 ");
            }
        }
        std::printf("\n");
    }
}

template <>
void print_packed_matrix(oneapi::mkl::uplo uplo, const int &n,
                         const sycl::double2 *A) {
    size_t off = 0;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if ((uplo == oneapi::mkl::uplo::upper && j >= i) ||
                (uplo == oneapi::mkl::uplo::lower && j <= i)) {
                std::printf("%6.2f + %6.2fj ", A[off].x(), A[off].y());
                off++;
            } else if (uplo == oneapi::mkl::uplo::upper) {
                std::printf("                 ");
            }
        }
        std::printf("\n");
    }
}

template <typename T> void print_vector(const int &m, const T *A);

template <> void print_vector(const int &m, const float *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f ", A[i]);
    }
    std::printf("\n");
}

template <> void print_vector(const int &m, const double *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f ", A[i]);
    }
    std::printf("\n");
}

template <> void print_vector(const int &m, const sycl::float2 *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f + %0.2fj ", A[i].x(), A[i].y());
    }
    std::printf("\n");
}

template <> void print_vector(const int &m, const sycl::double2 *A) {
    for (int i = 0; i < m; i++) {
        std::printf("%0.2f + %0.2fj ", A[i].x(), A[i].y());
    }
    std::printf("\n");
}

// Returns cudaDataType value as defined in library_types.h for the string
// containing type name
dpct::library_data_t get_cuda_library_type(std::string type_string) {
    if (type_string.compare("CUDA_R_16F") == 0)
        return dpct::library_data_t::real_half;
    else if (type_string.compare("CUDA_C_16F") == 0)
        return dpct::library_data_t::complex_half;
    else if (type_string.compare("CUDA_R_32F") == 0)
        return dpct::library_data_t::real_float;
    else if (type_string.compare("CUDA_C_32F") == 0)
        return dpct::library_data_t::complex_float;
    else if (type_string.compare("CUDA_R_64F") == 0)
        return dpct::library_data_t::real_double;
    else if (type_string.compare("CUDA_C_64F") == 0)
        return dpct::library_data_t::complex_double;
    else if (type_string.compare("CUDA_R_8I") == 0)
        return dpct::library_data_t::real_int8;
    else if (type_string.compare("CUDA_C_8I") == 0)
        return dpct::library_data_t::complex_int8;
    else if (type_string.compare("CUDA_R_8U") == 0)
        return dpct::library_data_t::real_uint8;
    else if (type_string.compare("CUDA_C_8U") == 0)
        return dpct::library_data_t::complex_uint8;
    else if (type_string.compare("CUDA_R_32I") == 0)
        return dpct::library_data_t::real_int32;
    else if (type_string.compare("CUDA_C_32I") == 0)
        return dpct::library_data_t::complex_int32;
    else if (type_string.compare("CUDA_R_32U") == 0)
        return dpct::library_data_t::real_uint32;
    else if (type_string.compare("CUDA_C_32U") == 0)
        return dpct::library_data_t::complex_uint32;
    else
        throw std::runtime_error("Unknown CUDA datatype");
}

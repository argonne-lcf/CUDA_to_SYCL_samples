#include <sycl/sycl.hpp>
#include <oneapi/mkl.hpp>
#include <complex>
#include <iostream>

using namespace sycl;
using scalar_type = float;
using data_type = std::complex<scalar_type>;

int main() {
    const int fft_size = 8;
    const int batch_size = 2;
    const int element_count = fft_size * batch_size;

    // Create DPC++ queue
    queue q{gpu_selector_v};

    // Allocate shared memory (USM)
    data_type* data = malloc_shared<data_type>(element_count, q);

    // Initialize input data
    for (int i = 0; i < element_count; i++) {
        data[i] = data_type(i, -i);
    }

    std::cout << "Input array:\n";
    for (int i = 0; i < element_count; ++i)
        std::printf("%f + %fj\n", data[i].real(), data[i].imag());
    std::cout << "=====\n";

    // Create FFT descriptor for C2C
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE,
                                 oneapi::mkl::dft::domain::COMPLEX> fft_desc(fft_size);
    fft_desc.set_value(oneapi::mkl::dft::config_param::NUMBER_OF_TRANSFORMS, batch_size);
    fft_desc.set_value(oneapi::mkl::dft::config_param::PLACEMENT,
                       oneapi::mkl::dft::config_value::INPLACE);
    fft_desc.set_value(oneapi::mkl::dft::config_param::FWD_STRIDES, std::vector<std::int64_t>{0, 1});
    fft_desc.set_value(oneapi::mkl::dft::config_param::BWD_STRIDES, std::vector<std::int64_t>{0, 1});
    fft_desc.set_value(oneapi::mkl::dft::config_param::FWD_DISTANCE, fft_size);
    fft_desc.set_value(oneapi::mkl::dft::config_param::BWD_DISTANCE, fft_size);
    fft_desc.commit(q);

    // Execute forward FFT
    oneapi::mkl::dft::compute_forward(fft_desc, data).wait();

    // Normalize output manually (1 / fft_size)
    q.parallel_for(range<1>(element_count), [=](id<1> i) {
        data[i] *= (1.0f / fft_size);
    }).wait();

    // Execute inverse FFT
    oneapi::mkl::dft::compute_backward(fft_desc, data).wait();

    std::cout << "Output array after Forward FFT, Normalization, and Inverse FFT:\n";
    for (int i = 0; i < element_count; ++i)
        std::printf("%f + %fj\n", data[i].real(), data[i].imag());
    std::cout << "=====\n";

    // Clean up
    free(data, q);

    return 0;
}

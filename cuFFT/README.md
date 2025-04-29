# cuFFT Library - APIs Examples

## Description

This folder demonstrates cuFFT APIs usage.

[cuFFT API Documentation](https://docs.nvidia.com/cuda/cufft/index.html?highlight=cufftExecC2C#c.cufftExecC2C)
[oneMKL API Documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/fourier-transform-functions.html)


## Description

This folder demonstrates how to port cuFFT APIs to Intel's equivalent via DPCT/SYCLomatic tool.
The source code involves classic 1D FFT using C2C operation using cuFFT API: 

[cufft API Documentation](https://docs.nvidia.com/cuda/cufft/index.html#cufft-t-gemm)
[SYCL oneMKL API Documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/gemm.html#GEMM-USM-VERSION)

## How to port
```bash
cd 1d_c2c
dpct --cuda-include-path=$SITE_CUDA_HEADERS_DIR/include 1d_c2c_cuda_cufft_example.cpp --out-root=1d_c2c_dpct_example
```

## How to compile
```bash
cd 1d_c2c_dpct_example
icpx -fsycl -qmkl 1d_c2c_cuda_cufft_example.cpp.dp.cpp -o 1d_c2c_example.cpp.dp.out
```

## How to run the executable
```bash
./1d_c2c_example.cpp.dp.out
```

## An optimized SYCL version

Also an optimized SYCL version using SYCL specification APIs with oneMKL USM APIs was also provided in `1d_c2c_sycl_onemkl_example.cpp`.
This does not use `dpct` header or it's APIs. The idea is how to migrate a DPCT source code to much stable/standard SYCL source code.
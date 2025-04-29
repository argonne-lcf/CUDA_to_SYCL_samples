# cuBLAS Library - APIs Examples

## Description

This folder demonstrates how to port cuBLAS APIs to Intel's equivalent via DPCT/SYCLomatic tool.
The source code involves classic GEMM operation using cuBLAS: 

[cuBLAS API Documentation](https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-gemm)
[SYCL oneMKL API Documentation](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-dpcpp/2023-0/gemm.html#GEMM-USM-VERSION)

## How to port
```bash
cd gemm
dpct --cuda-include-path=$SITE_CUDA_HEADERS_DIR/include cublas_gemm_example.cu --out-root=cublas_gemm_example_dpct
```

## How to compile
```bash
icpx -fsycl -qmkl cublas_gemm_example.dp.cpp -o cublas_gemm_example.dp.out
```

## How to run the executable
```bash
./cublas_gemm_example.dp.out
```

## An optimized SYCL version

Also an optimized SYCL version using SYCL specification APIs with oneMKL USM APIs was also provided in `sycl_onemkl_example.cpp`.
This does not use `dpct` header or it's APIs. The idea is how to migrate a DPCT source code to much stable/standard SYCL source code.
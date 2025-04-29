# Transitioning from CUDA to SYCL

Join us on April 29, 2025, for a webinar covering the process of porting CUDA code to SYCL, with a focus on high-performance math libraries like cuBLAS and cuFFT. ALCF's Thomas Applencourt and Abhishek Bagusetty will discuss key challenges, such as differences in API, memory management, and execution models, and provide strategies for achieving portability and performance. 

Special attention will be given to common pitfalls in porting, including synchronization issues and device memory handling. Additionally, we'll explore how to optimize SYCL code for Aurora's advanced architecture, highlighting techniques to avoid bottlenecks, map kernels efficiently, and leverage multi-level parallelism. Through case studies and practical examples, this talk will guide developers transitioning CUDA applications to SYCL without sacrificing performance, particularly on heterogeneous platforms like Aurora.

Welcome to the ALCF Aurora setup guide! This document provides step-by-step instructions for accessing Aurora, setting up your development environment, and cloning the necessary project repositories.

---

## 1. Accessing ALCF Aurora

To access Aurora, you must have:
- An approved project allocation at [ALCF](https://www.alcf.anl.gov/)
- A valid ALCF user account
- Duo two-factor authentication set up

**SSH Access:**
```bash
ssh <your_username>@aurora.alcf.anl.gov
```
If this is your first login, you may need to complete initial key setup and security procedures. For more details, see the [ALCF Accounts and Access Guide](https://www.alcf.anl.gov/support-center/finding-your-way/accounts-access).

---

## 2. Setting Up Your Environment

After logging in, you must load the appropriate modules to configure the programming environment for Aurora. By default, all necessary modules for Intel's oneAPI are available. Additionally, users must set up the CUDA headers to enable proper operation of porting tools from CUDA to SYCL.

### Load Environment Modules
```bash
module use /soft/modulefiles
module load cmake
module load headers/cuda/12.0.0
```

## 3. Build Instructions

Typical build workflow, manually working with each file:

```bash
cd CUDA_to_SYCL_samples
dpct --extra-arg="-I /opt/aurora/24.347.0/oneapi/compiler/2025.0/opt/compiler/include" --cuda-include-path=$SITE_CUDA_HEADERS_DIR/include hello_cuda_affinity.c --out-root=dpct_output

cd dpct_output
mpicxx -fsycl -qopenmp hello_cuda_affinity.c.dp.cpp -o hello_cuda_affinity_v1.out
```

Make sure to adjust compiler options if you need to use MPI and/or SYCL backends.

---

## 4. Running examples on Aurora

```bash
$ export OMP_NUM_THREADS=1
$ export CPU_BIND_SCHEME="--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100"
$ mpiexec -n 12 -ppn 12 ${CPU_BIND_SCHEME} gpu_tile_compact.sh ./hello_cuda_affinity_v1.out | sort

```

---

## 5. Need Help? Contact Us

For any technical questions or help setting up:

- **Abhishek Bagusetty**  
  - Email: abagusetty@anl.gov  
  - GitHub: [abagusetty](https://github.com/abagusetty)

- **Thomas Applencourt**  
  - Email: tapplenc@anl.gov  
  - GitHub: [tapplencourt](https://github.com/tapplencourt)

Feel free to reach out if you encounter issues with access, setup, builds, or performance tuning!

---

## Additional Resources

- [ALCF Aurora User Guide](https://docs.alcf.anl.gov/aurora/)
- [Intel oneAPI Documentation](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)

---

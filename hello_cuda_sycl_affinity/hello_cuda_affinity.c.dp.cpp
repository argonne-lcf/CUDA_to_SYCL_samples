#include <sycl/sycl.hpp>
#include <dpct/dpct.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <string.h>
#include <mpi.h>
#include <sched.h>
#include <sys/syscall.h>
#include <omp.h>
#include <unistd.h>

//Compile: CC -L/opt/nvidia/hpc_sdk/Linux_x86_64/23.9/cuda/12.2/targets/x86_64-linux/lib -lcudart hello_cuda_affinity.c -o hello_cuda_affinity.out
//Run: mpiexec -np 4 -ppn 4 --depth=4 --cpu-bind depth ./set_polaris_affinity.sh ./hello_cuda_affinity.out 

std::string get_cpu_affinity_range() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    pid_t pid = syscall(SYS_gettid);
    if (sched_getaffinity(pid, sizeof(mask), &mask) == -1) {
        perror("sched_getaffinity");
        return "";
    }

    std::string result;
    int i = 0;
    while (i < CPU_SETSIZE) {
        if (CPU_ISSET(i, &mask)) {
            int start = i;
            while (i + 1 < CPU_SETSIZE && CPU_ISSET(i + 1, &mask)) ++i;
            if (!result.empty()) result += ",";
            result += (start == i) ? std::to_string(start)
                                   : std::to_string(start) + "-" + std::to_string(i);
        }
        ++i;
    }

    return result.empty() ? "No CPUs found in affinity mask." : result;
}

int main(int argc, char *argv[]) try {

    MPI_Init(&argc, &argv);

    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char name[MPI_MAX_PROCESSOR_NAME];
    int resultlength;
    MPI_Get_processor_name(name, &resultlength);

    // If CUDA_VISIBLE_DEVICES is set, capture visible GPUs
    const char* gpu_id_list;
    const char* cuda_visible_devices = getenv("CUDA_VISIBLE_DEVICES");
    if (cuda_visible_devices == NULL) {
        gpu_id_list = "N/A";
    } else {
        gpu_id_list = cuda_visible_devices;
    }

    // Find how many GPUs CUDA runtime sees
    int num_devices = 0;
    dpct::err0 err = DPCT_CHECK_ERROR(num_devices = dpct::device_count());
    /*
    DPCT1000:1: Error handling if-stmt was detected but could not be rewritten.
    */
    if (err != 0) {
        /*
        DPCT1001:0: The statement could not be removed.
        */
        num_devices = 0;
    }

    std::string hwthread;
    int thread_id = 0;
    std::string busid = "";
    std::string busid_list = "";
    std::string rt_gpu_id_list = "";

    // Loop over available GPUs for this MPI rank
    for (int i = 0; i < num_devices; i++) {
        dpct::device_info prop;
        dpct::get_device(i).get_device_info(prop);

        // Build runtime GPU ID list
        if (i > 0) rt_gpu_id_list.append(",");
        rt_gpu_id_list.append(std::to_string(i));

        // Extract Bus ID
        char bus_id_str[8];
        /*
        DPCT1051:2: SYCL does not support a device property functionally
        compatible with pciBusID. It was migrated to -1. You may need to adjust
        the value of -1 for the specific device.
        */
        snprintf(bus_id_str, sizeof(bus_id_str), "%02x", -1);

        if (i > 0) busid_list.append(",");
        busid_list.append(bus_id_str);
    }

#pragma omp parallel default(shared) private(hwthread, thread_id)
    {
#pragma omp critical
        {
            thread_id = omp_get_thread_num();
            hwthread = get_cpu_affinity_range();
            int running_cpu = sched_getcpu();

            printf("MPI %03d - OMP %03d - HWT %s (Running on: %03d) - Node %s - RT_GPU_ID %s - GPU_ID %s - Bus_ID %s\n",
                   rank, thread_id, hwthread.c_str(), running_cpu, name,
                   rt_gpu_id_list.c_str(), gpu_id_list, busid_list.c_str());
        }
    }

    MPI_Finalize();
    return 0;
}
catch (sycl::exception const &exc) {
  std::cerr << exc.what() << "Exception caught at file:" << __FILE__
            << ", line:" << __LINE__ << std::endl;
  std::exit(1);
}

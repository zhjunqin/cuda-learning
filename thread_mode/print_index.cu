#include <cuda_runtime.h>
#include <assert.h>
#include <stdio.h>

__global__ void printIndex(void) {
    printf("threadIdx(%d, %d, %d); blockIdx(%d, %d, %d); blockDim(%d, %d, %d); gridDim(%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}

int main(void) {

    // Call the kernel
    dim3 block(2,3);
    dim3 grid(2);
    printIndex<<<grid, block>>>();

    cudaDeviceReset();

    return 0;
}

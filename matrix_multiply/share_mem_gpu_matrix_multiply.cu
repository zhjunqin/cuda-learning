#include <assert.h>
#include "Matrix.h"
#include "Common.h"

#define N 1024

void matrixMultiply(Matrix<float> m_a, Matrix<float> m_b, Matrix<float> m_c) {
    for (size_t i = 0; i < N; i++) {
	for (size_t j = 0; j < N; j++) {
            float temp = 0;
	    for (size_t k=0; k < N; k++) {
                temp += m_a.get(i, k) * m_b.get(k, j);
	    }
	    m_c.set(i, j, temp);
	}
    }
}

__global__ void simpleMatrixMultiply(Matrix<float> d_a, Matrix<float> d_b, Matrix<float> d_c) {
    size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    size_t iy = threadIdx.y + blockIdx.y * blockDim.y; 
    float temp = 0;
    if (ix < N && iy < N) {
        for (size_t k = 0; k < N; k++) {
            temp += d_a.get(iy, k) * d_b.get(k, ix);
        }
        d_c.set(iy, ix, temp);
    }
}


__global__ void shareMemMatrixMultiply(Matrix<float> d_a, Matrix<float> d_b, Matrix<float> d_c) { 
    __shared__ float A_tile[32][32];
    __shared__ float B_tile[32][32];
    size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    size_t iy = threadIdx.y + blockIdx.y * blockDim.y; 

    float temp = 0;
    if (ix < N && iy < N) {
        for (size_t i = 0; i < 1024; i +=32) {
            A_tile[threadIdx.y][threadIdx.x] = d_a.get(iy, threadIdx.x + i);
            B_tile[threadIdx.y][threadIdx.x] = d_b.get(threadIdx.y + i, ix);

	    __syncthreads();
	
	    for (size_t j = 0; j < 32; j++) {
                temp += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
	    }
	    __syncthreads();
        }
        d_c.set(iy, ix, temp);
    }
}

__global__ void UnrollshareMemMatrixMultiply(Matrix<float> d_a, Matrix<float> d_b, Matrix<float> d_c) { 
    __shared__ float A_tile[32][32];
    __shared__ float B_tile[32][32];
    size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    size_t iy = threadIdx.y + blockIdx.y * blockDim.y; 

    float temp = 0;
    if (ix < N && iy < N) {
        for (size_t i = 0; i < 1024; i +=32) {
            A_tile[threadIdx.y][threadIdx.x] = d_a.get(iy, threadIdx.x + i);
            B_tile[threadIdx.y][threadIdx.x] = d_b.get(threadIdx.y + i, ix);

	    __syncthreads();
	
	    for (size_t j = 0; j < 32; j +=8) {
                temp += A_tile[threadIdx.y][j] * B_tile[j][threadIdx.x];
                temp += A_tile[threadIdx.y][j+1] * B_tile[j+1][threadIdx.x];
                temp += A_tile[threadIdx.y][j+2] * B_tile[j+2][threadIdx.x];
                temp += A_tile[threadIdx.y][j+3] * B_tile[j+3][threadIdx.x];
                temp += A_tile[threadIdx.y][j+4] * B_tile[j+4][threadIdx.x];
                temp += A_tile[threadIdx.y][j+5] * B_tile[j+5][threadIdx.x];
                temp += A_tile[threadIdx.y][j+6] * B_tile[j+6][threadIdx.x];
                temp += A_tile[threadIdx.y][j+7] * B_tile[j+7][threadIdx.x];
	    }
	    __syncthreads();
        }
        d_c.set(iy, ix, temp);
    }
}

void compareResults(Matrix<float> h_a, Matrix<float> h_b) {
    for (int i = 0; i < h_a.width; i++) {
        for (int j = 0; j < h_a.height; j++) {
            assert(h_a.get(i, j) == h_b.get(i, j));
        }
    }
}

void onHost(Matrix<float> h_a, Matrix<float> h_b, Matrix<float> h_c) {
    double start = seconds();
    matrixMultiply(h_a, h_b, h_c); 
    double total = seconds() - start;
    cout << "Host Matrix multiply time(second): " << total << std::endl;
}


void onDevice(Matrix<float> h_a, Matrix<float> h_b, Matrix<float> h_c,
	      Matrix<float> h_c_ref, size_t bytes) {
    // malloc GPU memory
    Matrix<float> d_a, d_b, d_c;
    d_a.init(N, N);
    d_b.init(N, N);
    d_c.init(N, N);

    CHECK(cudaMalloc((void **)(&d_a.elements), bytes));
    CHECK(cudaMalloc((void **)(&d_b.elements), bytes));
    CHECK(cudaMalloc((void **)(&d_c.elements), bytes));

    // copy from host memory to device
    CHECK(cudaMemcpy(d_a.elements, h_a.elements, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_b.elements, h_b.elements, bytes, cudaMemcpyHostToDevice));

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // warm up
    simpleMatrixMultiply<<<grid, block>>>(d_a, d_b, d_c);
    CHECK(cudaDeviceSynchronize());

    double start = seconds();
    simpleMatrixMultiply<<<grid, block>>>(d_a, d_b, d_c);
    CHECK(cudaDeviceSynchronize());
    double total = seconds() - start;
    CHECK(cudaMemcpy(h_c_ref.elements, d_c.elements, bytes, cudaMemcpyDeviceToHost));
    compareResults(h_c, h_c_ref);
    cout << "Simple Matrix multiply time(ms): " << total * 1000 << std::endl;
    //h_c.print();
     
    start = seconds();
    shareMemMatrixMultiply<<<grid, block>>>(d_a, d_b, d_c);
    CHECK(cudaDeviceSynchronize());
    total = seconds() - start;
    CHECK(cudaMemcpy(h_c_ref.elements, d_c.elements, bytes, cudaMemcpyDeviceToHost));
    compareResults(h_c, h_c_ref);
    cout << "Share mem Matrix multiply time(ms): " << total * 1000 << std::endl;

    start = seconds();
    UnrollshareMemMatrixMultiply<<<grid, block>>>(d_a, d_b, d_c);
    CHECK(cudaDeviceSynchronize());
    total = seconds() - start;
    CHECK(cudaMemcpy(h_c_ref.elements, d_c.elements, bytes, cudaMemcpyDeviceToHost));
    compareResults(h_c, h_c_ref);
    cout << "Unroll Share mem Matrix multiply time(ms): " << total * 1000 << std::endl;

    CHECK(cudaFree(d_a.elements));
    CHECK(cudaFree(d_b.elements));
    CHECK(cudaFree(d_c.elements));
}

int main() {
    Matrix<float> h_a, h_b, h_c;
    size_t total_bytes = N * N * sizeof(float);

    h_a.init(N, N);
    h_a.host_alloc();
    h_b.init(N, N);
    h_b.host_alloc();
    h_c.init(N, N);
    h_c.host_alloc();

    h_a.randomInit();
    h_b.randomInit();

    // Host Matrix multiply
    onHost(h_a, h_b, h_c);
    //h_c.print();

    // GPU Matrix multiply
    Matrix<float> h_c_ref;
    h_c_ref.init(N, N);
    h_c_ref.host_alloc();
    onDevice(h_a, h_b, h_c, h_c_ref, total_bytes);


    h_a.remove();
    h_b.remove();
    h_c.remove();
    h_c_ref.remove();
    return 0;
}
    


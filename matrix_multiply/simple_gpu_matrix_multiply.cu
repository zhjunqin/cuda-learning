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


void onDevice(Matrix<float> h_a, Matrix<float> h_b, Matrix<float> h_c, size_t bytes) {
    // malloc GPU memory
    //printf("bytes %d", bytes);
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

    //simpleMatrixMultiply<<<1024, 1024>>>(d_a, d_b, d_c);
    double start = seconds();

    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    simpleMatrixMultiply<<<grid, block>>>(d_a, d_b, d_c);
    CHECK(cudaDeviceSynchronize());
    double total = seconds() - start;
    cout << "Simple Matrix multiply time(ms): " << total * 1000 << std::endl;

    CHECK(cudaMemcpy(h_c.elements, d_c.elements, bytes, cudaMemcpyDeviceToHost));
    //h_c.print();

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
    onDevice(h_a, h_b, h_c_ref, total_bytes);

    compareResults(h_c, h_c_ref);

    h_a.remove();
    h_b.remove();
    h_c.remove();
    h_c_ref.remove();
    return 0;
}
    


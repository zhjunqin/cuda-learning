#ifndef GPU_MATRIX_H__
#define GPU_MATRIX_H__

#include <iostream>
using namespace std;

template <typename T>
struct Matrix {
    size_t width;
    size_t height;
    T* elements;

    __host__ void init(size_t row, size_t col) {
	width = col;
	height = row;
    }

    __host__ void host_alloc() {
	size_t total_bytes =  width * height * sizeof(T);
	elements = (T *)(malloc(total_bytes));
    }

    __host__ void remove() {
	if (elements)
            free(elements);
    }

    __device__ __host__ T get(size_t row, size_t col) {
        return elements[row * width + col];
    }

    __device__ __host__ void set(size_t row, size_t col, T value) {
        elements[row * width + col] = value;
    }

    __host__ void print() {
        cout << "\n" << endl;
        for (size_t i = 0; i < height; i++) {
            for (size_t j = 0; j < width; j++) {
                cout << get(i, j) << " ";
            }
            cout << "\n";
        }
    }

    __host__ void constantInit(T val) {
	for(size_t i = 0; i < height; i++) {
	    for(size_t j = 0; j < width; j++) {
		set(i, j, val);
	    }
	}
    }

    __host__ void randomInit() {
	for(size_t i = 0; i < height; i++) {
	    for(size_t j = 0; j < width; j++) {
		set(i, j, (T)(rand() & 0xFF ));
	    }
	}
    }
};

#endif /* GPU_MATRIX_H__ */

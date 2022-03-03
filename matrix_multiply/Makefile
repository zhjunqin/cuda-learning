CU_APPS=host_matrix_multiply \
	simple_gpu_matrix_multiply \
	share_mem_gpu_matrix_multiply

all: ${CU_APPS}

%: %.cu
	nvcc -o $@ $<

clean:
	rm -f ${CU_APPS}


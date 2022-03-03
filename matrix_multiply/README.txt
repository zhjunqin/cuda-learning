# Cuda 矩阵乘

一个从简单的 Cuda 的逐步优化，优化步骤

- 主机矩阵乘
    - 对应文件 `host_matrix_multiply.cu`
- 每个 thread 拷贝显存做逐行加
    - 对应文件 `simple_gpu_matrix_multiply.cu`
- 使用 share memory tile 来临时保存，sync 后再逐行加
    - 对应文件 `share_mem_gpu_matrix_multiply.cu`
- 使用展开循环来提升性能
    - 对应文件 `share_mem_gpu_matrix_multiply.cu`


## 最后对比的结果

```
Host Matrix multiply time(second): 12.2588
Simple Matrix multiply time(ms): 1.1282
Share mem Matrix multiply time(ms): 0.640154
Unroll Share mem Matrix multiply time(ms): 0.635862
```

```
# nvprof ./share_mem_gpu_matrix_multiply
Host Matrix multiply time(second): 12.3612
==4964== NVPROF is profiling process 4964, command: ./share_mem_gpu_matrix_multiply
Simple Matrix multiply time(ms): 1.1332
Share mem Matrix multiply time(ms): 0.649929
Unroll Share mem Matrix multiply time(ms): 0.64683
==4964== Profiling application: ./share_mem_gpu_matrix_multiply
==4964== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   31.85%  2.2428ms         2  1.1214ms  1.1208ms  1.1221ms  simpleMatrixMultiply(Matrix<float>, Matrix<float>, Matrix<float>)
                   26.67%  1.8778ms         3  625.95us  381.60us  1.0192ms  [CUDA memcpy DtoH]
                   23.81%  1.6762ms         2  838.12us  799.67us  876.57us  [CUDA memcpy HtoD]
                    8.85%  622.91us         1  622.91us  622.91us  622.91us  shareMemMatrixMultiply(Matrix<float>, Matrix<float>, Matrix<float>)
                    8.82%  621.21us         1  621.21us  621.21us  621.21us  UnrollshareMemMatrixMultiply(Matrix<float>, Matrix<float>, Matrix<float>)
      API calls:   94.63%  265.28ms         3  88.428ms  195.34us  264.88ms  cudaMalloc
                    2.13%  5.9825ms         5  1.1965ms  550.70us  2.4087ms  cudaMemcpy
                    1.26%  3.5440ms         4  886.00us  623.79us  1.1708ms  cudaDeviceSynchronize
                    0.87%  2.4262ms         2  1.2131ms  1.2105ms  1.2157ms  cuDeviceTotalMem
                    0.78%  2.1978ms       202  10.880us     126ns  1.2168ms  cuDeviceGetAttribute
                    0.24%  660.51us         3  220.17us  189.71us  274.58us  cudaFree
                    0.05%  129.08us         2  64.541us  59.120us  69.962us  cuDeviceGetName
                    0.03%  92.156us         4  23.039us  7.5190us  43.328us  cudaLaunchKernel
                    0.00%  8.1920us         2  4.0960us  2.6990us  5.4930us  cuDeviceGetPCIBusId
                    0.00%  1.4020us         4     350ns     127ns     685ns  cuDeviceGet
                    0.00%  1.1920us         3     397ns     151ns     710ns  cuDeviceGetCount
                    0.00%     621ns         2     310ns     286ns     335ns  cuDeviceGetUuid
```

这里手画一张 share memory 的时候 tile 保存的过程。

![](./assets/matrix_multply.png)


## 参考文献



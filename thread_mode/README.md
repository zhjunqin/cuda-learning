# 线程模型

## Grid，BLock, Thread

CUDA 明确了线程层次抽象的概念以便于组织线程。这是一个两层的线程层次结构，由线程块和线程块网格构成。

![pic](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png)

一个 Kernel 启动所产生的所有线程称为一个 Grid。同一个 Grid 中的所有线程共享相同的全局内存空间。一个网格由多个线程 block 构成，一个线程 block 包含一组线程。

同一个线程 block 内的线程协作可以通过如下方式实现：

    - 同步
    - 共享内存

不同 block 的线程不能协作。

线程依赖以下两个坐标变量来区分彼此：

    - blockIdx：线程块在 Grid 内的索引
    - threadIdx： block 内的线程索引

这些变量是 kernel 函数中预初始化的内置变量。这些变量是基于 uint3 定义的 CUDA 内置的向量类型，是一个包含 3 个无符号整数的结构，可以通过 x,y,z 三个字段来指定。

    - blockIdx.x, blockIdx.y, blockIdx.z
    - threadIdx.x, threadIdx.y, threadIdx.z


CUDA 可以组织三维的网格和块。网格和块的维度由下面两个内置变量指定：
    - blockDim：线程 block 的维度
    - gridDim：线程 Grid 的维度

他们是 dim3 类型的变量，是基于 unint3 定义的整数型变量，用来表示维度。当定义一个 dim3 类型的变量时，所有未指定的元素都被初始化为 1。

dim3 类型的变量中的每个组件可以通过它的 x, y, z 字段获得。
    - blockDim.x, blockDim.y, blockDim.z
    - gridDim.x, gridDim.x, gridDim.z

一些示例：
    - dim3 block(32); dim3 grid(8)，其中总共 8 个 block，每个 block 有 32 个线程，均为一维
    - dim3 block(32,32); dim3 grid(8,8)，其中总共 8*8 个 block，每个 block 有 32*32 个线程，均为二维
    - dim3 block(32,32); dim3 grid(16)，其中总共 16 个 block，每个 block 有 32*32 个线程

### 代码示例

在代码中打印 grid 和 block 的索引和维度

```

__global__ void printIndex(void) {
    printf("threadIdx(%d, %d, %d); blockIdx(%d, %d, %d); blockDim(%d, %d, %d); gridDim(%d, %d, %d)\n",
           threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
}



// Call the kernel
dim3 block(2,3);
dim3 grid(2);
printIndex<<<grid, block>>>();

```

打印结果如下：
总共两个 block，这两个 block 的 index 分别是 blockIdx(0, 0, 0) 和 blockIdx(1, 0, 0)。
每个 block 总共 2*3=6 个 thread，thread 的 index x 维度为 0-1，y 维度为 0-2
```

threadIdx(0, 0, 0); blockIdx(1, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(1, 0, 0); blockIdx(1, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(0, 1, 0); blockIdx(1, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(1, 1, 0); blockIdx(1, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(0, 2, 0); blockIdx(1, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(1, 2, 0); blockIdx(1, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)

threadIdx(0, 0, 0); blockIdx(0, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(1, 0, 0); blockIdx(0, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(0, 1, 0); blockIdx(0, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(1, 1, 0); blockIdx(0, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(0, 2, 0); blockIdx(0, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)
threadIdx(1, 2, 0); blockIdx(0, 0, 0); blockDim(2, 3, 1); gridDim(2, 1, 1)

```

## kernel 函数

CUDA kernel 调用是对 C 语言函数调用的延伸，`<<<>>>` 运算符内是 kernel 函数的执行配置。

__global__ kernel_name <<<grid, block>>>(argument list);

比如上面的 `printIndex<<<grid, block>>>();`

前面的函数类型限定符
- __global__： 在 device 端执行，可从 host 调用，必须有一个 void 返回类型
- __device__： 在 device 端执行，仅能从 device 端调用
- __host__：   在 host 端执行，仅能从 host 端调用


## kernal 函数的异步和同步

- 同步：host 向 device 提交任务（如kernel），在同步的情况下，主机将会阻塞，知道设备将所提交任务完成，并将控制权交回主机，然后会继续执行主机的程序；
- 异步：host 向 device 提交任务后，设备开始执行任务，并立刻将控制权交回主机，所以主机将不会阻塞，而是直接继续执行主机的程序，即在异步的情况下，主机不会等待设备执行任务完成；

如上的 kernel 函数调用 `printIndex<<<grid, block>>>();` 是异步的，host 执行不会阻塞，如果没有执行同步的调用，那么 Host 主程序会直接退出。

比如调用 `cudaDeviceSynchronize()`


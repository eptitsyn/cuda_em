
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(float *a, size_t size);

__device__ int qpart(float* A, int lo, int hi)
{
	float pivot = A[hi];
	int i = lo - 1;
	int j = hi + 1;

	for (int j = lo; j <= hi; ++j)
	{
		if(A[j] <= pivot)
		{
			i++;
			float tmp = A[i];
			A[i] = A[j];
			A[j] = tmp;
		}
	}
	float tmp = A[i+1];
	A[i+1] = A[hi];
	A[hi] = tmp;
	return(i + 1);
}
__device__ void q1sort(float* A, int lo, int hi)
{
	if (lo < hi)
	{
		int p = qpart(A, lo, hi);
		q1sort(A, lo, p - 1);
		q1sort(A, p + 1, hi);
	}
}

__global__ void sortKernel(float *a,const int size)
{
	q1sort(a, 0, 9);
}

int main()
{
    int arraySize = 10;
    float a[10] = { 1, 0, 7, 2, 8, 3, 9, 4, 5, 6 };
   
    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(a, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

	for (int i = 0; i < 10; ++i)
	{
		printf("%f, ", a[i]);
	}

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *a, size_t size)
{
    float *dev_a = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    // Launch a kernel on the GPU with one thread for each element.
    sortKernel<<<1, 1>>>(dev_a, size);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(a, dev_a, size * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_a);
    
    return cudaStatus;
}

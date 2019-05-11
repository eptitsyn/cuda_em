//qsort test
#include "qsort.cu"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

using namespace std;

int main(){
	cudaSetDevice(0);
	cudaDeviceReset();
	double h_sigma[15] = {1.4, -1, 0, 1.7, 0, 2.2, 2.3, -7, -1.5, 1.5, 1.9, 1.8, 0.5, 2.5, 2.8};
	double *d_sigma;
	cudaMalloc(&d_sigma, 15*sizeof(double));
	cudaMemcpy(d_sigma, h_sigma, 15*sizeof(double), cudaMemcpyHostToDevice);
	cdp_simple_quicksort<<<1,1>>>(d_sigma, 0, 10-1, 0);
	cudaDeviceSynchronize();
	cudaMemcpy(h_sigma, d_sigma, 15*sizeof(double), cudaMemcpyDeviceToHost);
	for (int i = 0; i < 15; ++i)
	{
		cout << h_sigma[i] << ", ";
	}
	cout << endl;
}
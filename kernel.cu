#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

#define M_SQ2PI 2.506628274631000502416
#define M_SQPId2 1.253314137315500251208
#define k 10
#define MAX_ITERATIONS 100
#define TOLERANCE 0.000001
#define DATA_LENGTH 1100
#define WINDOW_LENGTH 1040

static void HandleError(cudaError_t err,
                        const char* file,
                        int line)
{
	if (err != cudaSuccess)
	{
		printf("%s in %s at line %d\n", cudaGetErrorString(err),
		       file, line);
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))

typedef struct
{
	int width;
	int height;
	double* elements;
} Matrix;

// Get a matrix element
__device__ double GetElement(const Matrix A, int row, int col)
{
	return A.elements[row * A.width + col];
}

// Set a matrix element
__device__ void SetElement(Matrix A, int row, int col, double value)
{
	A.elements[row * A.width + col] = value;
}

__device__ double* GetSubData(double* data, int i)
{
	return &data[i];
}

double* readfromfile(const char* DATA_FILENAME, int data_length)
{
	ifstream inFile;
	inFile.open(DATA_FILENAME);
	if (!inFile) {
		cerr << "Unable to open file datafile";
		exit(1);   // call system to stop
	}
	double *data = (double*)malloc(data_length * sizeof(double));
	double x;
	for (int i = 0; i < data_length; ++i)
	{
		inFile >> x;
		data[i] = x;
	}
	inFile.close();
	return data;
};

double std_dev(double* data, int data_length)
{
	double sum = 0;
	for (int i = 0; i < data_length; ++i)
	{
		sum += data[i];
	}
	double mean = sum / (double)data_length;
	double differ;
	double varsum = 0;
	for (int i = 0; i < data_length; ++i)
	{
		differ = data[i] - mean;
		varsum += pow(differ, 2);
	}
	double Variance = varsum / (double)data_length;
	return sqrt(Variance);
}

double randomdouble()
{
	double r = (double)rand() / (double)RAND_MAX;
	return r;
}


void set_initial_guess(double* data, int data_length, double* theta)
{
	//pi
	for (int i = 0; i < k; ++i)
	{
		theta[i] = randomdouble() * 0.9 + 0.1;
	}
	double tsum = 0;
	for (int i = 0; i < k; ++i)
	{
		tsum += theta[i];
	}
	for (int i = 0; i < k; ++i)
	{
		theta[i] /= tsum;
	}
	/*
	for (int i = 0; i < k; ++i)
	{
		theta[i] = theta[i] * 0.9 + 0.1;
	}*/
	//mu
	for (int i = 0; i < k; ++i)
	{
		theta[i + k] = 0;
	}
	//sigma
	for (int i = 0; i < k; ++i)
	{
		theta[i + k * 2] = randomdouble() * 1.5 + 0.25 * std_dev(data, data_length);
	}
}

__global__ void e_step1(double* glob_data, int data_off, double* theta, int theta_off, Matrix w)
{
	double* pi = &theta[theta_off];
	double* mu = &theta[theta_off + k];
	double* sigma = &theta[theta_off + 2 * k];
	double* data = &glob_data[data_off];

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < w.width && j < w.height)
		if (sigma[j] != 0)
		{
			SetElement(w, j, i, pi[j] * exp(-(pow(data[i] - mu[j], 2)) / (2 * pow(sigma[j], 2))) / (M_SQ2PI * sigma[j]));
		}
		else
		{
			SetElement(w, j, i, data[i] == mu[j] ? pi[j] : 0);
		}
}

__global__ void e_step2(Matrix w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i < w.width)
	{
		double wsum = 0;
		for (int e = 0; e < w.height; ++e)
		{
			wsum += GetElement(w, e, i);
		}
		for (int e = 0; e < w.height; ++e)
		{
			if (wsum != 0)
				SetElement(w, e, i, GetElement(w, e, i) / wsum);
			else
				SetElement(w, e, i, 0);
		}
	}
}

//__device__ int compare(const void * a, const void * b)
//{
//	double fa = *(const double*)a;
//	double fb = *(const double*)b;
//	return (fa > fb) - (fa < fb);
//}



//__device__ int qpart(double* A, int lo, int hi)
//{
//	double pivot = A[lo + (hi - lo) / 2];
//	int i = lo - 1;
//	int j = hi + 1;
//
//	while(1)
//	{
//		i++;
//		while (A[i]<pivot)
//		{
//			j--;
//		}
//		while (A[j]>pivot)
//		{
//			if (i >= j) return j;
//		}
//		double tmp = A[i];
//		A[i] = A[j];
//		A[j] = A[i];
//	}
//}
//__device__ void q1sort(double* A, int lo, int hi)
//{
//	if (lo < hi)
//	{
//		int p = qpart(A, lo, hi);
//		q1sort(A, lo, p);
//		q1sort(A, p + 1, hi);
//	}
//}
//
//__device__ void quiksort(double* base, size_t num)
//{
//	q1sort(base, 0, num - 1);
//}


__global__ void m_step(double* glob_data, int data_off, double* theta, int theta_off, Matrix w)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (j < w.height)
	{
		double* pi = &theta[theta_off];
		double* mu = &theta[theta_off + k];
		double* sigma = &theta[theta_off + 2 * k];
		double* data = &glob_data[data_off];

		pi[j] = 0;
		mu[j] = 0;
		sigma[j] = 0;
		for (int e = 0; e < w.width; e++)
		{
			pi[j] += GetElement(w, j, e);
		}
		if (pi[j] != 0) {
			for (int e = 0; e < w.width; e++)
			{
				mu[j] += GetElement(w, j, e) * data[e];
			}
			mu[j] /= pi[j];


			for (int e = 0; e < w.width; e++)
			{
				sigma[j] += GetElement(w, j, e) * pow(data[e] - mu[j], 2);
			}
			sigma[j] /= pi[j];
			sigma[j] = sqrt(sigma[j]);

			pi[j] /= w.width;
		}
	}
}

__global__ void compute_ll(Matrix w, double* ll, double* ll2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i < w.width)
	{
		double llsum = 0;
		for (int e = 0; e < w.height; e++)
		{
			llsum += GetElement(w, e, i);
		}
		ll[i] = log(llsum);
		ll2[i] = llsum;
	}
}

__global__ void compute_ll2(Matrix w, double* ll)
{
	double llsum = 0;
	for (int e = 0; e < w.width; e++)
	{
		llsum += ll[e];
	}
	ll[0] = llsum;
}


cudaError_t em_algorithm(double* d_data, int data_off, const int data_length, double* d_theta, int theta_offset, double* h_theta,  bool debug)
{
	double* d_theta_loc = &d_theta[theta_offset];
	double* h_theta_loc = &h_theta[theta_offset];
	//size_t theta_size = ((data_length - window_size) / window_step )* 3 * k * sizeof(double);
	size_t theta_loc_size = 3 * k * sizeof(double);
	Matrix d_W;
	d_W.width = data_length;
	d_W.height = k;
	HANDLE_ERROR(cudaMalloc(&d_W.elements, d_W.width * d_W.height * sizeof(double)));

	double* d_ll;
	double* d_ll2;
	double* h_ll = (double*)malloc(sizeof(double) * data_length);
	double* h_ll2 = (double*)malloc(sizeof(double) * data_length);
	HANDLE_ERROR(cudaMalloc(&d_ll, data_length * sizeof(double)));
	HANDLE_ERROR(cudaMalloc(&d_ll2, data_length * sizeof(double)));

	dim3 dimBlock(16, k);
	dim3 dimGrid(data_length / dimBlock.x + 1, 1);

	dim3 dimBlocke2(16, 1);
	dim3 dimGride2(data_length / dimBlocke2.x + 1, 1);

	dim3 dimBlockM(1, k);

	dim3 dimBlockLL(16, 1);
	dim3 dimGridLL(data_length / dimBlockLL.x + 1, 1);


	double ll_old = 0;
	for (int i = 0; i < MAX_ITERATIONS; i++)
	{
		//printf("iter = %d, ", i);
		e_step1 << <dimGrid, dimBlock >> >(d_data, data_off, d_theta, theta_offset, d_W);

		compute_ll << <dimGridLL, dimBlockLL >> >(d_W, d_ll, d_ll2);
		compute_ll2 << <1, 1 >> >(d_W, d_ll);
		cudaDeviceSynchronize();

		HANDLE_ERROR(cudaMemcpy(h_ll, d_ll, data_length * sizeof(double), cudaMemcpyDeviceToHost));
		HANDLE_ERROR(cudaMemcpy(h_ll2, d_ll2, data_length *sizeof(double), cudaMemcpyDeviceToHost));

		double ll_new = h_ll[0];

		/*if (!isnormal(ll_new))
		{
			cudaMemcpy(h_theta_loc, d_theta_loc, theta_loc_size, cudaMemcpyDeviceToHost);
			printf("\n");
			for (int i =0 ; i < 3; ++i)
			{
				for (int j = 0; j < k; ++j)
				{
					printf("%f, ", h_theta_loc[i*k+j]);
				}
				printf("\n");
			}
			printf("\n");
		}*/
		//printf("ll = %f;\n", ll_new);
		if (isnan(ll_new))
		{
			HANDLE_ERROR(cudaMemcpy(h_theta_loc, d_theta_loc, 3*k* sizeof(double), cudaMemcpyDeviceToHost));
			for (int i = 0; i < k * 3; ++i)
			{
				printf("%f, ", h_theta_loc[i]);
				(i + 1) % 10 == 0 ? printf("\n") : printf("");
			}
			printf("\n");
			/*
			for (int i = 0; i < data_length; ++i)
			{
				printf("%f, ", h_ll2[i]);
			}
			printf("\n\nW= \n");

			*/
			Matrix h_W;
			h_W.width = data_length;
			h_W.height = k;
			h_W.elements = (double*)malloc(h_W.width * h_W.height * sizeof(double));
			cudaMemcpy(h_W.elements, d_W.elements, d_W.width * d_W.height * sizeof(double), cudaMemcpyDeviceToHost);

			for (int i = 0; i < data_length * k * 3; i++)
			{
				printf("%f, ", h_W.elements[i]);
				(i + 1) % 10 == 0 ? printf("\n") : printf("");
			}
			printf("\n");

			exit(1);
		}
		if (abs(ll_new - ll_old) < TOLERANCE)
		{
			printf("end em step %d", i);
			break;
		}
		ll_old = ll_new;


		e_step2 << <dimGride2, dimBlocke2 >> >(d_W);
		m_step << <1, dimBlockM >> >(d_data, data_off, d_theta, theta_offset, d_W);
		if (debug)
		{
			cudaMemcpy(h_theta_loc, d_theta_loc, theta_loc_size, cudaMemcpyDeviceToHost);
			for (int i = 0; i < 3; ++i)
			{
				for (int j = 0; j < k; ++j)
				{
					printf("%f, ", h_theta_loc[i * k + j]);
				}
				printf("\n");
			}
			printf("\n");
		}
	}

	cudaFree(d_ll);
	cudaFree(d_ll2);
	cudaFree(d_W.elements);
	return cudaSuccess;
}

__global__ void copythetatonext(double *theta, int theta_offset, int theta_length)
{
		for (int j = theta_offset; j < theta_offset+theta_length; ++j)
		{
			theta[j+theta_length] = theta[j];
		}
}

cudaError_t slsalgorithm(double* h_data, const int data_length, double* h_theta, const int window_size, const int window_step, const int generate_theta_each_step)
{
	double* d_data = 0;
	double* d_theta = 0;
	int theta_offset = 0;
	int data_off = 0;

	const int steps = (data_length - window_size + 1) / window_step;
	size_t theta_size = (steps * k * 3 * sizeof(double));


	cudaError_t cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	HANDLE_ERROR(cudaMalloc(&d_theta, theta_size));
	HANDLE_ERROR(cudaMalloc(&d_data, data_length * sizeof(double)));

	///
	//set inint guess
	switch (generate_theta_each_step)
	{
	default:
		{
			set_initial_guess(h_data, window_size, h_theta);
			for (int i = 1; i < (steps); ++i)
			{
				for (int j = 0; j < k * 3; ++j)
				{
					h_theta[i * k * 3 + j] = h_theta[j];
				}
			}
			break;
		}
	case 1:
		{
			for (int i = 0; i < steps; ++i)
			{
				set_initial_guess(&h_data[i * window_step], window_size, &h_theta[i * 3 * k]);
			}
			break;
		}
	case 2:
		{
			set_initial_guess(h_data, window_size, h_theta);
			break;
		}
	}

	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMemcpy(d_theta, h_theta, theta_size, cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(d_data, h_data, data_length * sizeof(double), cudaMemcpyHostToDevice));
	/*for (int i = 0; i < k * 3 * steps; ++i)
	{
		h_theta[i] = 0;
	}*/
	/*
	 *
	 */

	for (int i = 0; i < steps; ++i)
	{
		time_t rawtime;
		time(&rawtime);
		struct tm* timeinfo = localtime(&rawtime);

		char* foo = asctime(timeinfo);
		foo[strlen(foo) - 1] = 0;
		printf("[%s] Start EM %d: \n", foo , i);
		data_off = i * window_step;
		theta_offset = i * k * 3;
		em_algorithm(d_data, data_off, window_size, d_theta, theta_offset, h_theta, false);
		if (generate_theta_each_step == 2 && i!=steps-1)
		{
			copythetatonext<<<1,1>>>(d_theta, theta_offset, k * 3);
		}

	}
	cudaDeviceSynchronize();
	HANDLE_ERROR(cudaMemcpy(h_theta, d_theta, theta_size, cudaMemcpyDeviceToHost));
	/*
	 *
	 */
Error:

	cudaFree(d_data);
	cudaFree(d_theta);
	return cudaStatus;
}

void savetofile(double* theta, int size)
{
	ofstream ofile("output.data");
	if (ofile.is_open())
	{
		for (int i = 0; i < size; ++i)
		{
			for (int j = 0; j < k * 3; ++j)
			{
				ofile << theta[i * k * 3 + j] << ", ";
			}
			if (i != size - 1)
				ofile << endl;
		}
		ofile.close();
	}
	else
		cout << "Unable to open file";
}

int main()
{
	HANDLE_ERROR(cudaDeviceReset());
	srand(time(NULL));
	const int data_length = DATA_LENGTH;
	const char* data_filename = "..//data//data_imoex_180323_180424_5min.txt";
	const int window_length = WINDOW_LENGTH;
	const int window_step = 1;
	const int generate_theta_each_step = 0;
	/* 0 - генерировать одно нач приближение и копировать его во все итерации
	 * 1 - генерировать нач прибл для всех итераций
	 * 2 - использовать предыдущий результат как начальное приближение
	 */
	const int steps = (data_length - window_length + 1) / window_step;

	double* data = readfromfile(data_filename, data_length);
	double* theta = (double*)malloc(steps * 3 * k * sizeof(double));

	cudaError_t cudaStatus = slsalgorithm(data, data_length, theta, window_length, window_step, generate_theta_each_step);
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	savetofile(theta, steps);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}
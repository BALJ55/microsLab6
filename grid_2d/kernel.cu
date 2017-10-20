#include "kernel.h"
#include <stdio.h>

// kernel definition
__global__ void kernel(unsigned char *d_input, unsigned char *d_output, int rows, int cols){
  
  // get correspondig coordinates from grid indexes
  int c = blockIdx.x*blockDim.x + threadIdx.x;
  int r = blockIdx.y*blockDim.y + threadIdx.y;
  const int i = r*cols + c;

  // check image bounds
  if( (r>=rows) || (c>=cols) ){
    return;
  }

  // perform operation
  d_output[i] = d_input[i];
}



// function called from main.cpp
// wrapper function
void wrapper_gpu(Mat input, Mat output){

  unsigned char *inputPtr = (unsigned char*) input.data;
  unsigned char *outputPtr = (unsigned char*) output.data;
  unsigned int cols = input.cols;
  unsigned int rows = input.rows;

  //block dimensions (threads)
  int Tx = 32;
  int Ty = 32;

  //grid size dimensions (blocks)
  int Bx = (Tx + rows -1)/Tx;
  int By = (Ty + cols -1)/Ty;

  // declare pointers to device memory
  unsigned char *d_in  = 0;
  unsigned char *d_out = 0;
 
  // allocate memory in device
  cudaMalloc(&d_in, cols*rows*sizeof(unsigned char));
  cudaMalloc(&d_out, cols*rows*sizeof(unsigned char));
 
  // copy input data from host to device	
  cudaMemcpy(d_in, inputPtr, cols*rows*sizeof(unsigned char), cudaMemcpyHostToDevice);

  //prepare kernel lauch dimensions
  const dim3 blockSize = dim3(Tx, Ty);
  const dim3 gridSize= dim3(Bx, By);

  // launch kernel in GPU
  kernel<<<gridSize, blockSize>>>(d_in, d_out, rows, cols);
 
  // copy output from device to host
  cudaMemcpy(outputPtr, d_out, rows*cols*sizeof(unsigned char), cudaMemcpyDeviceToHost);
 
  // free the memory allocated for device arrays
  cudaFree(d_in);
  cudaFree(d_out);
}

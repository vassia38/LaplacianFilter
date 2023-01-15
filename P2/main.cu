//
// Laplacian Filter using CUDA
//
#include "opencv2/imgproc/imgproc.hpp"
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>


#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <stdio.h>

#define BLOCK_SIZE      16
#define FILTER_WIDTH    3       
#define FILTER_HEIGHT   3       

using namespace cv;
using namespace std;
//extern "C" bool laplacianFilter_GPU_wrapper(const cv::Mat & input, cv::Mat & output);
//extern "C" bool laplacianFilter_CPU(const cv::Mat & input, cv::Mat & output);

// Run Laplacian Filter on GPU
__global__ void laplacianFilter(unsigned char* srcImage, unsigned char* dstImage, unsigned int width, unsigned int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x; //col
    int y = blockIdx.y * blockDim.y + threadIdx.y; //row

    //float kernel[3][3] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
    float kernel[3][3] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };

    // only threads inside image will write results
    if ((x >= FILTER_WIDTH / 2) && (x < (width - FILTER_WIDTH / 2)) && (y >= FILTER_HEIGHT / 2) && (y < (height - FILTER_HEIGHT / 2)))
    {
        // Sum of pixel values 
        float new_pixel_val = 0;


        int k_values[3] = { -1,0,1 };
        int l_values[3] = { -1,0,1 };

        int current_k_index = 0;

        while (current_k_index < FILTER_WIDTH) {
            for (int l_index = 0; l_index < FILTER_HEIGHT; l_index++) {
                new_pixel_val += kernel[current_k_index][l_index] * srcImage[(y + k_values[current_k_index]) * width + (x + l_values[l_index])];
            }
            current_k_index++;
        }
        dstImage[y * width + x] = new_pixel_val;

    }
}



// Program main
int main(void) {

    // name of image
    string image_name = "C:\\Users\\Razvan\\source\\repos\\data\\sample.jpeg";

    // input & output file names
    string input_file = image_name + ".jpeg";
    string output_file_cpu = image_name + "_cpu.jpeg";
    string output_file_gpu = image_name + "_gpu.jpeg";

    // Read input image 
    cv::Mat srcImage = cv::imread(input_file, IMREAD_UNCHANGED);
    if (srcImage.empty())
    {
        std::cout << "Image Not Found: " << input_file << std::endl;
        return -1;
    }
    cout << "\ninput image size: " << srcImage.cols << " " << srcImage.rows << " " << srcImage.channels() << "\n";

    // convert RGB to gray scale
    cv::cvtColor(srcImage, srcImage, COLOR_BGR2GRAY);

    // Declare the output image  
    cv::Mat dstImage(srcImage.size(), srcImage.type());


    // Use cuda event to catch time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Calculate number of input & output bytes in each block
    const int inputSize = srcImage.cols * srcImage.rows;
    const int outputSize = dstImage.cols * dstImage.rows;
    unsigned char* d_input, * d_output;

    // Allocate device memory
    cudaMalloc<unsigned char>(&d_input, inputSize);
    cudaMalloc<unsigned char>(&d_output, outputSize);

    // Copy data from OpenCV input image to device memory
    cudaMemcpy(d_input, srcImage.ptr(), inputSize, cudaMemcpyHostToDevice);

    // Specify block size
    const dim3 block(32, 32);

    // Calculate grid size to cover the whole image
    const dim3 grid((dstImage.cols + block.x - 1) / block.x, (dstImage.rows + block.y - 1) / block.y);
    //const dim3 grid(20,21);

    cout << "\nblock_size " << block.x << " " << block.y << "\n";
    cout << "\ngrid_size " << grid.x << " " << grid.y << "\n";


    // Start time
    cudaEventRecord(start);

    // Run BoxFilter kernel on CUDA 
    laplacianFilter <<<grid, block >>> (d_input, d_output, dstImage.cols, dstImage.rows);


    // Stop time
    cudaEventRecord(stop);

    //Copy data from device memory to output image
    cudaMemcpy(dstImage.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);

    //Free the device memory
    cudaFree(d_input);
    cudaFree(d_output);

    cudaEventSynchronize(stop);
    float milliseconds = 0;

    // Calculate elapsed time in milisecond  
    cudaEventElapsedTime(&milliseconds, start, stop);
    cout << "\nProcessing time for GPU (ms): " << milliseconds << "\n";

    // run laplacian filter on GPU  
   // laplacianFilter_GPU_wrapper(srcImage, dstImage);
    // normalization to 0-255
    dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
    dstImage *= 255;
    // Output image
    imwrite(output_file_gpu, dstImage);


    //// run laplacian filter on CPU  
    //laplacianFilter_CPU(srcImage, dstImage);
    //// normalization to 0-255
    //dstImage.convertTo(dstImage, CV_32F, 1.0 / 255, 0);
    //dstImage *= 255;
    //// Output image
    //imwrite(output_file_cpu, dstImage);

    return 0;
}

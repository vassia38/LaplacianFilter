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
#define KERNEL_WIDTH    3       
#define KERNEL_HEIGHT   3       


using namespace cv;
using namespace std;

__global__ void laplacianFilter(unsigned char* inputImage, unsigned  char* outputImage, unsigned char* gaussianInputImage ,unsigned char* gaussianOutputImage,unsigned int width, unsigned int height)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
   
    //float kernel[3][3] = { 0, -1, 0, -1, 4, -1, 0, -1, 0 };
    float kernel[3][3] = { -1, -1, -1, -1, 8, -1, -1, -1, -1 };

    if ((col >= KERNEL_WIDTH / 2) && (col < (width - KERNEL_WIDTH / 2)) && (row >= KERNEL_HEIGHT / 2) && (row < (height - KERNEL_HEIGHT / 2)))
    {
        float new_pixel_val = 0;
        float new_pixel_val_gaussian = 0;

        int k_values[3] = { -1,0,1 };
        int l_values[3] = { -1,0,1 };

        int current_k_index = 0;

        while (current_k_index < KERNEL_WIDTH) {
            for (int l_index = 0; l_index < KERNEL_HEIGHT; l_index++) {
                new_pixel_val += kernel[current_k_index][l_index] * inputImage[(row + k_values[current_k_index]) * width + (col + l_values[l_index])];
                new_pixel_val_gaussian += kernel[current_k_index][l_index] * gaussianInputImage[(row + k_values[current_k_index]) * width + (col + l_values[l_index])];
            }
            current_k_index++;
        }
        outputImage[row * width + col] = new_pixel_val;
        gaussianOutputImage[row * width + col] = new_pixel_val_gaussian;


    }
}



// Program main
int main() {

    string source_image_path= "input_data\\source_image.jpg";
    string filtered_image_path = "output_data\\filtered_image.jpg";
    string filtered_image_with_Gaussian_path = "output_data\\filtered_image_with_gaussian.jpg";


    cv::Mat inputImage = cv::imread(source_image_path, IMREAD_UNCHANGED);
    if (inputImage.empty())
    {
        std::cout << "Image Not Found: " << source_image_path << std::endl;
        return -1;
    }
    cv::cvtColor(inputImage, inputImage, COLOR_BGR2GRAY);


    cv::Mat gaussianImage(inputImage.size(), inputImage.type());
    cv::GaussianBlur(inputImage, gaussianImage, Size(7, 7), 1,1);
   
    cv::Mat outputImage(inputImage.size(), inputImage.type());
    cv::Mat outputImageWithGaussian(gaussianImage.size(), gaussianImage.type());

    
    // Time measurements
    cudaEvent_t startTime, endTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&endTime);

    const int inputSize = inputImage.cols * inputImage.rows;
    const int outputSize = outputImage.cols * outputImage.rows;
    unsigned char* d_input, * d_output, *gaussian_mat_input, *gaussian_mat_output;

    cudaMalloc<unsigned char>(&d_input, inputSize * sizeof(unsigned char));
    cudaMalloc<unsigned char>(&d_output, outputSize * sizeof(unsigned char));
    cudaMalloc<unsigned char>(&gaussian_mat_input, inputSize * sizeof(unsigned char));
    cudaMalloc<unsigned char>(&gaussian_mat_output, outputSize * sizeof(unsigned char));



    cudaMemcpy(d_input, inputImage.ptr(), inputSize, cudaMemcpyHostToDevice);
    cudaMemcpy(gaussian_mat_input, gaussianImage.ptr(), inputSize, cudaMemcpyHostToDevice);


    const dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    const dim3 grid((outputImage.cols + block.x - 1) / block.x, (outputImage.rows + block.y - 1) / block.y);
    
    cout << "\nblock_size " << block.x << " " << block.y << "\n";
    cout << "\ngrid_size " << grid.x << " " << grid.y << "\n";

    cudaEventRecord(startTime);

    laplacianFilter <<<grid, block >>> (d_input, d_output, gaussian_mat_input, gaussian_mat_output,outputImage.cols, outputImage.rows);

    cudaEventRecord(endTime);

    cudaMemcpy(outputImage.ptr(), d_output, outputSize, cudaMemcpyDeviceToHost);
    cudaMemcpy(outputImageWithGaussian.ptr(), gaussian_mat_output, outputSize, cudaMemcpyDeviceToHost);


    cudaFree(d_input);
    cudaFree(d_output);

    cudaFree(gaussian_mat_input);
    cudaFree(gaussian_mat_output);


    cudaEventSynchronize(endTime);
    float milliseconds = 0;
    
    cudaEventElapsedTime(&milliseconds, startTime,endTime);
    cout << "\nProcessing time for GPU (ms): " << milliseconds << "\n";

    outputImage.convertTo(outputImage, CV_32F, 1.0 / 255, 0);
    outputImage *= 255;

    gaussianImage.convertTo(gaussianImage, CV_32F, 1.0 / 255, 0);
    gaussianImage *= 255;
  
    imwrite(filtered_image_path, outputImage);
    imwrite(filtered_image_with_Gaussian_path, outputImageWithGaussian);


    return 0;
}

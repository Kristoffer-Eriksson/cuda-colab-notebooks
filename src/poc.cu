#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <vector>
#include <fstream>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

__global__ void render(unsigned char *rgb_data, size_t width, size_t height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int index = y * width + x;
        rgb_data[3 * i + 0] = 255;
        rgb_data[3 * i + 1] = 0;
        rgb_data[3 * i + 2] = 0;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "Usage: poc.exe <output_path.png>" << std::endl;
        return EXIT_FAILURE;
    }

    size_t width = 256;
    size_t height = 256;

    /** Generate image **/

    cudaError_t error;

    unsigned char *rgb_data;

    /* Allocate n floats on device */
    error = cudaMalloc((void **)&rgb_data, width * height * 3 * sizeof(unsigned char));

    if (error != cudaSuccess) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " error = " << error << std::endl;
        return EXIT_FAILURE;
    }

    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);

    render<<<gridDim, blockDim>>>(rgb_data, width, height);

    /** Transfer data from GPU to CPU **/

    unsigned char *host_rgb_data = (unsigned char *)malloc(width * height * 3 * sizeof(unsigned char));

    error = cudaMemcpy(host_rgb_data, rgb_data, width * height * 3 * sizeof(unsigned char), cudaMemcpyDeviceToHost);

    if (error != cudaSuccess) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " error = " << error << std::endl;
        return EXIT_FAILURE;
    }

    /** Save image as PNG **/

    stbi_write_png(argv[1], width, height, 3, host_rgb_data, width * 3);

    /** Cleanup **/

    cudaFree(rgb_data);
    free(host_rgb_data);

    return EXIT_SUCCESS;
}

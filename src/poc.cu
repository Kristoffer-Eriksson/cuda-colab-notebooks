
/*
 * This program uses the host CURAND API to generate 10000
 * pseudorandom floats.
 */
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <vector>
#include <fstream>
#include <iostream>

__global__ void dummy_rgb_data(unsigned char *rgb_data, size_t width, size_t height) {
    for(int i = 0; i < width * height; i++){
        rgb_data[3 * i + 0] = 255;
        rgb_data[3 * i + 1] = 0;
        rgb_data[3 * i + 2] = 0;
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        std::cout << "Usage: poc.exe <output_path.jpg>" << std::endl;
        return 1;
    }

    size_t width = 800;
    size_t height = 600;

    /** Generate image **/

    cudaError_t error;

    unsigned char * rgb_data;

    /* Allocate n floats on device */
    error = cudaMalloc((void **)&rgb_data, width * height * 3 * sizeof(unsigned char));

    if (error != cudaSuccess) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " error = " << error << std::endl;
    }

    dummy_rgb_data<<<1,1>>>(rgb_data, width, height);

    /** Encode as JPEG **/

    nvjpegStatus_t status;

    /* Create handle */
    nvjpegHandle_t nvjpeg_handle;
    status = nvjpegCreateSimple(&nvjpeg_handle);
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    /* Create stream */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* Create JPEG encoder state */
    nvjpegEncoderState_t nv_enc_state;
    status = nvjpegEncoderStateCreate(nvjpeg_handle, &nv_enc_state, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    /* Set JPEG parameters */
    nvjpegEncoderParams_t params;
    status = nvjpegEncoderParamsCreate(nvjpeg_handle, &params, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    status = nvjpegEncoderParamsSetQuality(params, 100, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    status = nvjpegEncoderParamsSetOptimizedHuffman(params, 0, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    status = nvjpegEncoderParamsSetSamplingFactors(params, NVJPEG_CSS_444, stream);
    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    /* Set image parameters */
    nvjpegImage_t source;
    source.channel[0] = rgb_data;
    source.pitch[0] = width * 3;

    /* Encode the image */
    status = nvjpegEncodeImage(nvjpeg_handle, nv_enc_state, params,
        &source, NVJPEG_INPUT_RGB, width, height, stream);

    if (status != NVJPEG_STATUS_SUCCESS) {
        std::cerr << "Error " << __FILE__ << ":" << __LINE__ << " status = " << status << std::endl;
    }

    cudaStreamSynchronize(stream);

    // get compressed stream size
    size_t length;
    status = nvjpegEncodeRetrieveBitstream(nvjpeg_handle, nv_enc_state, NULL, &length, stream);

    // get stream itself
    std::vector<unsigned char> jpeg(length);
    status = nvjpegEncodeRetrieveBitstream(nvjpeg_handle, nv_enc_state, jpeg.data(), &length, stream);

    // write stream to file
    cudaStreamSynchronize(stream);
    std::ofstream output_file(argv[1], std::ios::out | std::ios::binary);
    output_file.write((char*)jpeg.data(), length);
    output_file.close();

    /* Cleanup */
    cudaFree(rgb_data);

    nvjpegEncoderParamsDestroy(params);
    nvjpegEncoderStateDestroy(nv_enc_state);
    nvjpegDestroy(nvjpeg_handle);

    cudaStreamDestroy(stream);

    return EXIT_SUCCESS;
}

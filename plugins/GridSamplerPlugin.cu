#include "GridSamplerPlugin.h"
#include "cuda_fp16.h"
#include <chrono>
#include <thread>
#include <cudnn.h>

template<class T>
void GridSamplerPlugin::SpatialTfSamplerForward(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, cudnnDataType_t cudnnDataType) {
    ck(cudnnSetTensorNdDescriptorEx(xDesc, CUDNN_TENSOR_NCHW, cudnnDataType, inputDesc[0].dims.nbDims, inputDesc[0].dims.d));
    ck(cudnnSetTensorNdDescriptorEx(yDesc, CUDNN_TENSOR_NCHW, cudnnDataType, outputDesc[0].dims.nbDims, outputDesc[0].dims.d));
    ck(cudnnSetSpatialTransformerNdDescriptor(stDesc, CUDNN_SAMPLER_BILINEAR, cudnnDataType, inputDesc[0].dims.nbDims, inputDesc[0].dims.d));

    T alpha = 1.0f, beta = 0.0f;
    ck(cudnnSpatialTfSamplerForward(cudnnHandle, stDesc, &alpha, xDesc, inputs[0], inputs[1], &beta, yDesc, outputs[0]));
}

int GridSamplerPlugin::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
    const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream)
{
    ck(cudnnSetStream(cudnnHandle, stream));

    if (inputDesc[0].type == nvinfer1::DataType::kFLOAT) {
        SpatialTfSamplerForward<float>(inputDesc, outputDesc, inputs, outputs, CUDNN_DATA_FLOAT);
    } else if (inputDesc[1].type == nvinfer1::DataType::kHALF) {
        SpatialTfSamplerForward<half>(inputDesc, outputDesc, inputs, outputs, CUDNN_DATA_HALF);
    } else {
        std::cerr << "Unsupported data type: " << (int)inputDesc[1].type << std::endl;
    }

    return 0;
}

REGISTER_TENSORRT_PLUGIN(GridSamplerPluginCreator);


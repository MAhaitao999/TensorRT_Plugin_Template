#include "AddPluginDyn.h"
#include "cuda_fp16.h"
#include <chrono>
#include <thread>

template<typename T>
__global__ void AddValue(T *pDst, T *pSrc, int n, T valueToAdd) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= n) return;
    pDst[x + n] = pDst[x] = pSrc[x] + valueToAdd;
}

int AddPluginDyn::enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) {
    int n = 1;
    for (int i = 0; i < inputDesc->dims.nbDims; i++) {
        n *= inputDesc->dims.d[i];
    }
    if (m.dataType == nvinfer1::DataType::kFLOAT) {
        std::cout << "Running fp32 kernel" << std::endl;
        std::this_thread::sleep_for(20ms);
        AddValue<<<(n + 1023) / 1024, 1024>>>((float *)outputs[0], (float *)inputs[0], n, m.valueToAdd);
    } else if (m.dataType == nvinfer1::DataType::kHALF) {
        std::cout << "Running fp16 kernel" << std::endl;
        std::this_thread::sleep_for(10ms);
        AddValue<<<(n + 1023) / 1024, 1024>>>((__half *)outputs[0], (__half *)inputs[0], n, (__half)m.valueToAdd);
    } else {
        std::cout << "Running int8 kernel" << std::endl;
        std::this_thread::sleep_for(0ms);
        AddValue<<<(n + 1023) / 1024, 1024>>>((int8_t *)outputs[0], (int8_t *)inputs[0], n, (int8_t)(m.valueToAdd / m.scale));
    }
    return 0;
}

REGISTER_TENSORRT_PLUGIN(AddPluginDynCreator);


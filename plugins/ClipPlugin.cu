#include "ClipPlugin.h"
#include "cuda_fp16.h"
#include <chrono>
#include <thread>

template <typename T>
__device__ __forceinline__ const T& Min(const T& a, const T& b) {
    return (a > b) ? b : a;
}

template <typename T>
__device__ __forceinline__ const T& Max(const T& a, const T& b) {
    return (a > b) ? a : b;
}

template <typename T, unsigned nthdsPerCTA>
__launch_bounds__(nthdsPerCTA)
    __global__ void clipKernel(
        int n,
        const T clipMin,
        const T clipMax,
        const T* input,
        T* output) {
    for (int i = blockIdx.x * nthdsPerCTA + threadIdx.x; i < n; i += gridDim.x * nthdsPerCTA)
    {
        output[i] = Min<T>(Max<T>(input[i], clipMin), clipMax);
    }
}

//建立gpu网格，调用上面的kernel
int clipInference(
    cudaStream_t stream,
    int n,
    float clipMin,
    float clipMax,
    const void* input,
    void* output) {
    const int blockSize = 512;
    const int gridSize = (n + blockSize - 1) / blockSize;
    clipKernel<float, blockSize><<<gridSize, blockSize, 0, stream>>>(n, clipMin, clipMax,
                                                 static_cast<const float*>(input),
                                                 static_cast<float*>(output));
    return 0;
}

int ClipPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void*, cudaStream_t stream) {
    int status = -1;

    // Our plugin outputs only one tensor
    void* output = outputs[0];

    // Launch CUDA kernel wrapper and save its return value
    status = clipInference(stream, mInputVolume * batchSize, mClipMin, mClipMax, inputs[0], output);

    return status;
}

// Static class fields initialization, 下面两句务必加上.
nvinfer1::PluginFieldCollection ClipPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> ClipPluginCreator::mPluginAttributes;

REGISTER_TENSORRT_PLUGIN(ClipPluginCreator);


#include <NvInfer.h>
#include <iostream>
#include <cstring>
#include <assert.h>
#include <cudnn.h>

inline bool check(cudnnStatus_t e, int iLine, const char *szFile) {
    if (e != CUDNN_STATUS_SUCCESS) {
        std::cerr << "CUDA driver API error " << cudnnGetErrorString(e) << " at line " << iLine << " in file " << szFile << std::endl;;
        return false;
    }
    return true;
}

#define ck(call) check(call, __LINE__, __FILE__)

using namespace std;

class GridSamplerPlugin: public nvinfer1::IPluginV2DynamicExt {
public:
    GridSamplerPlugin() {
        ck(cudnnCreate(&cudnnHandle));

        ck(cudnnCreateTensorDescriptor(&xDesc));
        ck(cudnnCreateTensorDescriptor(&yDesc));
        ck(cudnnCreateSpatialTransformerDescriptor(&stDesc));
    }
    ~GridSamplerPlugin() {
        ck(cudnnDestroyTensorDescriptor(xDesc));
        ck(cudnnDestroyTensorDescriptor(yDesc));
        ck(cudnnDestroySpatialTransformerDescriptor(stDesc));

        ck(cudnnDestroy(cudnnHandle));
    }

    virtual size_t getSerializationSize() const override {
        return 0;
    }
    virtual void serialize(void *buffer) const override {}

    nvinfer1::IPluginV2DynamicExt* clone() const override {
        return new GridSamplerPlugin();
    }

    bool supportsFormatCombination(int pos, const nvinfer1::PluginTensorDesc* inOut, int nbInputs, int nbOutputs) override {
    	switch(pos) {
    	case 0:
    		printf("inOut[0].type = %d, format[0]=%d\n", (int)inOut[0].type, (int)inOut[0].format);
    		return (inOut[0].type == nvinfer1::DataType::kFLOAT || inOut[0].type == nvinfer1::DataType::kHALF) && inOut[0].format == nvinfer1::TensorFormat::kLINEAR;
    	case 1:
    	case 2:
    		printf("inOut[1].type = %d, format[1]=%d\n", (int)inOut[1].type, (int)inOut[1].format);
    		return inOut[pos].type == inOut[0].type && inOut[pos].format == inOut[0].format;
    	}
        printf("Error: unexpected in/out pos=%d\n", pos);
    	return false;
    }

    int getNbOutputs() const override {
        return 1;
    }
    nvinfer1::DimsExprs getOutputDimensions(int index, const nvinfer1::DimsExprs* pInputDim, int nInputDim, nvinfer1::IExprBuilder &exprBuilder) override {
        nvinfer1::DimsExprs ret(pInputDim[0]);
        ret.d[2] = pInputDim[1].d[1];
        ret.d[3] = pInputDim[1].d[2];
        return ret;
    }
    nvinfer1::DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const override {
    	return inputTypes[0];
    }

    virtual void configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInput, const nvinfer1::DynamicPluginTensorDesc* out, int nbOutput) override {
    	printf("configurePlugin type=%d\n", (int)out[0].desc.type);
    }

    size_t getWorkspaceSize(const nvinfer1::PluginTensorDesc *inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc *outputs, int32_t nbOutputs) const override {return 0;}
    int enqueue(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc, const void *const *inputs, void *const *outputs, void *workspace, cudaStream_t stream) override;

    int initialize() override {return 0;}
    void terminate() override {}
    void destroy() override { delete this; }
    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    const char* getPluginType() const override {return "grid_sampler";}
    const char* getPluginVersion() const override {return "1";}
    void attachToContext(cudnnContext* /*cudnn*/, cublasContext* /*cublas*/, nvinfer1::IGpuAllocator* /*allocator*/) {}
    void detachFromContext() {}

private:
    cudnnHandle_t cudnnHandle;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnSpatialTransformerDescriptor_t stDesc;

    template<class T> void SpatialTfSamplerForward(const nvinfer1::PluginTensorDesc *inputDesc, const nvinfer1::PluginTensorDesc *outputDesc,
        const void *const *inputs, void *const *outputs, cudnnDataType_t cudnnDataType);

    using nvinfer1::IPluginV2Ext::configurePlugin;
    using nvinfer1::IPluginV2::getOutputDimensions;
    using nvinfer1::IPluginV2::getWorkspaceSize;
    using nvinfer1::IPluginV2::enqueue;
};

class GridSamplerPluginCreator : public nvinfer1::IPluginCreator {
public:
    GridSamplerPluginCreator() {}

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        return new GridSamplerPlugin();
    }
    
    const char* getPluginName() const override {return "grid_sampler";}
    const char* getPluginVersion() const override {return "1";}

    void setPluginNamespace(const char* szNamespace) override {}
    const char* getPluginNamespace() const override {return "";}
    
    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        std::cout << __FUNCTION__ << std::endl;
        return &fc;
    }
    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        std::cout << __FUNCTION__ << std::endl;
        float valueToAdd = 0;
        for (int i = 0; i < fc->nbFields; i++) {
            if (!strcmp(fc->fields[i].name, "valueToAdd")) {
                valueToAdd = *(float *)fc->fields[i].data;
            }
        }
        return new GridSamplerPlugin();
    }

    static nvinfer1::PluginFieldCollection fc;
};

nvinfer1::PluginFieldCollection GridSamplerPluginCreator::fc = nvinfer1::PluginFieldCollection();


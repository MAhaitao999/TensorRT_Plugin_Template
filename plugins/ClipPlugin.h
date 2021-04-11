#include "NvInfer.h"
#include <iostream>
#include <cstring>
#include <assert.h>
#include <vector>
#include <string>

static const char* CLIP_PLUGIN_VERSION{"1"};
static const char* CLIP_PLUGIN_NAME{"CustomClipPlugin"};

// 帮助函数，用于序列化plugin
template<typename T>
void writeToBuffer(char*& buffer, const T& val) {
    *reinterpret_cast<T*>(buffer) = val;
    buffer += sizeof(T);
}

// 帮助函数，用于反序列化plugin
template<typename T>
T readFromBuffer(const char*& buffer) {
    T val = *reinterpret_cast<const T*>(buffer);
    buffer += sizeof(T);
    return val;
}

using namespace std;
using namespace nvinfer1;

class ClipPlugin: public nvinfer1::IPluginV2 {
public:
    ClipPlugin(const std::string name, float clipMin, float clipMax)
       : mLayerName(name), mClipMin(clipMin), mClipMax(clipMax) {
    }
    
    ClipPlugin(const std::string name, const void* data, size_t length) 
        : mLayerName(name) {
	
	// Deserialize in the same order as serialization
        const char *d = static_cast<const char *>(data);
        const char *a = d;

        mClipMin = readFromBuffer<float>(d);
        mClipMax = readFromBuffer<float>(d);

        assert(d == (a + length));
    }

    // 无参构造函数无意义, 所以这里删除默认构造函数.
    ClipPlugin() = delete;

    int getNbOutputs() const override {
        return 1;
    }

    nvinfer1::Dims getOutputDimensions(int index, const nvinfer1::Dims* inputs, int nbInputDims) override {
        // Validate input arguments
        assert(nbInputDims == 1);
        assert(index == 0);

        // Clipping doesn't change input dimension, so output Dims will be the same as input Dims
        return *inputs;
    }

    int initialize() override {
        return 0;
    }

    void terminate() override {}

    size_t getWorkspaceSize(int) const override { return 0; };

    int enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream) override;

    size_t getSerializationSize() const override {
        return 2 * sizeof(float);
    }

    void serialize(void* buffer) const {
        char *d = static_cast<char *>(buffer);
        const char *a = d;

        writeToBuffer(d, mClipMin);
        writeToBuffer(d, mClipMax);

        assert(d == a + getSerializationSize());
    }

    void configureWithFormat(const nvinfer1::Dims* inputs, 
		             int nbInputs, 
			     const nvinfer1::Dims* outputs, 
			     int nbOutputs, 
			     nvinfer1::DataType type, 
			     nvinfer1::PluginFormat format, int maxBatchSize) override {
        // Validate input arguments
        assert(nbOutputs == 1);
        assert(type == DataType::kFLOAT);
        assert(format == PluginFormat::kNCHW);

        // Fetch volume for future enqueue() operations
        size_t volume = 1;
        for (int i = 0; i < inputs->nbDims; i++) {
            volume *= inputs->d[i];
        }
        mInputVolume = volume;
    }

    bool supportsFormat(nvinfer1::DataType type, nvinfer1::PluginFormat format) const override {
        // This plugin only supports ordinary floats, and NCHW input format
        if (type == nvinfer1::DataType::kFLOAT && format == nvinfer1::PluginFormat::kNCHW)
            return true;
        else
            return false;
    }

    const char* getPluginType() const override {
        return CLIP_PLUGIN_NAME;
    }

    const char* getPluginVersion() const override {
        return CLIP_PLUGIN_VERSION;
    }

    void destroy() override {
        // This gets called when the network containing plugin is destroyed
        delete this;
    }

    nvinfer1::IPluginV2* clone() const override {
        return new ClipPlugin(mLayerName, mClipMin, mClipMax);
    }

    void setPluginNamespace(const char* pluginNamespace) override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override {
        return mNamespace.c_str();
    }


private:
    const std::string mLayerName;
    float mClipMin, mClipMax;
    size_t mInputVolume;
    std::string mNamespace;
};

// 定义ClipPluginCreator类
class ClipPluginCreator : public nvinfer1::IPluginCreator {
public:
    ClipPluginCreator() {
        // Describe ClipPlugin's required PluginField arguments
        mPluginAttributes.emplace_back(nvinfer1::PluginField("clipMin", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));
        mPluginAttributes.emplace_back(nvinfer1::PluginField("clipMax", nullptr, nvinfer1::PluginFieldType::kFLOAT32, 1));

        // Fill PluginFieldCollection with PluginField arguments metadata
        mFC.nbFields = mPluginAttributes.size();
        mFC.fields = mPluginAttributes.data();
    }

    const char* getPluginName() const override {
        return CLIP_PLUGIN_NAME;
    }

    const char* getPluginVersion() const override {
        return CLIP_PLUGIN_VERSION;
    }

    const nvinfer1::PluginFieldCollection* getFieldNames() override {
        return &mFC;
    }

    nvinfer1::IPluginV2* createPlugin(const char* name, const nvinfer1::PluginFieldCollection* fc) override {
        float clipMin, clipMax;
	const nvinfer1::PluginField* fields = fc->fields;

	// Parse fields from PluginFieldCollection
	assert(fc->nbFields == 2);
        for (int i = 0; i < fc->nbFields; i++){
            if (strcmp(fields[i].name, "clipMin") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                clipMin = *(static_cast<const float*>(fields[i].data));
            } 
	    else if (strcmp(fields[i].name, "clipMax") == 0) {
                assert(fields[i].type == nvinfer1::PluginFieldType::kFLOAT32);
                clipMax = *(static_cast<const float*>(fields[i].data));
            }
        }
        return new ClipPlugin(name, clipMin, clipMax);
    }

    nvinfer1::IPluginV2* deserializePlugin(const char* name, const void* serialData, size_t serialLength) override {
        // This object will be deleted when the network is destroyed, which will
        // call ClipPlugin::destroy()
        return new ClipPlugin(name, serialData, serialLength);
    }
    
    void setPluginNamespace(const char* pluginNamespace) override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const override {
        return mNamespace.c_str();
    }

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<PluginField> mPluginAttributes;
    std::string mNamespace;
};


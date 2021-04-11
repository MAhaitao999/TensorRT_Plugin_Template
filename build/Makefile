SHELL=/bin/bash
CUDA_PATH = /usr/local/cuda
TRT_PATH = /usr/local/tensorrt7.2-cuda11.1
GENCODE = -gencode arch=compute_60,code=sm_60 -gencode arch=compute_61,code=sm_61 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_75,code=sm_75 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86

GCC = g++
NVCC = $(CUDA_PATH)/bin/nvcc
CCFLAGS = -g -DNDEBUG -std=c++17
INCLUDES := -I../include -isystem $(CUDA_PATH)/include -isystem $(TRT_PATH)/include
LDFLAGS := -L$(CUDA_PATH)/lib64 -L$(TRT_PATH)/lib -Wl,-rpath=$(CUDA_PATH)/lib64
LDFLAGS += -lnvinfer -lnvonnxparser -lcudart -ldl -lpthread -lcuda

OBJ_DIR = obj
OBJ = $(shell find $(OBJ_DIR) -name *.o 2>/dev/null)
DEP = $(OBJ:.o=.d)

SO = AddPlugin.so AddPluginDyn.so GridSamplerPlugin.so ClipPlugin.so

all: $(SO)

AddPlugin.so: $(addprefix $(OBJ_DIR)/, AddPlugin.o)
AddPluginDyn.so: $(addprefix $(OBJ_DIR)/, AddPluginDyn.o)
GridSamplerPlugin.so: $(addprefix $(OBJ_DIR)/, GridSamplerPlugin.o)
ClipPlugin.so: $(addprefix $(OBJ_DIR)/, ClipPlugin.o)


-include $(DEP)

clean:
	rm -rf $(SO) $(DEP) $(OBJ_DIR)

VPATH = ../include:../plugins

$(OBJ_DIR)/%.o: %.cu
	@mkdir -p $(@D)
	$(NVCC) $(CCFLAGS) -M -MT $@ $(INCLUDES) -o $(@:.o=.d) $<
	$(NVCC) $(CCFLAGS) $(INCLUDES) -Xcompiler -fPIC $(GENCODE) -o $@ -c $<

$(SO):
	$(GCC) $(CCFLAGS) -shared -o $@ $+ $(LDFLAGS)


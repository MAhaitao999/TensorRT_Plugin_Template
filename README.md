# TensorRT_Plugin_Template
TensorRT实现Plugin的一些模板化流程

编译插件成动态库:

```
cd build
make
```

会在build/目录下生成个各个插件的.so文件.

用trtexec转换带插件的模型文件, 命令如下:

```bash_script
trtexec --uff=trained_lenet5.uff --uffInput=InputLayer,1,28,28 --output=MarkOutput_0 --plugins=../TensorRT_Plugin_Template/build/ClipPlugin.so --saveEngine=trained_lenet5.engine
```

用trtexec测试推理(仍需指定插件):

```bash_script
trtexec --loadEngine=trained_lenet5.engine --shapes=InputLayer:1x28x28 --plugins=../TensorRT_Plugin_Template/build/ClipPlugin.so
```

# Tensorflow Lite for Microcontrollers MNIST Example

Example MNIST ConvNet using Tensorflow Lite for Microcontrollers

###  Getting the sources

This repository uses submodules. You need the --recursive option to fetch the submodules automatically

    $ git clone --recursive https://github.com/darinkuo/mnist-tflm
    
Alternatively :

    $ git clone https://github.com/darinkuo/mnist-tflm
    $ cd mnist-tflm
    $ git submodule update --init --recursive

### Prerequisites
These packages are required to build the MNIST model
* [Tensorflow 2.1.0](https://www.tensorflow.org/install)


### Usage

Train the MNIST classifier and generate model files in the model/ folder:
* model_data.cc
* model_data.h
* mnist-model.h5
* mnist_quant.tflite

```
python train.py
```

If the model has been modified anyway, replace the model_data files in the src directory with ones generated above

Modify data location string with workspace path in src/main_function.cc
```
std::string mnist_data_location = "<workspace here>/mnist-tflm/third_party/mnist_reader";
```

Build the inference model
```
make
```

Run inference
```
./mnist_inference.out
```

### Expected Output
```
Retreiving the MNIST dataset

Details of input tensor:
Dims [4] Size :
Type [FLOAT32] Shape :
0 [ 1]
1 [ 28]
2 [ 28]
3 [ 1]

Details of output tensor:
Dims [2] Size :
Type [FLOAT32] Shape :
0 [ 1]
1 [ 10]

Model estimates [7] label [7]

Model estimates [2] label [2]

Model estimates [1] label [1]
```

### Based Off Of
* [MNIST classifcation Example](https://github.com/PeteBlackerThe3rd/tensorflow/tree/master/tensorflow/lite/micro/examples/micro_mnist)
* [Hello World Example](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world)

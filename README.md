# Tensorflow Lite for Microcontrollers MNIST Example

Example MNIST ConvNet using Tensorflow Lite for Microcontrollers

### Prerequisites
* [Tensorflow 2.1.0](https://www.tensorflow.org/install/pip)

### Installing

To build the MNIST ConvNet model alongside tflite and C source files
```
python train.py
```

Once the model files have been moved to the make/src folder. To make the project:

```
cd make
make
```

Run inference
```
./mnist_inference
```

Expected Output
```
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
* [MNIST classifcation Example] (https://github.com/PeteBlackerThe3rd/tensorflow/tree/master/tensorflow/lite/micro/examples/micro_mnist)
* [Hello World Example] (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/micro/examples/hello_world)
* [Simple C++ Reader for MNIST dataset] (https://github.com/wichtounet/mnist)
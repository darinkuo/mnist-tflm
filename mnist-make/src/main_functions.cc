/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "src/main_functions.h"

#include "src/output_handler.h"
#include "src/model_settings.h"
#include "src/mnist_reader.hpp"
#include "src/model_data.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

// Globals, used for compatibility with Arduino-style sketches.
namespace {
tflite::ErrorReporter* error_reporter = nullptr;
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;
int inference_count = 0;
std::string mnist_data_location = "/home/kuodm/research/tests/mnist/mnist-make"; 
mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset;
// Create an area of memory to use for input, output, and intermediate arrays.
// Finding the minimum value for your model may require some trial and error.
constexpr int kTensorArenaSize = 90 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

// Helper fn to log the shape and datatype of a tensor
void printTensorDetails(TfLiteTensor* tensor,
                        tflite::ErrorReporter* error_reporter) {
  error_reporter->Report("Dims [%d] Size :", (tensor->dims->size));
  error_reporter->Report("Type [%s] Shape :", TfLiteTypeGetName(tensor->type));
  for (int d = 0; d < tensor->dims->size; ++d) {
    error_reporter->Report("%d [ %d]", d, tensor->dims->data[d]);
  }
  error_reporter->Report("");
}

// The name of this function is important for Arduino compatibility.
void setup() {
  // Set up logging. Google style is to avoid globals or statics because of
  // lifetime uncertainty, but since this has a trivial destructor it's okay.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Retreive the MNIST dataset
  dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>(mnist_data_location);

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  model = tflite::GetModel(mnist_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    TF_LITE_REPORT_ERROR(error_reporter,
                         "Model provided is schema version %d not equal "
                         "to supported version %d.",
                         model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  // This pulls in all the operation implementations we need.
  // NOLINTNEXTLINE(runtime-global-variables)
  static tflite::ops::micro::AllOpsResolver resolver;

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  // The ops listed below are an example, not the ones for this model.
  // static tflite::MicroOpResolver<3> micro_op_resolver;
  // micro_op_resolver.AddBuiltin(
  //    tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
  //    tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
  //                              tflite::ops::micro::Register_CONV_2D());
  // micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_AVERAGE_POOL_2D,
  //                              tflite::ops::micro::Register_AVERAGE_POOL_2D());

  // Build an interpreter to run the model with.
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize, error_reporter);
  interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors.
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(error_reporter, "AllocateTensors() failed");
    return;
  }
  // Get information about the models input and output tensors.
  input = interpreter->input(0);
  error_reporter->Report("Details of input tensor:");
  printTensorDetails(input, error_reporter);
  output = interpreter->output(0);
  error_reporter->Report("Details of output tensor:");
  printTensorDetails(output, error_reporter);
  return;
}

// The name of this function is important for Arduino compatibility.
void loop() {
  // Get image from provider.
  for(int i = 0; i < numInferences; ++i){
    for(int j = 0; j < kMaxImageSize; ++j){
      input->data.f[j] = dataset.test_images[i][j];
    }

    // Run the model on this input and make sure it succeeds.
    if (kTfLiteOk != interpreter->Invoke()) {
      TF_LITE_REPORT_ERROR(error_reporter, "Invoke failed.");
    }

    // Iterate over the predicted values and print the largest probability
    float Yhat_prob = 0;
    int Yhat = 0;
    for(int j = 0; j < kCategoryCount; j++) {
      float tmp = output->data.f[j];
      if(tmp > Yhat_prob) {
        Yhat = j;
      }
    }
    // Output the results. A custom HandleOutput function can be implemented
    // for each supported hardware target.
    HandleOutput(error_reporter, Yhat, dataset.test_labels[i]);
  }
}

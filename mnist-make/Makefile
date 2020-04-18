CXX = g++
CC  = gcc
RM  = rm -f

# Required Tensorflow Lite files to compile the binary
TF_SRCS := \
tensorflow/lite/micro/simple_memory_allocator.cc tensorflow/lite/micro/memory_helpers.cc tensorflow/lite/micro/test_helpers.cc tensorflow/lite/micro/micro_error_reporter.cc tensorflow/lite/micro/micro_time.cc tensorflow/lite/micro/debug_log.cc tensorflow/lite/micro/micro_allocator.cc tensorflow/lite/micro/micro_string.cc tensorflow/lite/micro/micro_utils.cc tensorflow/lite/micro/micro_optional_debug_tools.cc tensorflow/lite/micro/micro_interpreter.cc tensorflow/lite/micro/kernels/comparisons.cc tensorflow/lite/micro/kernels/reshape.cc tensorflow/lite/micro/kernels/depthwise_conv.cc tensorflow/lite/micro/kernels/conv.cc tensorflow/lite/micro/kernels/mul.cc tensorflow/lite/micro/kernels/prelu.cc tensorflow/lite/micro/kernels/dequantize.cc tensorflow/lite/micro/kernels/pooling.cc tensorflow/lite/micro/kernels/activations.cc tensorflow/lite/micro/kernels/logistic.cc tensorflow/lite/micro/kernels/ceil.cc tensorflow/lite/micro/kernels/arg_min_max.cc tensorflow/lite/micro/kernels/reduce.cc tensorflow/lite/micro/kernels/split.cc tensorflow/lite/micro/kernels/add.cc tensorflow/lite/micro/kernels/softmax.cc tensorflow/lite/micro/kernels/pad.cc tensorflow/lite/micro/kernels/floor.cc tensorflow/lite/micro/kernels/circular_buffer.cc tensorflow/lite/micro/kernels/svdf.cc tensorflow/lite/micro/kernels/sub.cc tensorflow/lite/micro/kernels/concatenation.cc tensorflow/lite/micro/kernels/unpack.cc tensorflow/lite/micro/kernels/neg.cc tensorflow/lite/micro/kernels/quantize.cc tensorflow/lite/micro/kernels/all_ops_resolver.cc tensorflow/lite/micro/kernels/fully_connected.cc tensorflow/lite/micro/kernels/maximum_minimum.cc tensorflow/lite/micro/kernels/elementwise.cc tensorflow/lite/micro/kernels/strided_slice.cc tensorflow/lite/micro/kernels/round.cc tensorflow/lite/micro/kernels/pack.cc tensorflow/lite/micro/kernels/logical.cc tensorflow/lite/micro/memory_planner/linear_memory_planner.cc tensorflow/lite/micro/memory_planner/greedy_memory_planner.cc tensorflow/lite/c/common.c tensorflow/lite/core/api/error_reporter.cc tensorflow/lite/core/api/flatbuffer_conversions.cc tensorflow/lite/core/api/op_resolver.cc tensorflow/lite/core/api/tensor_utils.cc tensorflow/lite/kernels/internal/quantization_util.cc tensorflow/lite/kernels/kernel_util.cc tensorflow/lite/micro/testing/test_utils.cc  

# Project source files appended to required TF files
SRCS := \
$(TF_SRCS) src/main.cc src/main_functions.cc src/model_data.cc src/output_handler.cc

# Object files
OBJS := \
$(patsubst %.cc,%.o,$(patsubst %.c,%.o,$(SRCS)))

CXXFLAGS += -g --static -std=c++11 -DTF_LITE_STATIC_MEMORY -O3 -I. -I./third_party/gemmlowp -I./third_party/flatbuffers/include -I./third_party/kissfft
CCFLAGS +=  -g --static -std=c11   -DTF_LITE_STATIC_MEMORY -O3 -I. -I./third_party/gemmlowp -I./third_party/flatbuffers/include -I./third_party/kissfft

LDFLAGS +=  -lm

%.o: %.cc
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

%.o: %.c
	$(CC) $(CCFLAGS) $(INCLUDES) -c $< -o $@

hello_world : $(OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $(OBJS) $(LDFLAGS)

all: hello_world

clean:
	-$(RM) $(OBJS)
	-$(RM) hello_world

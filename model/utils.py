import os
from pathlib import Path

import numpy as np
import tensorflow as tf


def rep_data_gen(x_test):
    a = []
    for i in range(x_test.shape[0]):
        a.append(x_test[i])
    a = np.array(a)
    img = tf.data.Dataset.from_tensor_slices(a).batch(1)
    for i in img.take(1):
        yield [i]


"""A single function used to export a TensorFlow lite flatbuffer as a header
and source file pair compatable with TensorFlow lite micro projects.

Author: PeteBlackerThe3rd
Source: https://github.com/PeteBlackerThe3rd/tensorflow/blob/master/tensorflow/lite/micro/examples/micro_mnist/model/flatbuffer_2_tfl_micro.py
"""


def write_tf_lite_micro_model(flatbuffer,
                              base_file_name="model_data",
                              data_variable_name="",
                              header_comment=None, directory=Path("make/src")):
    """
    Method to generate a .h and .cc file containing a string literal and length
    definition of the given tflite flatbuffer. Used to automatically generate
    the model data sourcen files for a TF lite micro project, where the weights
    and model are compiled into the binary itself.
    :param flatbuffer: a byte array of the .tflite flatbuffer containing the
            saved model
    :param base_file_name: the base filename of the generated source files
    :param data_variable_name: optional indentifier of C++ literal to create,
            an indentifier based on the filename will be generated if this is
            omitted.
    :param header_comment: optional header comment which will be added at the
            top of each source file if given
    :return: True on success False on failiure
    """
    (_, base_name_wo_path) = os.path.split(base_file_name)
    if data_variable_name == "":
        data_variable_name = base_name_wo_path + "_tflite"

    if header_comment is not None:
        header_lines = header_comment.split("\n")
        header_comment = "// " + "\n// ".join(header_lines)

    with open(directory / f"{base_file_name}.h", "w") as header:
        header.write("#ifndef __%s_H__\n" % base_name_wo_path.upper())
        header.write("#define __%s_H__\n" % base_name_wo_path.upper())
        if header_comment is not None:
            header.write(header_comment + "\n\n")
        header.write("// Model tflite flatbuffer.\n")
        header.write("extern const unsigned char %s[];\n\n" %
                     data_variable_name)

        header.write("// Length of model tflite flatbuffer.\n")
        header.write("extern const unsigned int %s_len;\n\n" %
                     data_variable_name)
        header.write("#endif  // __%s_H__\n" % base_name_wo_path.upper())
        header.close()

    with open(directory / f"{base_file_name}.cc", "w") as source:
        if header_comment is not None:
            source.write(header_comment + "\n\n")
        source.write("#include \"%s.h\"\n" % base_name_wo_path)
        source.write("\n// Model data tflite flatbuffer.\n")
        source.write("const unsigned char %s[] = {\n" % data_variable_name)

        flatbuffer_pos = 0
        while flatbuffer_pos < len(flatbuffer):
            chunk = flatbuffer[flatbuffer_pos:flatbuffer_pos + 12]
            flatbuffer_pos += 12
            source.write("    " +
                         ', '.join("0x" + ('{:02X}'.format(x)) for x in chunk))
            if flatbuffer_pos < len(flatbuffer):
                source.write(",")
            source.write("\n")
        source.write("\n")
        source.write("};\n")

        source.write("// Length of model tflite flatbuffer.\n")
        source.write("const unsigned int %s_len = %d;\n" %
                     (data_variable_name,
                      len(flatbuffer)))
        source.close()

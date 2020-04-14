import argparse
import os

import tensorflow as tf
import flatbuffer_2_tfl_micro as save_tflm

def convert_to_flatbuffer(path, convert_name, quantized=False):
	# Convert the Model
	converter = tf.lite.TFLiteConverter.from_saved_model(path)
	if (quantized):
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
	tflite_model = converter.convert()

	# Save the TFLite Model
	with open(convert_name + '.tflite', "wb") as f:
		f.write(tflite_model)

	# Generate the header and source files
	save_tflm.write_tf_lite_micro_model(tflite_model, 
										data_variable_name=convert_name + "_data")
	
	return

def main(argv):
	
	# Determines name for converted model
	tflite_model_name = 'mnist_model'
	if(argv.o != None):
		if(args.o.endswith('.tflite')):
			tflite_model_name = argv.o[:-len('.tflite')]
		else:
			tflite_model_name = argv.o
	
	# Check that path exists and is to a directory
	if(not os.path.exists(argv.model_path)):
		print(argv.model_path + ": No such directory")
		return
	elif(os.path.isfile(argv.model_path)):
		print(argv.model_path + ": is a file, not a directory")
		return
	
	convert_to_flatbuffer(argv.model_path, tflite_model_name, argv.q)

	return

if __name__ == '__main__':
	# Argument Parsing
	parser = argparse.ArgumentParser(description="A script to convert" + 
									"Tensorflow SavedModels into TFLite format")
	parser.add_argument("model_path", metavar='P', type=str, 
						help="The relative path to SavedModel directory")
	parser.add_argument('-o', type= str, help="Name for TFLite model."
						+ "(default: converted_model.tflite)")
	parser.add_argument('-q', help="Generate Quantized model with respect to"
						+ "size", action="store_true")
	args = parser.parse_args()
	main(args)

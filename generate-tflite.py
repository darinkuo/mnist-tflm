from os import path, system
import argparse
import tensorflow as tf

def convert(path, convert_name, quantized=False):
	# Convert the Model
	converter = tf.lite.TFLiteConverter.from_saved_model(path)
	tflite_model = converter.convert()

	# Save the nonQuantized TFLite Model
	open(convert_name + '.tflite', "wb").write(tflite_model)

	if (quantized):
		converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
		tflite_model = converter.convert()
		# Save the model to disk
		open(convert_name + "_quantized.tflite", "wb").write(tflite_model)
	
	return 

def main():
	# Argument Parsing
	parser = argparse.ArgumentParser(description="A script to convert "
							+ "Tensorflow SavedModels into TFLite format")
	parser.add_argument("model_path", metavar='P', type=str, 
							help="The relative path to SavedModel directory")
	parser.add_argument('-o', type= str, help="Name for TFLite model."
							+ "(default: converted_model.tflite)")
	parser.add_argument('-q', help="Generate Quantized model with respect to size",
							action="store_true")
	parser.add_argument('-c', help="Generate C++ source file for Model called"
							+ " model_data.cc",
							action="store_true")
	args = parser.parse_args()

	# Determines name for converted model
	tflite_model_name = 'converted_model.tflite'
	if(args.o != None):
		if(args.o.endswith('.tflite')):
			tflite_model_name = args.o[:-len('.tflite')]
		else:
			tflite_model_name = args.o
	
	# Check that path exists and is to a directory
	if(not path.exists(args.model_path)):
		print(args.model_path + ": No such directory")
		return
	elif(path.isfile(args.model_path)):
		print(args.model_path + ": is a file, not a directory")
		return

	# Convert to TFlite Model
	print("=== Begining conversion ===")
	convert(args.model_path, tflite_model_name, args.q)
	print("=== Conversion complete ===")

	if(args.c == True):
		print("=== Generating C++ source file ===")
		system('xxd -i ' + tflite_model_name + ' > model_data.cc')
		print("=== model_data.cc generated ===")

	return

if __name__ == '__main__':
	main()

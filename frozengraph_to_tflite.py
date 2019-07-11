import tensorflow as tf

graph_def_file = "/home/bende/programmes_test/ssd_keras_files/tflite/tflite_graph.pb"
input_arrays = ["normalized_input_image_tensor"]
output_arrays = ['TFLite_Detection_PostProcess','TFLite_Detection_PostProcess:1','TFLite_Detection_PostProcess:2','TFLite_Detection_PostProcess:3']
output_file = "/home/bende/programmes_test/ssd_keras_files/tflite/converted_model.tflite"

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays, input_shapes={"normalized_input_image_tensor": [1, 300, 300, 3]})
"""
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
"""
converter.allow_custom_ops = True
tflite_model = converter.convert()
open(output_file, "wb").write(tflite_model)

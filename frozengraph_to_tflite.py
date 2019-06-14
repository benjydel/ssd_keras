import tensorflow as tf

graph_def_file = "../ssd_keras_files/3-ssd300_OID_plates_1000step_per_epoch_reduceOnPlateau.pb"
input_arrays = ["input_1"]
output_arrays = ["predictions/concat"]

converter = tf.lite.TFLiteConverter.from_frozen_graph(graph_def_file, input_arrays, output_arrays)
converter.inference_type = tf.lite.constants.QUANTIZED_UINT8
input_arrays = converter.get_input_arrays()
converter.quantized_input_stats = {input_arrays[0] : (0., 1.)}  # mean, std_dev
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

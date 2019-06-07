import os
import argparse

# exemple :
# python3 keras_SSD300_to_frozengraph_tf.py --model_path "../ssd_keras_files/ssd300_OID_plates_2105_MODEL.h5" --save_dir "../ssd_keras_files/" --save_name "ssd300_OID_plates_frozen_model.pb"
parser = argparse.ArgumentParser(description='Convert TensorFlow frozen graph .pb file to optimized OpenVINO .xml and .bin files')
parser.add_argument("--input_model", help="Path to the input file to be converted", required=True)
parser.add_argument("--output_dir", help="Path to the output dir where to save the .xml and .bin files (same name as input file)", required=True)

args = parser.parse_args()

input_model = args.input_model
output_dir = args.output_dir

# OpenVINO 2019
# mo_tf.py path in Linux
mo_tf_path = '/opt/intel/openvino/deployment_tools/model_optimizer/mo_tf.py'

img_height = 300
img_width = 300

input_shape = [1,img_width,img_height,3]
input_shape_str = str(input_shape).replace(' ','')

os.system('python3 '+mo_tf_path+' --log_level DEBUG --input_model '+input_model+' --output_dir '+output_dir+' --input_shape '+input_shape_str+' --data_type FP16 > ../ssd_keras_files/log_openvino.log')
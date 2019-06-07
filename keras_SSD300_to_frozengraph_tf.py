import tensorflow as tf
from tensorflow.python.framework import graph_io
from keras import backend as K
from keras.models import load_model
from keras_loss_function.keras_ssd_loss import SSDLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_DecodeDetections import DecodeDetections
from keras_layers.keras_layer_DecodeDetectionsFast import DecodeDetectionsFast
from keras_layers.keras_layer_L2Normalization import L2Normalization

import argparse

# exemple :
# python3 keras_SSD300_to_frozengraph_tf.py --model_path "../ssd_keras_files/ssd300_OID_plates_2105_MODEL.h5" --save_dir "../ssd_keras_files/" --save_name "ssd300_OID_plates_frozen_model.pb"
parser = argparse.ArgumentParser(description='Convert keras weight model .h5 file to frozen graph .pb for TensorFlow')
parser.add_argument("--model_path", help="Path to the input file to be converted", required=True)
parser.add_argument("--save_dir", help="Path to the output dir where to save the frozen graph", required=True)
parser.add_argument("--save_name", help="The output name for the frozen graph", required=True)

args = parser.parse_args()

model_fname = args.model_path
save_pb_dir = args.save_dir
save_pb_name=args.save_name


# Clear any previous session.
K.clear_session()


def freeze_graph(graph, session, output_names, keep_var_names=None, clear_devices=True):
    with graph.as_default():
        """
        #test
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        #fintest
        """
        input_graph_def = graph.as_graph_def()
        """
        #test
        if clear_devices:
                for node in input_graph_def.node:
                        node.device = ""
        #fintest
        """
        graphdef_inf = tf.graph_util.remove_training_nodes(input_graph_def)

        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output_names)
        return graphdef_frozen

# This line must be executed before loading Keras model.
K.set_learning_phase(0) 

# We need to create an SSDLoss object in order to pass that to the model loader.
ssd_loss = SSDLoss(neg_pos_ratio=3, n_neg_min=0, alpha=1.0)

model = load_model(model_fname, custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'L2Normalization': L2Normalization,
                                               'DecodeDetections': DecodeDetections,
                                               'compute_loss': ssd_loss.compute_loss})

session = K.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print("\n\nKeep Input and Output layer names !!!\nThey are needed by tensorflow to run the inference")
print("\tInput NODE name : ",INPUT_NODE,"\n\tOutput NODE name : ", OUTPUT_NODE,"\n\n")
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs])

graph_io.write_graph(frozen_graph, save_pb_dir, save_pb_name, as_text=False)

model.summary()

print(model.layers[-1].name," : ",model.layers[-1].output_shape)
[print(t.op.name) for t in model.outputs]
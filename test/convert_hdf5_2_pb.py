import os
import argparse

parser = argparse.ArgumentParser(description='Deploy keras model.')
parser.add_argument('--input', required=True, type=str, help="Input Keras model (hdf5)")
parser.add_argument('--output', required=False, type=str, default=None, help="Output Protocol Buffers file")
args = parser.parse_args()

import tensorflow as tf
from tensorflow.python.framework.graph_io import write_graph
from tensorflow.python.framework.graph_util import convert_variables_to_constants
#from tensorflow.tools.graph_transforms import TransformGraph
from keras import backend as K
#from common import LoadModel
from keras.models import load_model

K.set_learning_phase(0)
model = load_model(args.input)

print(model.inputs[0].name)
print(model.outputs[0].name)
#raise RuntimeError("stop")

print ("teste", [node.op.name for node in model.outputs])
input_nodes = [model.inputs[0].name] #"main_input"]#  [model.inputs[0].name]
output_nodes = [model.outputs[0].name] #"main_output/Softmax"]#["output_node"]
#node_wrapper = tf.identity(model.outputs[0], name=output_nodes[0])

with K.get_session() as sess:
    ops = sess.graph.get_operations()
    const_graph = convert_variables_to_constants(sess, sess.graph.as_graph_def(), [node.op.name for node in model.outputs])
    # final_graph = const_graph
    transforms = [
        "strip_unused_nodes",
        "remove_nodes(op=Identity, op=CheckNumerics)",
        "fold_constants(ignore_errors=true)",
        "fold_batch_norms"#,
        #"quantize_weights" # it was added due to cmssw restrictions on the network size
    ]
    final_graph = const_graph # TransformGraph(const_graph, input_nodes, output_nodes, transforms)

if args.output is None:
    input_base = os.path.basename(args.input)
    out_dir = '.'
    out_file = os.path.splitext(input_base)[0] + ".pb"
else:
    out_dir, out_file = os.path.split(args.output)
write_graph(final_graph, out_dir, out_file, as_text=False)


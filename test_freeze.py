
import tensorflow as tf
from tensorflow.python.platform import gfile
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

# y = tf.placeholder(tf.float32, [None, 10], name='LabelData')
# acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
# acc = tf.reduce_mean(tf.cast(acc, tf.float32))

frozen_graph_filename = './example/train_model.pb'

with gfile.FastGFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    byte = f.read()
    graph_def.ParseFromString(byte)

tf.import_graph_def(graph_def, name='')

for node in graph_def.node:
    print(node.name)


with tf.Session() as sess:
    detection_graph = tf.get_default_graph()
    input_tensor = detection_graph.get_tensor_by_name('input_tensor:0')
    output_tensor = detection_graph.get_tensor_by_name('output_tensor:0')
    acc = detection_graph.get_tensor_by_name('accuracy:0')
    y = detection_graph.get_tensor_by_name('labeled_data:0')
    # print(input_tensor.shape)
    print("Accuracy:", acc.eval({input_tensor: mnist.test.images, y: mnist.test.labels}))

    correct = 0
    i = 0

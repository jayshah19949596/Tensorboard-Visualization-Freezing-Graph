"""
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

"""


from __future__ import print_function
import os
import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector
from tensorflow.python.tools import freeze_graph
import keras
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data


def model(inputs):
    # Set model weights
    W = tf.Variable(tf.zeros([784, 10]), name='Weights')
    b = tf.Variable(tf.zeros([10]), name='Bias')

    tf.summary.histogram("weights", W)
    tf.summary.histogram("biases", b)

    # Construct model and encapsulating all ops into scopes, making
    # Tensorboard's Graph visualization more convenient

    output = tf.nn.softmax(tf.matmul(inputs, W) + b, name="output_tensor")  # Softmax
    return output


mnist = input_data.read_data_sets("./data/", one_hot=True)

# Parameters
learning_rate = 0.01
training_epochs = 10
batch_size = 50
display_epoch = 1
logs_path = './example/'
LOG_DIR = "/example"
cmd_path = "\"" + os.getcwd() + "/example\""

PATH_TO_SPRITE_IMAGE = os.path.join(logs_path, 'sprite_1024.png')
# def save_metadata(file):
if not os.path.exists(logs_path):
    os.mkdir(logs_path)

with open(os.path.join(logs_path, "s-metadata.tsv"), 'w') as metadata_file:
    for row in range(10000):
        c = np.nonzero(mnist.test.labels[::1])[1:][0][row]
        metadata_file.write('{}\n'.format(c))

metadata = 's-metadata.tsv'

# perparing the embedding
images = tf.Variable(mnist.test.images, name='images')

config = projector.ProjectorConfig()
# One can add multiple embeddings.
embedding = config.embeddings.add()
embedding.tensor_name = images.name
# Link this tensor to its metadata file (e.g. labels).
embedding.metadata_path = metadata
# Saves a config file that TensorBoard will read during startup.
projector.visualize_embeddings(tf.summary.FileWriter(logs_path), config)
embedding.sprite.image_path = PATH_TO_SPRITE_IMAGE
# Specify the width and height of a single thumbnail.
embedding.sprite.single_image_dim.extend([28, 28])

# tf Graph Input
# mnist data image of shape 28*28=784
x = tf.placeholder(tf.float32, [None, 784], name='input_tensor')
# 0-9 digits recognition => 10 classes
y = tf.placeholder(tf.float32, [None, 10], name='labeled_data')

pred = model(x)
# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# Accuracy
acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
acc = tf.reduce_mean(tf.cast(acc, tf.float32), name="accuracy")

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Create a summary to monitor cost tensor
tf.summary.scalar("loss", cost)
# Create a summary to monitor accuracy tensor
tf.summary.scalar("accuracy", acc)
# Create a summary to visualize input images
tf.summary.image("input", tf.reshape(x, (50, 28, 28, 1)), 9)

# Merge all summaries into a single op
merged_summary_op = tf.summary.merge_all()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    saver = tf.train.Saver(tf.global_variables())

    # op to write logs to Tensorboard
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    summary_writer.add_graph(sess.graph)

    # sess.run(images.initializer)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Run optimization op (backprop), cost op (to get loss value)
            # and summary nodes
            _, c, summary = sess.run([optimizer, cost, merged_summary_op],
                                     feed_dict={x: batch_xs, y: batch_ys})
            # Write logs at every iteration
            summary_writer.add_summary(summary, epoch * total_batch + i)
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_epoch == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))

        if epoch % 2 == 0:
            saver.save(sess, os.path.join(logs_path, 'images.ckpt'))

    print("Optimization Finished!")

    # Test model
    # Calculate accuracy
    print("Accuracy:", acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    print("Run the command line:\n" \
          "--> tensorboard --logdir={}".format(cmd_path),
          "\nThen open http://0.0.0.0:6006/ into your web browser")

    tf.train.write_graph(sess.graph.as_graph_def(), logs_path, 'train.pb')
    input_graph_path = os.path.join(logs_path, 'train.pb')
    input_saver_def_path = ""
    input_binary = False
    input_checkpoint_path = os.path.join(logs_path, 'images.ckpt')
    output_graph_path = os.path.join(logs_path, 'train_model.pb')
    clear_devices = False
    output_node_names = "output_tensor,input_tensor,labeled_data,Weights,Bias,accuracy"
    restore_op_name = "save/restore_all"
    filename_tensor_name = "save/Const:0"
    initializer_nodes = ""
    freeze_graph.freeze_graph(input_graph_path,
                              input_saver_def_path,
                              input_binary,
                              input_checkpoint_path,
                              output_node_names,
                              restore_op_name,
                              filename_tensor_name,
                              output_graph_path,
                              clear_devices,
                              initializer_nodes)

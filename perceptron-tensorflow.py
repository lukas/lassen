import tensorflow as tf
import data
import numpy as np

images, one_hot_labels = data.load_mnist("data/train-images-idx3-ubyte", "data/train-labels-idx1-ubyte")
test_images, one_hot_test_labels = data.load_mnist("data/t10k-images-idx3-ubyte","data/t10k-labels-idx1-ubyte")

num_classes = 10
num_images = images.shape[0]
num_pixels = images.shape[1]

num_test_images = len(one_hot_test_labels)

sess = tf.InteractiveSession()

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x,W) + b

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

for _ in range(1000):
  #batch = mnist.train.next_batch(100)
  rand_idx = np.floor(np.multiply(np.random.rand(100), num_images)).astype(int)
  batch_labels = one_hot_labels[rand_idx,:]
  batch_images = images[rand_idx,:]
  train_step.run(feed_dict={x: batch_images, y_: batch_labels})

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print(accuracy.eval(feed_dict={x: test_images, y_: one_hot_test_labels}))

saver = tf.train.Saver()

save_path = saver.save(sess, "tensorflow-checkpoint")

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

mnist = input_data.read_data_sets("./MNIST/data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.matmul(x, W) + b
y_ = tf.placeholder(tf.float32, [None, 10])

cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))
summary_cross_entropy = tf.summary.scalar("cross_entropy", cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.InteractiveSession()
train_writer = tf.summary.FileWriter("./MNIST/result/", sess.graph)
summary = tf.summary.merge_all()

tf.global_variables_initializer().run()
# Train
for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    _, summary_value = sess.run([train_step, summary], feed_dict={x: batch_xs, y_: batch_ys})

    train_writer.add_summary(summary_value,i)

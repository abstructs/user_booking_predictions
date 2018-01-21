import csv
import numpy as np
import tensorflow as tf
import temp
import os

# DataLoader = temp.DataLoader()

Model = temp.Model()

Model.train(10000)

# [X, Y, classification_count] = temp.load_data(5)

# x_train = tf.placeholder(dtype=tf.float32, shape=X.shape, name="X")
# y_train = tf.placeholder(dtype=tf.float32, shape=Y.shape, name="Y")

# [x_test, y_test, _] = temp.load_data(os.path.relpath("data/test_users.csv", cur_path))

# W = tf.get_variable("W", shape=[classification_count, X.shape[1]], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64))
# b = tf.Variable(tf.zeros((1, 1)))

# p_x = tf.matmul(x_train, tf.transpose(W)) + b

# # explicitly broadcast p_x
# p_x = p_x + tf.zeros((tf.shape(p_x)[0], classification_count))

# # print(tf.Session().run(y_train, feed_dict={y_train: Y}))

# cost = tf.losses.softmax_cross_entropy(onehot_labels=y_train, logits=p_x)

# opt = tf.train.AdamOptimizer(0.01).minimize(cost)

# init = tf.global_variables_initializer()



# with tf.Session() as sess:
#     sess.run(init)
#     for i in range(0, 15000):
#         [c,w,_] = sess.run([cost, W, opt], feed_dict={x_train: X, y_train: Y})
#         if i % 1000 == 0:
#             print("Cost after " + str(i) + " iterations: " + str(c))
    
#     temp.save_weights(w)
#     w = temp.load_weights()

#     temp.get_accuracy(w, X, Y, classification_count)
#     train_acc = sess.run(temp.get_accuracy(w, X, Y, classification_count))
#     test_acc = sess.run(temp.get_accuracy(w, x_test, y_test, classification_count))

#     print("Training accuracy: %.2f" % train_acc)
#     print("Testing accuracy: %.2f" % test_acc)
        
import numpy as np
import tensorflow as tf
import os
import dataloader

# manages interating with the model

class Model:
    def __init__(self):
        self.DataLoader = dataloader.DataLoader()
        [self.x_train, self.y_train, self.classification_count] = self.DataLoader.load_data("data/train_users_2.csv", (0,100))
        [self.x_test, self.y_test, _] = self.DataLoader.load_data("data/train_users_2.csv", (100,200))
        
        self.parameters = {
            "W": tf.get_variable("W", shape=[self.classification_count, self.x_train.shape[1]], initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64)), 
            "b": tf.Variable(tf.zeros((1, 1)))
        }

        self.cost_function = self.get_cost()
        self.optimizer = self.get_optimizer()

        self.init = tf.global_variables_initializer()

    def train(self, num_iterations=100):
        """
        EFFECTS: minimizes the cost function and saves the weights
        """
        with tf.Session() as sess:
            sess.run(self.init)
            for i in range(0, num_iterations):
                [c,w,_] = sess.run([self.cost_function, self.parameters["W"], self.optimizer], feed_dict={"X:0": self.x_train, "Y:0": self.y_train})
                if i % 1000 == 0:
                    print("Cost after " + str(i) + " iterations: " + str(c))
            
            self.DataLoader.save_weights(w)
    
    def get_weights(self):
        return self.DataLoader.load_weights()

    def get_cost(self):        
        placeholder_y = tf.placeholder(dtype=tf.float32, shape=self.y_train.shape, name="Y")
        return tf.losses.softmax_cross_entropy(onehot_labels=placeholder_y, logits=self.get_activation())

    def get_activation(self):
        placeholder_x = tf.placeholder(dtype=tf.float32, shape=self.x_train.shape, name="X")
        return tf.add(tf.matmul(placeholder_x, tf.transpose(self.parameters["W"])), self.parameters["b"])

    def get_optimizer(self):
        return tf.train.AdamOptimizer(0.01).minimize(self.cost_function)

    def predict(self, W, X):
        """
        EFFECTS: uses argmax to return the index corresponding to the country
                 the model predicts
        """
        p_x = tf.multiply(X, tf.transpose(W))

        predictions = tf.argmax(p_x, 1)

        return tf.cast(tf.one_hot(predictions, self.classification_count), tf.float64)


    def get_accuracy(self, distribution="test"):
        """
        EFFECTS: does a comparison on the predicted values and expected values
                returns the correct predictions over the total number of predictions
        """

        W = self.get_weights()

        if distribution == "test":
            Y = self.y_test
            X = self.x_test
        elif distribution == "train":
            Y = self.y_train
            X = self.x_train
        else:
            raise ValueError('Distribution should be set to "test" or "train"')

        Y = tf.cast(Y, dtype=tf.float64)
        X = tf.cast(X, dtype=tf.float64)
        predictions = self.predict(W, X)

        comparison = tf.equal(tf.argmax(predictions, 1), tf.Session().run(tf.argmax(Y, 1)))

        total_predictions = tf.size(predictions, out_type=tf.float64)
        
        correct_predictions = tf.reduce_sum(tf.cast(comparison, dtype=tf.float64))

        training_accuracy = tf.divide(correct_predictions, total_predictions)
    
        return tf.Session().run(training_accuracy)
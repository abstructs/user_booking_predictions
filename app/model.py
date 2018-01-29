import numpy as np
import tensorflow as tf
import dataloader


# manages interating with the model

class Model:
    def __init__(self):
        self.DataLoader = dataloader.DataLoader()
        [self.x_train, self.y_train, self.classification_count] = self.DataLoader.get_data("data/train_users_2.csv", (0, 100000))

        # self.y_train = np.reshape(np.array([row[0] for row in tf.Session().run(self.y_train)]), (self.x_train.shape[0], 1))

        # print(self.y_train)
        # return
        # self.DataLoader.load_data("data/countries.csv")
        # self.x_train = [[param ** i for i, param in enumerate(row)] for row in self.x_train]

        # self.x_train = np.array([param in enumerate(self.x_train)])

        self.x_train = tf.Session().run(tf.nn.l2_normalize(self.x_train, 0))

        self.x_test = self.DataLoader.get_data("data/test_users.csv", (0, 100000), True)
        # print(self.x_test.shape)

        # self.x_test = [row + self.y_test[i] for i,row in enumerate(self.x_test)]

        self.parameters = {
            "W": tf.get_variable("W", shape=[self.classification_count, self.x_train.shape[1]],
                                 initializer=tf.contrib.layers.xavier_initializer(dtype=tf.float64),
                                 dtype=tf.float64, regularizer=self.get_regularizer()),
            "b": tf.ones((1, 1), dtype=tf.float64)
        }

        self.activation = self.get_activation()
        self.cost_function = self.get_cost()
        self.optimizer = self.get_optimizer()


    def train(self, num_iterations=100):
        """
        EFFECTS: minimizes the cost function and saves the weights
        """


        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)

            # print(sess.run(self.cost_function, feed_dict={"X:0": self.x_train, "Y:0": self.y_train}))

            # a = sess.run(self.cost_function, feed_dict={"X:0": self.x_train, "Y:0": self.y_train})

            # print(a)
            for i in range(0, num_iterations):
                [c, w, _] = sess.run([self.cost_function, self.parameters["W"], self.optimizer],
                                     feed_dict={"X:0": self.x_train, "Y:0": self.y_train})
                if i % 1000 == 0:
                    print("Cost after " + str(i) + " iterations: " + str(c))

            self.DataLoader.save_weights(sess)

    def get_weights(self, sess):
        return self.DataLoader.get_weights(sess)

    def get_regularizer(self):
        return tf.contrib.layers.l2_regularizer(.8)

    def get_cost(self):
        y = tf.placeholder(dtype=tf.float64, shape=self.y_train.shape, name="Y")

        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y_train, logits=self.activation))

        # return self.activation
        # return tf.losses.softmax_cross_entropy(onehot_labels=self.y_train, logits=self.x_train, weights=self.parameters["W"])
        # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=placeholder_y, logits=self.get_activation()))
        # return tf.contrib.losses.sigmoid_cross_entropy_with_logits(multi_class_labels=placeholder_y, logits=self.get_activation())
        # return tf.nn.sigmoid_cross_entropy_with_logits(labels=placeholder_y, logits=self.get_activation())

    def get_activation(self):
        placeholder_x = tf.placeholder(dtype=tf.float64, shape=self.x_train.shape, name="X")

        # return tf.add(tf.matmul(placeholder_x, tf.transpose(self.parameters["W"])), self.parameters["b"])

        return tf.add(tf.matmul(placeholder_x, tf.transpose(self.parameters["W"])), self.parameters["b"])

    def get_optimizer(self):
        return tf.train.AdamOptimizer(.3).minimize(self.cost_function)

    def predict(self, X):
        """
        EFFECTS: uses argmax to return the index corresponding to the country
                 the model predicts
        """

        W = tf.cast(self.get_weights(tf.Session()), dtype=tf.float64)

        p_x = tf.matmul(X, tf.transpose(W)) + self.parameters["b"]

        return tf.Session().run(tf.argmax(p_x, 1))

        # predictions = tf.argmax(p_x, 1)

        # print(tf.Session().run(predictions))

        # print([prediction for prediction in tf.Session().run(p_x)])

        # return p_x

    def output_data(self):
        """
        EFFECTS: outputs a csv with the user ids and predictions
        :return:
        """
        predictions = self.predict(self.x_test)
        str = "id,country\n"
        # print(predictions)
        for i, prediction in enumerate(predictions):
            data_map = self.DataLoader.data_map['country_destination']

            str += self.DataLoader.user_ids[i] + "," + list(data_map.keys())[list(data_map.values()).index(prediction)] + '\n'

        with open("submission.csv", "w") as csvfile:
            csvfile.write(str)

        # print(str)
            # print(prediction)
            # print(list(data_map.keys())[list(data_map.values()).index(prediction)])
        #     print(list(data_map.values()))
            # for country, index in self.DataLoader.data_map['country_destination'].iteritems():
            #     print(country)

    def get_accuracy(self, distribution="test"):
        """
        EFFECTS: does a comparison on the predicted values and expected values
                returns the correct predictions over the total number of predictions
        """

        if distribution == "test":
            Y = self.y_test
            X = self.x_test
        elif distribution == "train":
            Y = self.y_train
            X = self.x_train
        else:
            raise ValueError('Distribution should be set to "test" or "train"')

        # print(self.DataLoader.data_map)
        Y = tf.cast(Y, dtype=tf.float64)
        X = tf.cast(X, dtype=tf.float64)
        predictions = self.predict(X)

        # predictions = tf.argmax(predictions, 1)

        Y = tf.argmax(Y, -1)

        with tf.Session() as sess:
            Y = sess.run(Y)

            equal = tf.equal(Y, predictions)
            acc = tf.reduce_mean(tf.cast(equal, tf.float32))

            return sess.run(acc)
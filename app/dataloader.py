import csv
import numpy as np
import tensorflow as tf
import os


# manages loading the data into vectors X and Y
class DataLoader:
    def __init__(self, cur_path=os.path.dirname(__file__)):
        self.cur_path = cur_path
        self.data_map = {}

    def save_weights(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.cur_path + "/data/my_model")

    def load_weights(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, self.cur_path + "/data/my_model")
        return sess.run('W:0')

    def try_parse(self, num):
        """
        EFFECTS: tries to parse num to a float, returns false if not possible
        """
        try:
            return float(num)
        except ValueError:
            return False

    def load_params(self, labels, file_content):
        """
        EFFECTS: takes file content and loads the parameters from it
                 creates a dictionary with categories of data mapped to classes
                 for numeric values, they are parsed and added to the parameter array
                 
                 data map is stored in "self.data_map" with key equal to "key_label"
        """
        class_map = self.data_map
        params = []

        for label in labels:
            if not (label in class_map):
                class_map[label] = {}

        for row in file_content:
            param_list = []
            for i, val in enumerate(row):
                if val == '':
                    param_list.append(0)
                    continue
                # try to parse param to float, add it to params and continue if convertable

                num = self.try_parse(val)
                if num:
                    param_list.append(num)
                    continue

                # gets dict for label with the entry
                map_for_label = class_map[labels[i]]

                if type(val) is str and not (val in map_for_label):
                    try:
                        map_for_label.update({val: max(map_for_label.values()) + 1})
                    except ValueError:
                        map_for_label.update({val: 1})

                # get class for value and add param to list
                param_list.append(class_map[labels[i]][val])

            params.append(param_list)

        # self.data_map.update(class_map)

        return params

    def load_country_data(self, user_params, user_labels):
        """
        EFFECTS: returns a matrix of statistics for each country in the labels
        """

        with open(self.cur_path + "/data/countries.csv", "r") as csvfile:
            reader = csv.reader(csvfile, quotechar='|')

            file_contents = list(reader)
            labels = file_contents[0]

            params = self.load_params(labels, file_contents[1:])

            return params

    def load_user_data(self, file_name=False, training_example_range=(0, 100)):
        """
        EFFECTS:
        """

        X = []
        Y = []

        with open(self.cur_path + "/" + file_name, 'r') as csvfile:
            reader = csv.reader(csvfile, quotechar='|')
            # -2 so we don't count the labels
            i = -2

            # read X and Y values from training data set

            file_contents = list(reader)
            labels = file_contents[0]

            # strip out first 4 columns
            file_contents = [row[4:] for row in
                             file_contents[(training_example_range[0] + 1):training_example_range[1] + 1]]

            labels = labels[4:]

            params = self.load_params(labels, file_contents)

            for row in params:
                X.append(row[:-1])
                Y.append(row[-1])

        X = np.array(X)
        X = X.astype(float)

        # print(X)

        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0], 1))
        Y = Y.astype(np.float64)

        # the number of classes we have
        classification_count = np.max(Y)
        #
        # one_hot_matrix = tf.one_hot(tf.cast(Y, tf.int32), classification_count)
        #
        # one_hot_matrix = tf.reshape(one_hot_matrix, (tf.shape(Y)[0], classification_count))

        return X, Y, classification_count

    def load_data(self, file_name=False, training_example_range=(0, 100)):
        """
        EFFECTS: loads the parameters X and target Y into two vectors
                 from the file specified in "file_name"
                 also returns the number of classifications tuple returned is in
                 form (X, Y, classificaiton_count)
        """
        user_params, user_labels, classification_count = self.load_user_data(file_name, training_example_range)

        country_data = self.load_country_data(user_params, user_labels)

        # convert all arrays to numpy arrays
        country_data = np.array([np.array(row) for row in country_data])

        country_params = np.zeros((user_params.shape[0], country_data.shape[1]))

        # find the country statistics that match the user's destination country
        for i, params in enumerate(country_params):
            new_param = np.array(list(filter(lambda row: int(user_labels[i][0]) == int(row[0]), country_data)))
            if len(new_param) == 0:
                continue

            country_params[i, :] = new_param

        params = np.append(user_params, country_params, 1)

        one_hot_matrix = tf.one_hot(tf.cast(user_labels, tf.int32), classification_count)

        one_hot_matrix = tf.reshape(one_hot_matrix, (tf.shape(user_labels)[0], classification_count))

        return params, tf.Session().run(one_hot_matrix), classification_count

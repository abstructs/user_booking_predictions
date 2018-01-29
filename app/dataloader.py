import csv
import numpy as np
import tensorflow as tf
import os
import json


# manages loading the data into vectors X and Y
class DataLoader:
    def __init__(self, cur_path=os.path.dirname(__file__)):
        self.cur_path = cur_path

        try:
            f = open(self.cur_path + '/data_map.json')
            self.data_map = json.load(f)
        except json.decoder.JSONDecodeError:
            self.data_map = {}

    def save_weights(self, sess):
        saver = tf.train.Saver()
        saver.save(sess, self.cur_path + "/data/my_model")

    def get_weights(self, sess):
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

    def get_params(self, labels, file_content):
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

        return params

    def get_country_params(self, country_data, user_labels):
        """
        :param country_data: The file contents from the country.csv file
        :param user_labels: the target labels for each user
        EFFECTS: returns a matrix of parameters for each user corresponding to the target country
        """
        country_params = np.zeros((user_labels.shape[0], country_data.shape[1]))

        # find the country statistics that match the user's destination country
        for i, params in enumerate(country_params):
            new_param = np.array(list(filter(lambda row: int(user_labels[i][0]) == int(row[0]), country_data)))
            if len(new_param) == 0:
                continue

            country_params[i, :] = new_param
        return country_params


    def get_country_data(self, user_params, user_labels):
        """
        EFFECTS: returns a matrix of statistics for each country in the labels
        """

        with open(self.cur_path + "/data/countries.csv", "r") as csvfile:
            reader = csv.reader(csvfile, quotechar='|')

            file_contents = list(reader)
            labels = file_contents[0]

            params = self.get_params(labels, file_contents[1:])

            return params

    def get_user_data(self, file_name=False, training_example_range=(0, 100)):
        """
        EFFECTS: returns a matrix of parameters for users
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

            self.user_ids = [row[0] for row in
                        file_contents[(training_example_range[0] + 1):training_example_range[1] + 1]]

            # strip out first 4 columns
            file_contents = [row[4:] for row in
                             file_contents[(training_example_range[0] + 1):training_example_range[1] + 1]]

            labels = labels[4:]

            params = self.get_params(labels, file_contents)

            for row in params:
                X.append(row[:-1])
                Y.append(row[-1])

        X = np.array(X)
        X = X.astype(float)

        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0], 1))
        Y = Y.astype(np.float64)

        # the number of classes we have
        classification_count = np.max(Y)

        return X, Y, 12

    def get_age_bucket_data(self):
        """
        EFFECTS: Gets data from the age/gender file
        """

        with open(self.cur_path + "/data/age_gender_bkts.csv", 'r') as csvfile:
            reader = csv.reader(csvfile, quotechar='|')
            file_contents = list(reader)
            # labels = file_contents[0]

            return file_contents

    def get_age_bucket_params(self, user_params, user_labels):
        age_bucket_data = self.get_age_bucket_data()

        data = []
        labels = age_bucket_data[0]
        bucket_params = np.zeros((user_params.shape[0], len(age_bucket_data[0]) - 2))

        for row in age_bucket_data[1:]:
            age_range = tuple([int(num) for num in row[labels.index("age_bucket")].strip('+').split('-')])
            country = self.data_map['country_destination'][row[labels.index("country_destination")]]
            gender = self.data_map['gender'][row[labels.index("gender")].upper()]
            pop_in_thousands = float(row[labels.index("population_in_thousands")])
            data.append([age_range] + [country] + [gender] + [pop_in_thousands])

        def is_in_bucket(age_range, user_vector):
            user_age = user_vector[1]
            if len(age_range) == 1:
                return user_age >= age_range[0]
            return age_range[0] <= user_age <= age_range[1]

        # get the params out of our data that correspond with the user's information
        for i, row in enumerate(user_params):
            user_target = int(user_labels[i][0])

            params_for_country = list(filter(lambda row: row[1] == user_target, data))

            curr_user = user_params[i]

            bucket_for_gender = list(filter(lambda row: is_in_bucket(row[labels.index("age_bucket")], curr_user), params_for_country))

            user_gender = curr_user[0]

            new_params = list(filter(lambda row: row[labels.index("gender")] == user_gender, bucket_for_gender))

            if len(new_params) == 0:
                continue

            bucket_params[i] = new_params[0][1:]

        return bucket_params

    def get_data(self, file_name=False, training_example_range=(0, 100), no_labels=False):
        """
        EFFECTS: loads the parameters X and target Y into two vectors
                 from the file specified in "file_name"
                 also returns the number of classifications tuple returned is in
                 form (X, Y, classificaiton_count)
        """
        user_params, user_labels, classification_count = self.get_user_data(file_name, training_example_range)

        # get data in a numpy array
        country_data = np.array([np.array(row) for row in self.get_country_data(user_params, user_labels)])

        country_params = self.get_country_params(country_data, user_labels)

        # age_bucket_data = self.get_age_bucket_data()
        age_bucket_params = self.get_age_bucket_params(user_params, user_labels)

        params = np.append(np.append(user_params, country_params, 1), age_bucket_params, 1)

        if no_labels:
            return np.array([np.append(row, user_labels[i]) for i, row in enumerate(params)])

        # return params, tf.Session().run(tf.cast(user_labels, tf.float64)), classification_count

        one_hot_matrix = tf.one_hot(tf.cast(user_labels, tf.int32), classification_count)

        # one_hot_matrix = tf.reshape(one_hot_matrix, (tf.shape(user_labels)[0], classification_count))

        with open(self.cur_path + '/data_map.json', 'w') as f:
            json.dump(self.data_map, f)

        return params, tf.Session().run(one_hot_matrix), classification_count

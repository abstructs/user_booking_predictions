import csv
import numpy as np
import tensorflow as tf
import os

# manages loading the data into vectors X and Y
class DataLoader:
    def __init__(self, cur_path=os.path.dirname(__file__)):
        self.cur_path = cur_path

    def save_weights(self, W):
        str = ""
        f = open(self.cur_path + "/data/weights.txt", "w")
        for arr in W:
            for i in arr:
                str += "%f," % i
        print(str)
        f.write(str)
        f.close()

    def load_weights(self):
        f = open(self.cur_path + "/data/weights.txt", "r")
        file_contents = f.read()
        
        np_arr = np.asarray(str.split(file_contents, ",")[:-1], dtype=np.float64)

        return np_arr.reshape((np_arr.shape[0], 1))

    def load_params(self, row):
        """
        EFFECTS: Takes a row of data and loads the parameters into an 
                array of size (training_example_count, param_count)
        """
        
        # print(row)

        return row[5]

    def categories_to_classes(self, categories):
        """
        EFFECTS: Takes a list of categories and maps the category name to
                a number that represents that category
        """


        for i in range(categories):
            # categories[i]

            Y[indicies] = i

    def classify_data(self, Y):
        """
        EFFECTS: takes a Y vector of countries and an array of countries
                and returns a one hot matrix with the data properly classified
                the tuple is in form of (one_hot_matrix, classification_count)
        """

        countries = self.get_countries()
        
        # classify where the user booked their location
        classification_count = len(countries)

        # category_map = categories_to_classes(countries)
        
        for i in range(classification_count):
            indicies = np.where(Y == countries[i])

            Y[indicies] = i
        
            # print(np.where(Y == country))

        # special cases
        # NDF means user did not book a location
        ndf_indicies = np.where(Y == 'NDF')
        Y[ndf_indicies] = len(countries)
        other_indicies = np.where(Y == 'other')
        Y[other_indicies] = len(countries) + 1
        
        Y = Y.astype(int)
        # print(Y)
        # get one hot
        tf.one_hot(Y, classification_count)

        one_hot_matrix = tf.one_hot(tf.cast(Y, tf.int32), classification_count + 2)

        one_hot_matrix = tf.reshape(one_hot_matrix, (tf.shape(Y)[0], 12))

        return (one_hot_matrix, classification_count + 2)

    def get_countries(self):
        """
        EFFECTS: returns an array of country codes that are used in
                the dataset.
        """
        with open(self.cur_path + "/data/countries.csv", "r") as csvfile:
            arr = []
            
            reader = csv.reader(csvfile, quotechar='|')
            for row in reader:
                arr.append(row[0])
            return arr[1:]

    def load_data(self, file_name, start_from=0, training_example_range=(0, 100)):
        """
        EFFECTS: loads the parameters X and target Y into two vectors
                 from the file specified in "file_name"
                 also returns the number of classifications tuple returned is in
                 form (X, Y, classificaiton_count)

                 start_from: the line to start reading data from
                 training_example_count: amount of training examples to read
        """

        X = []
        Y = []
        
        with open(self.cur_path + "/" + file_name, 'r') as csvfile:
            reader = csv.reader(csvfile,  quotechar='|')
            # -2 so we don't count the labels
            i = -2

            # read X and Y values from training data set

            file_contents = list(reader)
            for i in range(training_example_range[0] + 1, training_example_range[1] + 2):
                X.append(self.load_params(file_contents[i]))
                Y.append(file_contents[i][-1])
        
        # strip labels and verify vector shapes with reshape
        X = np.array(X[1:])
        
        X[X==''] = '0'
        X = X.astype(float)
        X = np.reshape(X, (X.shape[0], 1))
        Y = np.array(Y[1:])
        Y = np.reshape(Y, (Y.shape[0], 1))
        
        [one_hot_matrix, classification_count] = self.classify_data(Y)

        return (X, tf.Session().run(one_hot_matrix), classification_count)

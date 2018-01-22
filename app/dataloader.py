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
        # str = ""
        # f = open(self.cur_path + "/data/weights.txt", "w")
        # for arr in W:
        #     for i in arr:
        #         str += "%f," % i
        
        # f.write(str)
        # f.close()

    def load_weights(self, sess):
        saver = tf.train.Saver()
        saver.restore(sess, self.cur_path + "/data/my_model")
        return sess.run('W:0')
        # f = open(self.cur_path + "/data/weights.txt", "r")
        # file_contents = f.read()
        
        # np_arr = np.asarray(str.split(file_contents, ",")[:-1], dtype=np.float64)

        # return np_arr.reshape((np_arr.shape[0], 1))

    
    def try_parse(self, num):
        """
        EFFECTS: tries to parse num to a float, returns false if not possible
        """
        try:
            return float(num)
        except ValueError:
            return False

    def load_params(self, labels, file_content, key_label):
        """
        EFFECTS: takes file content and loads the parameters from it
                 creates a dictionary with categories of data mapped to classes
                 for numeric values, they are parsed and added to the parameter array
                 
                 data map is stored in "self.data_map" with key equal to "key_label"
        """
        class_map = {}
        params = []

        for label in labels:
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

                # get class for value
                param_list.append(class_map[labels[i]][val])

            params.append(param_list)

        self.data_map.update({key_label: class_map})

        return params


    # def categories_to_classes(self, categories):
    #     """
    #     EFFECTS: Takes a list of categories and maps the category name to
    #             a number that represents that category
    #     """

    #     for i in range(categories):
    #         # categories[i]

    #         Y[indicies] = i

    # def classify_data(self, Y):
    #     """
    #     EFFECTS: takes a Y vector of countries and an array of countries
    #             and returns a one hot matrix with the data properly classified
    #             the tuple is in form of (one_hot_matrix, classification_count)
    #     """

    #     countries = self.get_countries()
        
    #     # classify where the user booked their location
    #     classification_count = len(countries)

    #     # category_map = categories_to_classes(countries)
        
    #     for i in range(classification_count):
    #         indicies = np.where(Y == countries[i])

    #         Y[indicies] = i

    #     # special cases
    #     # NDF means user did not book a location
    #     ndf_indicies = np.where(Y == 'NDF')
    #     Y[ndf_indicies] = len(countries)
    #     other_indicies = np.where(Y == 'other')
    #     Y[other_indicies] = len(countries) + 1
        
    #     Y = Y.astype(int)
        
    #     # get one hot
    #     tf.one_hot(Y, classification_count)

    #     one_hot_matrix = tf.one_hot(tf.cast(Y, tf.int32), classification_count + 2)

    #     one_hot_matrix = tf.reshape(one_hot_matrix, (tf.shape(Y)[0], 12))

    #     return (one_hot_matrix, classification_count + 2)

    # def get_countries(self):
    #     """
    #     EFFECTS: returns an array of country codes that are used in
    #             the dataset.
    #     """
    #     with open(self.cur_path + "/data/countries.csv", "r") as csvfile:
    #         arr = []
            
    #         reader = csv.reader(csvfile, quotechar='|')
    #         for row in reader:
    #             arr.append(row[0])
    #         return arr[1:]

    def load_data(self, file_name, training_example_range=(0, 100)):
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
            labels = file_contents[0]

            # strip out first 4 columns
            file_contents = [row[4:] for row in file_contents[(training_example_range[0]+1):training_example_range[1]+1]]
            labels = labels[4:]

            

            params = self.load_params(labels, file_contents, file_name)

            for row in params:
                X.append(row[:-1])
                Y.append(row[-1])
        

        X = np.array(X)
        X = X.astype(float)
        
        Y = np.array(Y)
        Y = np.reshape(Y, (Y.shape[0], 1))
        Y = Y.astype(float)

        # the number of classes we have
        classification_count = np.max(Y)
        
        one_hot_matrix = tf.one_hot(tf.cast(Y, tf.int32), classification_count)

        one_hot_matrix = tf.reshape(one_hot_matrix, (tf.shape(Y)[0], classification_count))

        return (X, tf.Session().run(one_hot_matrix), classification_count)

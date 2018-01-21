import csv
import numpy as np
import tensorflow as tf
import os

# data_labels = {
#     countries: {
#         'US': 1
#     }
# }

cur_path = os.path.dirname(__file__)

def get_countries():
    """
    EFFECTS: returns an array of country codes that are used in
             the dataset.
    """
    with open(cur_path + "/data/countries.csv", "r") as csvfile:
        arr = []
        
        reader = csv.reader(csvfile, quotechar='|')
        for row in reader:
            arr.append(row[0])
        return arr[1:]

# def categories_to_classes(categories):
#     """
#     EFFECTS: Takes a list of categories and maps the category name to
#              a number that represents that category
#     """


#     for i in range(categories):
#         categories[i]

#         Y[indicies] = i


def classify_data(Y, countries):
    """
    EFFECTS: takes a Y vector of countries and an array of countries
            and returns a one hot matrix with the data properly classified
            the tuple is in form of (one_hot_matrix, classification_count)
    """

    countries = get_countries()
    
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

def load_params(row):
    """
    EFFECTS: Takes a row of data and loads the parameters into an 
             array of size (training_example_count, param_count)
    """
    
    print(row)

    return row[5]

def load_data(training_example_count=100, file_path=cur_path + "/data/train_users_2.csv"):
    """
    EFFECTS: loads the parameters X and target Y into two vectors
             also returns the number of classifications
             tuple returned is in form (X, Y, classificaiton_count)
    """

    X = []
    Y = []
    
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile,  quotechar='|')
        # -2 so we don't count the labels
        i = -2

        # read X and Y values from training data set

        for row in reader:
            i += 1
            
            if i == training_example_count:
                break
            X.append(load_params(row))
            Y.append(row[-1])

    
    # strip labels and verify vector shapes with reshape
    X = np.array(X[1:])
    X[X==''] = '0'
    X = X.astype(float)
    X = np.reshape(X, (X.shape[0], 1))
    Y = np.array(Y[1:])
    Y = np.reshape(Y, (Y.shape[0], 1))
    
    [one_hot_matrix, classification_count] = classify_data(Y, get_countries())

    return (X, tf.Session().run(one_hot_matrix), classification_count)
    

def predict(W, X, classification_count):
    """
    EFFECTS: uses argmax to return the index corresponding to the country
             the model predicts
    """
    p_x = tf.multiply(X, tf.transpose(W))

    predictions = tf.argmax(p_x, 1)

    return tf.cast(tf.one_hot(predictions, classification_count), tf.float64)

def save_weights(W):
    str = ""
    f = open(cur_path + "/data/weights.txt", "w")
    for arr in W:
        for i in arr:
            str += "%f," % i
    print(str)
    f.write(str)
    f.close()

def load_weights():
    f = open(cur_path + "/data/weights.txt", "r")
    file_contents = f.read()
    
    np_arr = np.asarray(str.split(file_contents, ",")[:-1], dtype=np.float64)

    return np_arr.reshape((np_arr.shape[0], 1))

def get_accuracy(W, X, Y, classification_count):
    """
    EFFECTS: does a comparison on the predicted values and expected values
             returns the correct predictions over the total number of predictions
    """
    Y = tf.cast(Y, dtype=tf.float64)
    X = tf.cast(X, dtype=tf.float64)
    predictions = predict(W, X, classification_count)

    comparison = tf.equal(tf.argmax(predictions, 1), tf.Session().run(tf.argmax(Y, 1)))

    # tf.equal(tf.arg_max(predictions), tf.arg_max(Y))

    # print(tf.Session().run(predictions))

    total_predictions = tf.size(predictions, out_type=tf.float64)
    
    correct_predictions = tf.reduce_sum(tf.cast(comparison, dtype=tf.float64))
    
    # print(correct_predictions)

    training_accuracy = tf.divide(correct_predictions, total_predictions)

    return training_accuracy
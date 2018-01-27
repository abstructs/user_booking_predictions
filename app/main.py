import csv
import numpy as np
import tensorflow as tf
import model
import dataloader
import os

Model = model.Model()

# dl = dataloader.DataLoader()

# x_test = dl.get_data("data/test_users.csv", (0, 100000), True)
#
# print(Model.predict(x_test))

# [x_test, y, _] = dl.get_data("data/test_users.csv", (0, 100))

# print(Model.predict(Model.x_test))

# x_test = [np.append(row, y[i]) for i,row in enumerate(x_test)]

# print(Model.predict(x_test))

Model.train(10000)

# Model.load_data()

# Model.get_accuracy("train")
# Model.get_accuracy("train")

print("Training accuracy: %.2f" % Model.get_accuracy("train"))
# print("Testing accuracy: %.2f" % Model.get_accuracy("test"))

Model.output_data()

#  print(Model.predict(Model.x_test))
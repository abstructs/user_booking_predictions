import csv
import numpy as np
import tensorflow as tf
import model
import dataloader
import os

Model = model.Model()

dl = dataloader.DataLoader()

[x_test, y_test, _] = dl.DataLoader.get_data("data/test_users.csv", (0, 100000))

# Model.train(10000)

# Model.load_data()

# Model.get_accuracy("train")
# Model.get_accuracy("train")

# print("Training accuracy: %.2f" % Model.get_accuracy("train"))
# print("Testing accuracy: %.2f" % Model.get_accuracy("test"))

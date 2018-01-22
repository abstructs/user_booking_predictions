import csv
import numpy as np
import tensorflow as tf
import model
import os

Model = model.Model()

Model.train(20000)

# Model.load_data()

print("Training accuracy: %.2f" % Model.get_accuracy("train"))
print("Testing accuracy: %.2f" % Model.get_accuracy("test"))

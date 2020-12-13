from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split 
import random 
import tensorflow as tf
import os.path

dataset_splits = os.listdir("seg_train_images")

#print(dataset_splits)

X_train, X_test = train_test_split(dataset_splits, test_size=0.2,
               random_state=True)

for i in range(len(X_train)):
    print(X_train[i])

print("---------------------------")

for i in range(len(X_test)):
    print(X_test[i])


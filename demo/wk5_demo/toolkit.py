"""
Ultilities for ML education
"""
import numpy as np
import pickle

import tensorflow as tf

"""
Class and functions for MNIST
@SNTs
"""
def _change_ont_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1

    return T

class MNIST(object):
    def __init__(self,download=True):
        if download:
            #print("here")
            (x_train, y_train), (x_test, y_test)  = tf.keras.datasets.mnist.load_data()
            x_train = np.reshape(x_train,(-1,784))
            x_test = np.reshape(x_test,(-1,784))
        else:
            (x_train, y_train), (x_test, y_test) = self.load_mnist_from_local_files(normalize=False, flatten=True)
        self.x_train = x_train[:50000,:]
        self.y_train = y_train[:50000]
        self.x_val   = x_train[50000:,:]
        self.y_val   = y_train[50000:]
        self.x_test  = x_test[:5000,:]
        self.y_test  = y_test[:5000]
        
    def load_mnist_from_local_files(self,normalize=True,flatten=True,one_hot_label=False):
        save_file = "./mnist/mnist.pkl"
        with open(save_file, 'rb') as f:
            dataset = pickle.load(f)

        if normalize:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].astype(np.float32)
                dataset[key] /= 255.0

        if not flatten:
            for key in ('train_img', 'test_img'):
                dataset[key] = dataset[key].reshape(-1, 1, 28, 28)

        if one_hot_label:
            dataset['train_label'] = _change_ont_hot_label(dataset['train_label'])
            dataset['test_label'] = _change_ont_hot_label(dataset['test_label'])

        return (dataset['train_img'], dataset['train_label']), (dataset['test_img'], dataset['test_label'])


    def samples(self,batch_size=1):
        x,y = self.mnist.train.next_batch(batch_size)
        return x

    @property
    def dimension(self):
        return  784

    

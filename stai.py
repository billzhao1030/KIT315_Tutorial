"""
Ultilities for ML education
@SNT
"""
import numpy as np
import pandas as pd
import seaborn as sn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import cv2 as cv

import glob
import os

from tensorflow.examples.tutorials.mnist import input_data

# import keras.datasets.mnist
# input_data = keras.datasets.mnist.load_data

"""
Functions to work with image classification
"""
def get_categories(ddir):
    cats = []
    for x in os.listdir(ddir):
        if x[0]!='.':
            cats.append(x)

    return cats
def read_samples(ddir,cats):
    samples = {}
    c = 0
    for cat in cats:
        imgs_dir = os.path.join(ddir,cat)
        fs = glob.glob(os.path.join(imgs_dir,"*.*"))
        img = cv.imread(fs[0])
        samples[cat]=img
        c+=1
        if c>=10:
            break
    return samples

def show_image_data(ddir,cats=None,samples=None):
    if cats is None:
        cats = get_categories(ddir)
    
    if samples is None:
        samples = read_samples(ddir,cats)

    fig, axs = plt.subplots(1,len(samples),figsize=(20, 3))
    c = 0
    for cat in samples:
        axs[c].imshow(samples[cat])
        axs[c].set_title(cat)
        c+=1
    #fig.tight_layout()
    plt.show()

def get_metadata(ddir):
    cats = get_categories(ddir)
    samples = read_samples(ddir,cats)
    
    show_image_data(ddir,cats,samples)
    s = list(samples.values())[0]
    return cats,s.shape[0],s.shape[1],s.shape[2]


def load_and_resize(ddir,new_size=None):
    cats = get_categories(ddir)
    x = None
    y = []
    for cat in cats:
        imgs_dir = os.path.join(ddir,cat)
        fs = glob.glob(os.path.join(imgs_dir,"*.*"))
        print("Load images from %s ... "%(cat))
        for f in fs:
            im = cv.imread(f)
            if new_size is not None:
                im = cv.resize(im,(new_size,new_size))
            im = im[np.newaxis,:,:,:]
            if x is None:
                x = im
            else:
                x = np.append(x,im,axis=0)
            y.append(cat)
    return x,y


"""
Functions for plotting
"""

def plot2d(x_,y_):
    if x_.shape[1]!=2:
        print("Cannot plot data with dimension different from 2!!!!")
        return
    # creating a new data fram which help us in ploting the result data
    df = pd.DataFrame(data=np.vstack((x_.T, y_)).T, columns=("1st_principal", "2nd_principal", "label"))
    sn.FacetGrid(df, hue="label", size=6).map(plt.scatter, '1st_principal', '2nd_principal').add_legend()
    plt.show()

def plot_kmeans(kmeans,x_):
    if x_.shape[1]!=2:
        print("Cannot plot data with dimension different from 2!!!!")
        return

    h = .02
    centroids = kmeans.cluster_centers_
    plt_data = plt.scatter(x_[:, 0], x_[:, 1], c=kmeans.labels_, cmap=plt.cm.get_cmap('Spectral', 10))
    plt.colorbar()
    plt.scatter(centroids[:, 0], centroids[:, 1],marker='x')
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    labels = ["c"+str(i) for i in range(10)]
    for i in range (10):
        xy=(centroids[i, 0],centroids[i, 1])
        plt.annotate(labels[i],xy, horizontalalignment='right', verticalalignment='top')
    plt.show()
"""
Class and functions for MNIST
@SNT
"""
class MNIST(object):
    def __init__(self,dat_dir):
        self.mnist = mnist = input_data.read_data_sets(dat_dir, one_hot=True)
        self.x_train = mnist.train.images[:50000,:]
        self.y_train = np.argmax(mnist.train.labels[:50000,:],axis=1)
        self.x_val   = mnist.train.images[50000:,:]
        self.y_val   = np.argmax(mnist.train.labels[50000:,:],axis=1)
        self.x_test  = mnist.test.images[:50000,:]
        self.y_test  = np.argmax(mnist.test.labels[:50000,:],axis=1)
        
    
    def samples(self,batch_size=1):
        x,y = self.mnist.train.next_batch(batch_size)
        return x

    @property
    def dimension(self):
        return  784

    def show_samples(self,samples):
        n = math.floor(math.sqrt(samples.shape[0]))
        img = np.zeros((n*28,n*28))
        
        for i in range(n):
            for j in range(n):
                img[i*28:(i+1)*28,j*28:(j+1)*28] = np.reshape(samples[i*n+j],[28,28])

        #plt.show()
        #plt.savefig('./vis.png', bbox_inches='tight')
        #plt.close(fig)
        
        pl.imshow(img,cmap="gray")
        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(1.0)    
#get_categories("./grape_leaf_disease_detection/image_data/train")
#show_image_data("./grape_leaf_disease_detection/image_data/train")

"""
Ultilities for ML education
@SNT
"""
import numpy as np
import pandas as pd
import seaborn as sn

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

#import cv2 as cv

import glob
import os
import gzip
import pickle

from gccNMF.gccNMFFunctions import *
from gccNMF.gccNMFPlotting import *
#import tensorflow as tf

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

#mnist = MNIST(download=False)

#### temporal data
def prepare_data(data,features_inds,output_inx,N=5):
    x = None
    y = []
    CONTEXT_LEN = N
    for i in range(CONTEXT_LEN,data.shape[0]):
        features = []
        # features in previous CONTEXT_LEN days
        for t in range(1,CONTEXT_LEN):
            f = data[i-t,features_inds]
            features = np.append(features,f)
        if x is None:
            x = features[np.newaxis,:]
        else:
            x = np.append(x,features[np.newaxis,:],axis=0) 
        
        y.append(data[i,output_inx])
        
    return x, np.array(y)
##########gccnmf
def gccnmf(stereoSamples,sampleRate,numSources = 6):
    windowSize = 1024
    fftSize = windowSize
    hopSize = 128
    windowFunction = hanning

    # TDOA params
    numTDOAs = 128

    # NMF params
    dictionarySize = 128
    numIterations = 100
    sparsityAlpha = 0

    # Input params    
    microphoneSeparationInMetres = 1.0
    
    numChannels, numSamples = stereoSamples.shape
    durationInSeconds = numSamples / float(sampleRate)

    complexMixtureSpectrogram = computeComplexMixtureSpectrogram( stereoSamples, windowSize,
                                                                  hopSize, windowFunction ) 
    numChannels, numFrequencies, numTime = complexMixtureSpectrogram.shape
    frequenciesInHz = getFrequenciesInHz(sampleRate, numFrequencies)

    V = concatenate( abs(complexMixtureSpectrogram), axis=-1 )
    W, H = performKLNMF(V, dictionarySize, numIterations, sparsityAlpha)

    numChannels = stereoSamples.shape[0]
    stereoH = array( hsplit(H, numChannels) )
    spectralCoherenceV = complexMixtureSpectrogram[0] * complexMixtureSpectrogram[1].conj() \
                         / abs(complexMixtureSpectrogram[0]) / abs(complexMixtureSpectrogram[1])

    angularSpectrogram = getAngularSpectrogram( spectralCoherenceV, frequenciesInHz,
                                            microphoneSeparationInMetres, numTDOAs )
    meanAngularSpectrum = mean(angularSpectrogram, axis=-1) 
    targetTDOAIndexes = estimateTargetTDOAIndexesFromAngularSpectrum( meanAngularSpectrum,
                                                                  microphoneSeparationInMetres,
                                                                  numTDOAs, numSources)

    targetTDOAGCCNMFs = getTargetTDOAGCCNMFs( spectralCoherenceV, microphoneSeparationInMetres,
                                              numTDOAs, frequenciesInHz, targetTDOAIndexes, W,
                                              stereoH )
    targetCoefficientMasks = getTargetCoefficientMasks(targetTDOAGCCNMFs, numSources)

    targetSpectrogramEstimates = getTargetSpectrogramEstimates( targetCoefficientMasks,
                                                                complexMixtureSpectrogram, W,
                                                                stereoH )

    targetSignalEstimates = getTargetSignalEstimates( targetSpectrogramEstimates, windowSize,
                                                  hopSize, windowFunction )
    saveTargetSignalEstimates(targetSignalEstimates, sampleRate, "audio")
    
    return targetSignalEstimates

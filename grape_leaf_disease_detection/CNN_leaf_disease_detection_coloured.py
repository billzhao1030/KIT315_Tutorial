
# coding: utf-8

import tensorflow as tf
print("tensorflow version: ", tf.__version__)
from keras import layers,models,utils
from keras.models import Sequential
from keras.layers import Flatten, Activation
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pickle
import random

# model_name = 'cnn'
model_name = 'Alexnet'



fig_size = 256

dirtrain='/home/sntran/Projects/plant_disease_detection/cnn/Grapes_Leaves_Dataset_images/train'
# dirtrain='D:\\grape_leaves_DB\\train' #change to the path of the training dataset 

dirtest='/home/sntran/Projects/plant_disease_detection/cnn/Grapes_Leaves_Dataset_images/test'
# dirtest='D:\\grape_leaves_DB\\test' #change to the path of the testing dataset 
dir_save='/home/sntran/Projects/plant_disease_detection/'

#variable that has the name of the diseases in oerder of the class
categories=["Black_rot","Esca_(Black_Measles)","Healthy","Leaf_blight_(Isariopsis_Leaf_Spot)"]


# just to display an image

# for c in categories:
#     path=os.path.join(dirtrain,c)
#     for i in os.listdir(path):
#         img=cv2.imread(os.path.join(path,i))
#         #print(img_array.shape)
#         img_array=cv2.resize(img,(fig_size,fig_size)) 
#         plt.imshow(img_array)
#         plt.show()
#         break
#     break

#Save the resized data to a file

training_data = []
def create_training_data():
    count=[]
    for c in categories:
        path=os.path.join(dirtrain,c)#creating the path of each class (folder)
        class_num=categories.index(c)#label is equal to the position of the class in 'categories' variable
        c=0
        for i in os.listdir(path):
            c=c+1
            try:
                img_array=cv2.imread(os.path.join(path,i))#creating the path of each image
                img_array = cv2.resize(img_array, (fig_size, fig_size))
                training_data.append([img_array,class_num])
            except Exception as e:
                pass
        count.append(c)
    return count

count_train=create_training_data() #function called to extract images from the training folder


testing_data = []
def create_testing_data():
    count=[]
    for c in categories:
        path=os.path.join(dirtest,c)
        class_num=categories.index(c)
        c=0
        for i in os.listdir(path):
            c=c+1
            try:
                img_array=cv2.imread(os.path.join(path,i))
                #img_array=cv2.resize(img_array,(128,128))
                img_array = cv2.resize(img_array, (fig_size, fig_size))
                testing_data.append([img_array,class_num])
            except Exception as e:
                pass
        count.append(c)
    return count

count_test=create_testing_data() #function called to extract images from the testing folder


"""
print(len(training_data))
print(count_train)
print(len(testing_data))
print(count_test)
"""

#shuffling the dataset to avoid successive training on same class of images
random.shuffle(training_data)
random.shuffle(testing_data)



x_train = []
y_train = []
x_test = []
y_test = []


#separating the images and label for the model
for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
x_train=np.array(x_train).reshape(-1,fig_size,fig_size,3)
#reshaping -1 means that the it can be any value i.e. the original value which is the no. of images
#256x256 for the dimension of the image and 3 for the the layers Red Green and Blue (RGB)

# #displaying an image
# x=cv2.resize(training_data[0][0],(fig_size,fig_size))
# plt.imshow(x,cmap='gray')


#separating the images and label for evaluation

for features, label in testing_data:
    x_test.append(features)
    y_test.append(label)
x_test=np.array(x_test).reshape(-1,fig_size,fig_size,3)


#saving the constructed training dataset using pickle

# def save_training_data(x_train,y_train):
#     pickle_out=open("x_train_coloured.pickle","wb")
#     pickle.dump(x_train,pickle_out)
#     pickle_out.close()

#     pickle_out=open("y_train_coloured.pickle","wb")
#     pickle.dump(y_train,pickle_out)
#     pickle_out.close
# save_training_data(x_train,y_train)


# #saving the constructed testing dataset using pickle

# def save_testing_data(x_test,y_test):
#     pickle_out=open("x_test_coloured.pickle","wb")
#     pickle.dump(x_test,pickle_out)
#     pickle_out.close()

#     pickle_out=open("y_test_coloured.pickle","wb")
#     pickle.dump(y_test,pickle_out)
#     pickle_out.close()
# save_testing_data(x_test,y_test)


#example of loading data from picle file

# def load_data():
#     pickle_in=open("x_train_coloured.pickle","rb")
#     x_train=pickle.load(pickle_in)
#     return x_train
#once the pickle files are ready no need to process the images form folder again and again


#creating the neural network model

if model_name == 'cnn':   
    K.clear_session()
    model=Sequential() 
    
    #CNN
    model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=(fig_size,fig_size,3),activation='relu'))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    
    
    model.add(layers.MaxPool2D(pool_size=(8,8)))
    
    model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
    model.add(layers.Conv2D(32,(3,3),activation='relu'))
    
    model.add(layers.MaxPool2D(pool_size=(8,8)))
    
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(layers.Dense(fig_size,activation='relu'))
    model.add(layers.Dense(4,activation='softmax'))
    

elif model_name == 'Alexnet':
    model = models.Sequential([
        # layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(fig_size,fig_size,3)),
        # layers.BatchNormalization(),
        # layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        # layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
        # layers.BatchNormalization(),
        # layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        # layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        # layers.BatchNormalization(),
        # layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        # layers.BatchNormalization(),
        # layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        # layers.BatchNormalization(),
        # layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
        # layers.Flatten(),
        # layers.Dense(512, activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(fig_size, activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(4, activation='softmax')
###################
        layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(fig_size,fig_size,3)),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3)),
        layers.Conv2D(filters=256, kernel_size=(5,5) , strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3)),
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
        layers.BatchNormalization(),
        layers.MaxPool2D(pool_size=(3,3)),
        layers.Flatten(),
        # layers.Dense(512, activation='relu'),
        # layers.Dropout(0.5),
        layers.Dense(fig_size, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax')
])


model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])


#compiling the network using the following loss and optimizer
model.summary()
# from keras.utils.vis_utils import plot_model
# plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)


#converting the training label to categorical
y_train_cat=to_categorical(y_train,4) #4 in the no. of categories


# converting the training label to categorical

y_test_cat=to_categorical(y_test,4)


#fit the model i.e. training the model and batch size can be varies
cnn_model=model.fit(x_train,y_train_cat,batch_size=16,
          epochs=20,verbose=1,validation_split=0.15,shuffle=True)
# validating the model with 15% data after every epoch which is also shuffled after each epoch




#evaluating the saved model
# loss, acc = new_model.evaluate(x_test,y_test_cat, verbose=1)
loss, acc = model.evaluate(x_test,y_test_cat, verbose=1)

print("Colour Model's accuracy: {:5.2f}%".format(100*acc)," | Fig_size: ", fig_size, '*', fig_size)
# print("Fig_size: ", fig_size, ' * ', fig_size)



'''from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())'''


model_history = cnn_model.history

plt.figure()
plt.plot(model_history['acc'])
plt.plot(model_history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('accuracy')
plt.show()


plt.figure()
plt.plot(model_history['loss'])
plt.plot(model_history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'])
plt.savefig('loss')
plt.show()

path_save = dir_save + model_name+'/model_save/model_'+ acc.astype(str)
if not os.path.exists(path_save):
    print('not exists, makedir')
    os.makedirs(path_save)

#f = open("saving.csv".'w')
np.savetxt(path_save +'/training_acc_hist.txt', model_history['acc'],fmt='%s',delimiter=',')
# np.savetxt(path_save +'/model_summary.txt', 'model.summary')
with open(path_save + '/model_summary.txt', 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))
    
# #saving the trained model so that no need to fit again for next time
model.save(path_save+'/leaf_disease_coloured.h5')


# np.savetxt("alexnet/training_acc_hist.csv",model_history['acc'])
print(model_history['acc'])
# #'d' is the path of the image
# d='D:\\jupyter_test\\Grapes-Leaf-Disease-detection-master\\Grapes_Leaves_Dataset_images\\test\\Black_rot\\0e7726c0-a309-4194-b2e6-d0e33af39373___FAM_B.Rot 0530_final_masked.jpg'  
# # d='D:\\grape_leaves_DB\\0ad02171-f9d0-4d0f-bdbd-36ac7674fafc___FAM_B.Msls 4356_final_masked.jpg'
# img=cv2.imread(d)
# #uncomment the below line if the image is not 256x256 by default
# img_array=cv2.resize(img,(fig_size,fig_size)) 
# plt.imshow(img_array)


# #reshaping the image to make it compatible for the argument of predict function
# img=img_array.reshape(-1,fig_size,fig_size,3)


# #predicting the class of the image
# # predict_class=new_model.predict_classes(img)
# predict_class=model.predict_classes(img)

# #will print a no. of the class to which the leaf belongs
# print(predict_class)

# #using the predict class as the index for categories defined at the beginning to display the name
# categories[predict_class[0]]




#example of loading the saved model
# new_model=models.load_model("/home/sntran/Projects/plant_disease_detection/cnn/test/leaf_disease_coloured_95_23.h5")


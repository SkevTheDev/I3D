#Ranjana Koshy
#CPSE 710
#Face Liveness Detection using I3D network, on video frames (Replay-Attack dataset)
#TF2.0

#https://github.com/LossNAN/I3D-Tensorflow

from __future__ import print_function
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.optimizers import SGD

from PIL import Image
from InceptionI3d import InceptionI3d

import tensorflow as tf
import os
import numpy as np
import time

batch_size = 32
num_classes = 2 
epochs = 15
 
# input image dimensions 
img_rows, img_cols = 64, 64

#Reading the train images
tr_Images = []
tr_Labels = []
tr_Filenames = []

trainImagesDirectory = "./diffusedimages/Matlab100/frames20TrainingDiffused/"
print(trainImagesDirectory)
trainImages = os.listdir(trainImagesDirectory)
count=0
trainIndexFile = "./diffusedimages/frames20TrainingDiffused.txt";
for trainImage in trainImages:    
    fobj = open(trainIndexFile,"r")
    for line in fobj:            
        filename, label = line.split("\t") #the filename and label are read from the .txt file            
        lbl = int(label); #int object                
        if(trainImage == filename):
            fname = os.path.join(trainImagesDirectory, trainImage);                                  
            break;
    fobj.close();
    count=count+1
    #print("count = ", count)
    image = Image.open(fname, mode = 'r') #reading an image.
    image = np.array(image) #the 2-d array of integer pixel values    
    image = image/255.0     #the 2-d array of float pixel values (between 0 and 1)
    tr_Images.append(image) #adding the train image to list
    if (count % 20 == 0):
        #print("in if, count = ", count)
        #print("\n")
        tr_Labels.append(lbl)   #adding the train label to list
    tr_Filenames.append(fname); #adding the train image filename to list
    
n = len(trainImages)
print("\n" + "number of training images = " + str(n)); #7200
print("length of tr_Images = " + str(len(tr_Images))); #7200
print("length of tr_Labels = " + str(len(tr_Labels))); #7200

tr_Images =np.reshape(np.array(tr_Images), (360, 20, 64, 64, 3)) #360 sequences of 20 frames each
##tr_Labels = np.reshape(np.array(tr_Labels), (360, 20))
tr_Labels = np.array(tr_Labels)
tr_Filenames = np.reshape(np.array(tr_Filenames), (360, 20))

print("\n..tr_Images.shape = ", tr_Images.shape) #(360, 20, 64, 64, 3)
print("..tr_Labels.shape = ", tr_Labels.shape)   #(360,)
print("..tr_Filenames.shape = ", tr_Filenames.shape) #(360, 20)

(X_train, Y_train) = tr_Images, tr_Labels

#Reading the test images
test_Images = []
test_Labels = []
test_Filenames = []

testImagesDirectory = "./diffusedimages/Matlab100/frames20TestingDiffused/"
print("\n" + testImagesDirectory)
testImages = os.listdir(testImagesDirectory)

count1=0
testIndexFile = "./diffusedimages/frames20TestingDiffused.txt";
for testImage in testImages:    
    fobj = open(testIndexFile,"r")
    for line in fobj:            
        filename, label = line.split("\t") #the filename and label are read from the .txt file            
        lbl = int(label); #int object                
        if(testImage == filename):
            fname = os.path.join(testImagesDirectory, testImage);                                  
            break;
    fobj.close();  
    count1=count1+1
    #print("count1 = ", count1)
    image = Image.open(fname, mode = 'r') #reading an image.
    image = np.array(image)   #the 2-d array of integer pixel values    
    image = image/255.0       #the 2-d array of float pixel values (between 0 and 1)    
    test_Images.append(image) #adding the test image to list                        
    if (count1 % 20 == 0):
        #print("in if, count1 = ", count1)
        #print("\n")
        test_Labels.append(lbl)   #adding the test label to list 
    test_Filenames.append(fname) #adding the test image filename to list 

n = len(testImages)
print("\n" + "number of test images = " + str(n)); #9600
print("length of test_Images = " + str(len(test_Images))); #9600
print("length of test_Labels = " + str(len(test_Labels))); #9600

test_Images = np.reshape(np.array(test_Images), (480, 20, 64, 64, 3)) #480 sequences of 20 frames each
##test_Labels = np.reshape(np.array(test_Labels), (480, 20))
test_Labels = np.array(test_Labels)
test_Filenames = np.reshape(np.array(test_Filenames), (480, 20))

print("\n..test_Images.shape = ", test_Images.shape) #(480, 20, 64, 64, 3)
print("..test_Labels.shape = ", test_Labels.shape)   #(480,)
print("..test_Filenames.shape = ", test_Filenames.shape) #(480, 20)

(X_test, Y_test) = test_Images, test_Labels

###########For HTER measurement##################

#Reading the real test images
test_Images_real = []
test_Labels_real = []
test_Filenames_real = []

testImagesRealDirectory = "./diffusedimages/Matlab100/frames20TestingDiffused_real/"
print("\n" + testImagesRealDirectory)
testImages_real = os.listdir(testImagesRealDirectory)

count2=0
testIndexFile_real = "./diffusedimages/frames20TestingDiffused_real.txt";

for testImage_real in testImages_real:    
    fobj = open(testIndexFile_real,"r")
    for line in fobj:            
        filename, label = line.split("\t") #the filename and label are read from the .txt file            
        lbl = int(label); #int object                
        if(testImage_real == filename):
            fname = os.path.join(testImagesRealDirectory, testImage_real);                                  
            break;
    fobj.close();   
    count2=count2+1
    #print("count2 = ", count2)    
    image = Image.open(fname, mode = 'r') #reading an image.
    image = np.array(image)   #the 2-d array of integer pixel values    
    image = image/255.0       #the 2-d array of float pixel values (between 0 and 1)    
    test_Images_real.append(image) #adding the test image to list                        
    if (count2 % 20 == 0):
        #print("in if, count2 = ", count2)
        #print("\n")    
        test_Labels_real.append(lbl)   #adding the test label to list 
    test_Filenames_real.append(fname) #adding the test image filename to list 

n = len(testImages_real)
print("\n" + "number of test images = " + str(n)); #1600
print("length of test_Images_real = " + str(len(test_Images_real))); #1600
print("length of test_Labels_real = " + str(len(test_Labels_real))); #1600

test_Images_real = np.reshape(np.array(test_Images_real), (80, 20, 64, 64, 3)) #80 sequences of 20 frames each
##test_Labels_real = np.reshape(np.array(test_Labels_real), (80, 20))
test_Labels_real = np.array(test_Labels_real) 
test_Filenames_real = np.reshape(np.array(test_Filenames_real), (80, 20))

print("\n..test_Images_real.shape = ", test_Images_real.shape) #(80, 20, 64, 64, 3)
print("..test_Labels_real.shape = ", test_Labels_real.shape)   #(80, )
print("..test_Filenames_real.shape = ", test_Filenames_real.shape) #(80, 20)

(X_test_real, Y_test_real) = test_Images_real, test_Labels_real

#Reading the attack test images
test_Images_attack = []
test_Labels_attack = []
test_Filenames_attack = []

testImagesAttackDirectory = "./diffusedimages/Matlab100/frames20TestingDiffused_attack/"
print("\n" + testImagesAttackDirectory)
testImages_attack = os.listdir(testImagesAttackDirectory)

count3=0
testIndexFile_attack = "./diffusedimages/frames20TestingDiffused_attack.txt";

for testImage_attack in testImages_attack:    
    fobj = open(testIndexFile_attack,"r")
    for line in fobj:            
        filename, label = line.split("\t") #the filename and label are read from the .txt file            
        lbl = int(label); #int object                
        if(testImage_attack == filename):
            fname = os.path.join(testImagesAttackDirectory, testImage_attack);                                  
            break;
    fobj.close();   
    count3=count3+1
    #print("count3 = ", count3)   
    image = Image.open(fname, mode = 'r') #reading an image.
    image = np.array(image)   #the 2-d array of integer pixel values    
    image = image/255.0       #the 2-d array of float pixel values (between 0 and 1)    
    test_Images_attack.append(image) #adding the test image to list                        
    if (count3 % 20 == 0):
        #print("in if, count3 = ", count3)
        #print("\n")        
        test_Labels_attack.append(lbl)   #adding the test label to list 
    test_Filenames_attack.append(fname) #adding the test image filename to list 

n = len(testImages_attack)
print("\n" + "number of test images = " + str(n)); #8000
print("length of test_Images_attack = " + str(len(test_Images_attack))); #8000
print("length of test_Labels_attack = " + str(len(test_Labels_attack))); #8000

test_Images_attack = np.reshape(np.array(test_Images_attack), (400, 20, 64, 64, 3)) #400 sequences of 20 frames each
##test_Labels_attack = np.reshape(np.array(test_Labels_attack), (400, 20))
test_Labels_attack = np.array(test_Labels_attack)
test_Filenames_attack = np.reshape(np.array(test_Filenames_attack), (400, 20))

print("\n..test_Images_attack.shape = ", test_Images_attack.shape) #(400, 20, 64, 64, 3)
print("..test_Labels_attack.shape = ", test_Labels_attack.shape)   #(400, )
print("..test_Filenames_attack.shape = ", test_Filenames_attack.shape) #(400, 20)

(X_test_attack, Y_test_attack) = test_Images_attack, test_Labels_attack

#################################################

if tf.keras.backend.image_data_format() == 'channels_first': 
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 3, img_rows, img_cols) 
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 3, img_rows, img_cols) 
    X_test_real = X_test_real.reshape(X_test_real.shape[0], X_test_real.shape[1], 3, img_rows, img_cols)     
    X_test_attack = X_test_attack.reshape(X_test_attack.shape[0], X_test_attack.shape[1], 3, img_rows, img_cols)     
else:     
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], img_rows, img_cols, 3)             
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], img_rows, img_cols, 3)     
    X_test_real = X_test_real.reshape(X_test_real.shape[0], X_test_real.shape[1], img_rows, img_cols, 3)     
    X_test_attack = X_test_attack.reshape(X_test_attack.shape[0], X_test_attack.shape[1], img_rows, img_cols, 3)     

# convert class vectors to binary class matrices 
Y_train = tf.keras.utils.to_categorical(Y_train, num_classes) #(Y_train, num_classes) 
Y_test = tf.keras.utils.to_categorical(Y_test, num_classes)
Y_test_real = tf.keras.utils.to_categorical(Y_test_real, num_classes)
Y_test_attack = tf.keras.utils.to_categorical(Y_test_attack, num_classes)

print("\n" + "after keras.utils.to_categorical()") 

print ("number of training examples = " + str(X_train.shape[0])) #360
print ("number of test examples = " + str(X_test.shape[0])) #480
print ("X_train shape: " + str(X_train.shape)) #(360, 20, 64, 64, 3)
print ("Y_train shape: " + str(Y_train.shape)) #(360, 2)
print ("X_test shape: " + str(X_test.shape))   #(480, 20, 64, 64, 3)
print ("Y_test shape: " + str(Y_test.shape))   #(480, 2)
print ("X_test_real shape: " + str(X_test_real.shape))   #(80, 20, 64, 64, 3)
print ("Y_test_real shape: " + str(Y_test_real.shape))   #(80, 2)
print ("X_test_attack shape: " + str(X_test_attack.shape))   #(400, 20, 64, 64, 3)
print ("Y_test_attack shape: " + str(Y_test_attack.shape))   #(400, 2)

i3d = InceptionI3d()

model = i3d._build(True, 1.0)

print("\n" + "model.compile()") 
"""model.compile(loss=tf.keras.losses.mean_squared_error,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=['accuracy'])"""

model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

start_time = time.time()    
start_time1 = time.time()  
print("\n" + "model.fit()") 
model.fit(X_train, Y_train, 
          batch_size=batch_size, 
          epochs=epochs, 
          verbose=1, 
          validation_data=(X_test, Y_test))

end_time1 = time.time()   
time_elapsed = end_time1 - start_time1    
minutes = int(time_elapsed/60)
seconds = time_elapsed - (minutes * 60)
print("\n" + "time taken (training) = " + str(minutes) + "minutes " + str(round(seconds, 2)) + "seconds")
          
start_time2 = time.time()
print("\n" + "model.evaluate()") 
score = model.evaluate(X_test, Y_test, verbose=0) 
print('Test loss:', score[0]) 
print('Test accuracy:', score[1])

end_time2 = time.time()   
time_elapsed = end_time2 - start_time2    
minutes = int(time_elapsed/60)
seconds = time_elapsed - (minutes * 60)
print("\n" + "time taken (evaluation) = " + str(minutes) + "minutes " + str(round(seconds, 2)) + "seconds")

end_time = time.time()   
time_elapsed = end_time - start_time    
minutes = int(time_elapsed/60)
seconds = time_elapsed - (minutes * 60)
print("\n" + "time taken (training and evaluation) = " + str(minutes) + "minutes " + str(round(seconds, 2)) + "seconds")

############Computing HTER##################

print("\n" + "Computing HTER:")

print("\n" + "model.evaluate() -> real test images") 
score_real = model.evaluate(X_test_real, Y_test_real, verbose=0) 
print('Test loss (real):', score_real[0]) 
print('Test accuracy (real):', score_real[1])

print("\n" + "model.evaluate() -> attack test images") 
score_attack = model.evaluate(X_test_attack, Y_test_attack, verbose=0) 
print('Test loss (attack):', score_attack[0]) 
print('Test accuracy (attack):', score_attack[1])

FRR = 100 - (score_real[1] * 100)
FAR = 100 - (score_attack[1] * 100)
print("\nFRR = " + str(FRR) + ", FAR = " + str(FAR))
hter = (FRR + FAR)/2
print("\nhter = " + str(hter))
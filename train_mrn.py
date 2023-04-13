# Importing all the required libraries.
import argparse
import os
import cv2
from keras import models
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D,Add
from keras.layers import BatchNormalization, concatenate
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from keras import regularizers
import numpy as np
import numpy.random as rng
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

x_shape = 256
y_shape = 320
channels = 1

#Training Folders (All these folders need to be manually added to your working directory.)
train_folder = './fvc_2006/'
train_img = os.path.join(train_folder, 'train')
train_gt = os.path.join(train_folder, 'gtruth')
mask_folder = './fvc_2006/'
mask_gt = os.path.join(mask_folder, 'mln_output')

#Testing Folders (All these folders need to be manually added to your working directory.)
test_folder = './fvc_2006/'
model_folder = './model/'
test_img = os.path.join(test_folder, "test")
test_gt = os.path.join(test_folder, "gtruth")
test_visual = os.path.join(test_folder, 'Visual_predictions_4')
mask_test_gt = os.path.join(test_folder, '4_hourglass_mln_output')
model_weights=os.path.join(model_folder,"Weights")
loss_files=os.path.join(model_folder,"Loss_Files")
val_sample=os.path.join(model_folder,"Validation_Samples")
plot=os.path.join(model_folder,"Plot")


def load_data():

	'''
	This function is used to load ground truth, fingerprint image and mask image data.
	The images have been resized appropriately for MRN training.
	'''


	imagePath = train_img
	gtPath = train_gt
	maskPath = mask_gt

	# It is very important to define image, mask and ground truth file extension.
	imageExt = ".jpg"
	maskExt = ".png"
	gtExt = ".txt"

	files = []
	files = os.listdir(imagePath)

	images = []
	mask = []
	gt = []
	for file in files:
		filename = file.split('.')[0]
		imagefile = os.path.join(imagePath,file)
		maskfile = os.path.join(maskPath,filename+maskExt)
		gtfile = os.path.join(gtPath,filename+gtExt)
		if not(os.path.exists(imagefile)) or not(os.path.exists(maskfile)) or not(os.path.exists(gtfile)):
			continue

		im = cv2.imread(imagefile,0)
		original_shape1, original_shape2 = im.shape #shape1 is y
		im = cv2.resize(im, (y_shape,x_shape))
		im = im[:,:,np.newaxis]
		images.append(im)

		im = cv2.imread(maskfile,0)
		im = cv2.resize(im, (y_shape,x_shape))
		im = im[:,:,np.newaxis]
		mask.append(im)

		f = open(gtfile, 'r')
		y, x = map(float, f.readline().split())
		x = (x*x_shape)/original_shape2
		y = (y*y_shape)/original_shape1
		gt.append((x/x_shape,y/y_shape))

	x = np.array(images)
	y = np.array(gt)
	z = np.array(mask)
	X_train,X_test,Y_train,Y_test, Z_train, Z_test=train_test_split(x,y,z,test_size=0.001)
	return X_train, X_test, Y_train, Y_test, Z_train, Z_test


# Model Development

###########################################  Regressor  ####################################################
def Regressor(input_img, decoded):
	merg1 = concatenate([input_img, decoded], axis = 3)
	reg_conv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1", kernel_initializer = 'he_uniform')(merg1)
	reg_conv1_1 = BatchNormalization()(reg_conv1_1)
	reg_conv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same',  name = "block1_conv2", kernel_initializer = 'he_uniform')(reg_conv1_1)
	reg_conv1_2 = BatchNormalization()(reg_conv1_2)
	reg_pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(reg_conv1_2)

	reg_conv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1", kernel_initializer = 'he_uniform')(reg_pool1)
	reg_conv2_1 = BatchNormalization()(reg_conv2_1)
	reg_conv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2", kernel_initializer = 'he_uniform')(reg_conv2_1)
	reg_conv2_2 = BatchNormalization()(reg_conv2_2)
	reg_pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(reg_conv2_2)

	reg_conv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1", kernel_initializer = 'he_uniform')(reg_pool2)
	reg_conv3_1 = BatchNormalization()(reg_conv3_1)
	reg_conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2", kernel_initializer = 'he_uniform')(reg_conv3_1)
	reg_conv3_2 = BatchNormalization()(reg_conv3_2)
	reg_pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(reg_conv3_2)

	reg_flat = Flatten()(reg_pool3)
	fc1 = Dense(256, activation='relu', kernel_initializer = 'he_uniform')(reg_flat)
	fc2 = Dense(64, activation='relu', kernel_initializer = 'he_uniform')(fc1)
	fc3 = Dense(16, activation='relu', kernel_initializer = 'he_uniform')(fc2)
	fc4 = Dense(2, activation='sigmoid')(fc3)
	regress = Model([input_img, decoded], fc4, name = "Output_layer")
	return regress

#############################################################################################################

# Training Setup.

input_img = Input(shape = (x_shape, y_shape, channels))
ae_output = Input(shape = (x_shape, y_shape, channels))
reg = Regressor(input_img, ae_output)

output_img = reg([input_img, ae_output])
model = Model([input_img, ae_output], output_img)

model.summary()

losses = {
	"Output_layer": "mean_squared_error"
}

model.load_weights(model_weights + './pre_trained')
model.compile(optimizer = Adam(0.00005), loss= losses, metrics=['accuracy'])

gtPath = train_gt
train_files = os.listdir(train_img)
print("Data_splitting..")
X_train, X_test, Y_train, Y_test, Mask_train, Mask_test = load_data()

# Data Normalization.
X_train = np.asarray(X_train, np.float16)/255
X_test = np.asarray(X_test, np.float16)/255
Mask_train = np.asarray(Mask_train, np.float16)/255
Mask_test = np.asarray(Mask_test, np.float16)/255

saveModel = os.path.join(model_weights, 'trained_mrn.h5')
numEpochs = 100
batch_size = 8
num_batches = int(len(X_train)/batch_size)
print("Number of batches: %d\n" % num_batches)
loss=[]
acc=[]
epoch=0

while epoch <numEpochs :

  history=model.fit([X_train, Mask_train], {'Output_layer':Y_train}, batch_size=batch_size, epochs=1, shuffle=True, verbose=1)
  model.save_weights(saveModel, overwrite = True)

  # Loss curve.
  epoch=epoch+1
  print("EPOCH NO. : "+str(epoch)+"\n")
  loss.append(float(history.history['loss'][0]))
  loss_arr=np.asarray(loss)
  e=range(epoch)
  plt.plot(e,loss_arr)
  plt.xlabel('Number of Epochs')
  plt.ylabel('Training Loss')
  plt.savefig(os.path.join(plot,str(epoch)+'.png'))
  plt.close()
  loss1=np.asarray(loss)
  np.savetxt(os.path.join(loss_files,'Loss.txt'),loss1)

  #Sampling random images to see model performance.
  s=rng.randint(len(train_files))
  filename=train_files[s]
  print(filename)
  path = os.path.join(train_img,filename)
  mask_name = filename.split('.')[0]
  mask_path = os.path.join(mask_gt, mask_name + '.png')
  save_path = os.path.join(val_sample,filename)

  # Sampling random image and its mask.
  x_test = cv2.imread(path,0)
  x_test = cv2.resize(x_test, (y_shape,x_shape))
  x_test = x_test[:,:,np.newaxis]
  x_test = np.array([x_test])
  x_test = np.asarray(x_test, np.float16)/255

  mask_test = cv2.imread(mask_path,0)
  mask_test = cv2.resize(mask_test, (y_shape,x_shape))
  mask_test = mask_test[:,:,np.newaxis]
  mask_test = np.array([mask_test])
  mask_test = np.asarray(mask_test, np.float16)/255

  # Validating on unseen fingerprint images.
  y_test = model.predict([x_test, mask_test])
  print(y_test[0][0]*x_shape, y_test[0][1]*y_shape)
  x_test = cv2.imread(path,0)
  original_shape1, original_shape2 = x_test.shape #shape1 is y
  x_test = x_test[:,:,np.newaxis]
  x_test = np.array(x_test)

  name = filename.split('.')[0]
  gtfile = os.path.join(gtPath,name+".txt")
  f = open(gtfile, 'r')
  y, x = map(float, f.readline().split())

  cv2.circle(x_test,(int(((y_test[0][0]*original_shape2))),int(((y_test[0][1]*original_shape1)))),4,(0,0,255),-1)#black
  cv2.circle(x_test,(int(x),int(y)),4,(255,0,0),-1)
  cv2.imwrite(save_path,x_test)

if os.path.exists(test_img):
	files = os.listdir(test_img)
else:
	sys.exit("Invalid Path")

# Saving all the predictions of our trained MRN on the testing data.
for filename in files:
  i=0
  i+=1
  print(i)
  path = os.path.join(test_img,filename)
  mask_name = filename.split('.')[0]
  mask_path = os.path.join(mask_test_gt, mask_name + '.png')
  save_path = os.path.join(test_visual,filename)
  x_test = cv2.imread(path,0)
  x_test = cv2.resize(x_test, (y_shape,x_shape))
  x_test = x_test[:,:,np.newaxis]
  x_test = np.array([x_test])
  x_test = np.asarray(x_test, np.float16)/255
  Mask_test = cv2.imread(mask_path,0)
  Mask_test = cv2.resize(Mask_test, (y_shape,x_shape))
  Mask_test = Mask_test[:,:,np.newaxis]
  Mask_test = np.array([Mask_test])
  Mask_test = np.asarray(Mask_test, np.float16)/255

  y_test = model.predict([x_test, Mask_test]) #y_test[0][0][0] is x , y_test[0][0][1] is y
  x_test = cv2.imread(path,0)
  original_shape1,original_shape2 = x_test.shape
  x_test = x_test[:,:,np.newaxis]
  x_test = np.array(x_test)

  name = filename.split('.')[0]
  gtfile = os.path.join(test_gt,name+".txt")
  if(os.path.exists(gtfile)):
    f = open(gtfile, 'r')
  else:
    continue
  try:
    y, x = map(float, f.readline().split())
  except:
    print(filename, " has no core point")

  cv2.circle(x_test,(int(((y_test[0][0]*original_shape2))),int(((y_test[0][1]*original_shape1*255)/y_shape))),4,(0,0,255),-1)#black
  cv2.circle(x_test,(int(x),int(y)),4,(255,0,0),-1)#white
  ############################################# To save coordinates in files ########################
  f=open(os.path.join(os.path.join(test_folder,"Predicted_Core_Point"), name+".txt"),'w')
  f.write(str((y_test[0][1]*original_shape1))+" "+str((y_test[0][0]*original_shape2)))
  f.close()

  cv2.imwrite(save_path,x_test)
  ###################################################################################################

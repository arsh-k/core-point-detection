'''Import the libraries'''
import os
import cv2
from keras.layers.core import *
from keras.layers import  Input,Dense,Flatten,Dropout, Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose,ZeroPadding2D, Add
from keras.layers import BatchNormalization, concatenate
from keras.models import Model,Sequential,load_model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adadelta,RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
import numpy as np
import scipy
import numpy.random as rng
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf

'''Set Keras image format '''
K.set_image_data_format('channels_last')


# Model Development

###########################################  Encoder  ####################################################
def Encoder(input_img):

	Econv1_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "block1_conv1", kernel_initializer = 'he_uniform')(input_img)
	Econv1_1 = BatchNormalization()(Econv1_1)
	Econv1_2 = Conv2D(16, (3, 3), activation='relu', padding='same',  name = "block1_conv2", kernel_initializer = 'he_uniform')(Econv1_1)
	Econv1_2 = BatchNormalization()(Econv1_2)
	pool1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "block1_pool1")(Econv1_2)

	Econv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv1", kernel_initializer = 'he_uniform')(pool1)
	Econv2_1 = BatchNormalization()(Econv2_1)
	Econv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "block2_conv2", kernel_initializer = 'he_uniform')(Econv2_1)
	Econv2_2 = BatchNormalization()(Econv2_2)
	pool2= MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block2_pool1")(Econv2_2)

	Econv3_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv1", kernel_initializer = 'he_uniform')(pool2)
	Econv3_1 = BatchNormalization()(Econv3_1)
	Econv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "block3_conv2", kernel_initializer = 'he_uniform')(Econv3_1)
	Econv3_2 = BatchNormalization()(Econv3_2)
	pool3 = MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same', name = "block3_pool1")(Econv3_2)

	encoded = Model(input_img, [pool3, Econv1_2, Econv2_2, Econv3_2], name='mln_encoder')
	return encoded

#########################################  Bottleneck ##################################################
def neck(input_layer):

	Nconv = Conv2D(256, (3,3),padding = "same", name = "neck1", kernel_initializer = 'he_uniform')(input_layer)
	Nconv = BatchNormalization()(Nconv)
	Nconv = Conv2D(128, (3,3),padding = "same", name = "neck2", kernel_initializer = 'he_uniform')(Nconv)
	Nconv = BatchNormalization()(Nconv)

	neck_model = Model(input_layer, Nconv, name = 'mln_neck')
	return neck_model
#########################################  Hourglass ##################################################

def Hourglass(input_layer):

	conv_1 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_conv1", kernel_initializer = 'he_uniform')(input_layer)

	conv1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block1_conv1", kernel_initializer = 'he_uniform')(conv_1)
	conv1_1 = BatchNormalization()(conv1_1)
	conv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block1_conv2", kernel_initializer = 'he_uniform')(conv1_1)
	conv1_2 = BatchNormalization()(conv1_2)
	conv1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block1_conv3", kernel_initializer = 'he_uniform')(conv1_2)
	conv1_3 = BatchNormalization()(conv1_3)
	residual1 = Add(name = "hg_block1_add")([conv_1,conv1_3])

	pool1_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "hg_block1_pool1")(residual1) #56

	branch1_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_branch_block1_conv1", kernel_initializer = 'he_uniform')(residual1)
	branch1_1 = BatchNormalization()(branch1_1)
	branch1_2= Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_branch_block1_conv2", kernel_initializer = 'he_uniform')(branch1_1)
	branch1_2 = BatchNormalization()(branch1_2)
	branch1_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_branch_block1_conv3", kernel_initializer = 'he_uniform')(branch1_2)
	branch1_3 = BatchNormalization()(branch1_3)
	bresidual1 = Add(name = "hg_branch_block1_add")([residual1,branch1_3])

	conv2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block2_conv1", kernel_initializer = 'he_uniform')(pool1_1)
	conv2_1 = BatchNormalization()(conv2_1)
	conv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block2_conv2", kernel_initializer = 'he_uniform')(conv2_1)
	conv2_2 = BatchNormalization()(conv2_2)
	conv2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block2_conv3", kernel_initializer = 'he_uniform')(conv2_2)
	conv2_3 = BatchNormalization()(conv2_3)
	residual2 = Add( name = "hg_block2_add")([pool1_1,conv2_3])

	pool2_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "hg_block2_pool1")(residual2) #28

	branch2_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_branch_block2_conv1", kernel_initializer = 'he_uniform')(residual2)
	branch2_1 = BatchNormalization()(branch2_1)
	branch2_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_branch_block2_conv2", kernel_initializer = 'he_uniform')(branch2_1)
	branch2_2 = BatchNormalization()(branch2_2)
	branch2_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_branch_block2_conv3", kernel_initializer = 'he_uniform')(branch2_2)
	branch2_3 = BatchNormalization()(branch2_3)
	bresidual2 = Add(name = "hg_branch_block2_add")([residual2,branch2_3])

	conv3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block3_conv1", kernel_initializer = 'he_uniform')(pool2_1)
	conv3_1 = BatchNormalization()(conv3_1)
	conv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block3_conv2", kernel_initializer = 'he_uniform')(conv3_1)
	conv3_2 = BatchNormalization()(conv3_2)
	conv3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block3_conv3", kernel_initializer = 'he_uniform')(conv3_2)
	conv3_3 = BatchNormalization()(conv3_3)
	residual3 = Add(name = "hg_block3_add")([pool2_1,conv3_3])

	pool3_1 = MaxPooling2D(pool_size=(2, 2),strides=(2,2),padding='same', name = "hg_block3_pool1")(residual3) #14

	branch3_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_branch_block3_conv1", kernel_initializer = 'he_uniform')(residual3)
	branch3_1 = BatchNormalization()(branch3_1)
	branch3_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_branch_block3_conv2", kernel_initializer = 'he_uniform')(branch3_1)
	branch3_2 = BatchNormalization()(branch3_2)
	branch3_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_branch_block3_conv3", kernel_initializer = 'he_uniform')(branch3_2)
	branch3_3 = BatchNormalization()(branch3_3)
	bresidual3 = Add(name = "hg_branch_block3_add")([residual3,branch3_3])

	###########################BOTLLENECK######################################

	conv4_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block4_conv1", kernel_initializer = 'he_uniform')(pool3_1)
	conv4_1 = BatchNormalization()(conv4_1)
	conv4_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block4_conv2", kernel_initializer = 'he_uniform')(conv4_1)
	conv4_2 = BatchNormalization()(conv4_2)
	conv4_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block4_conv3", kernel_initializer = 'he_uniform')(conv4_2)
	conv4_3 = BatchNormalization()(conv4_3)
	residual4 = Add(name = "hg_block4_add")([pool3_1,conv4_3])


	conv5_1 = Conv2D(128, (1, 1), activation='relu', padding='same', name = "hg_block5_conv1", kernel_initializer = 'he_uniform')(residual4)
	conv5_1 = BatchNormalization()(conv5_1)
	conv5_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "hg_block5_conv2", kernel_initializer = 'he_uniform')(conv5_1)
	conv5_2 = BatchNormalization()(conv5_2)
	conv5_3 = Conv2D(256, (1, 1), activation='relu', padding='same', name = "hg_block5_conv3", kernel_initializer = 'he_uniform')(conv5_2)
	conv5_3 = BatchNormalization()(conv5_3)
	residual5 = Add(name = "hg_block5_add")([residual4,conv5_3])

	#############################################################################

	up1_1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name = "hg_up1", kernel_initializer = 'he_uniform')(residual5)
	up1_1 = BatchNormalization()(up1_1) #28
	add1 = Add(name = "hg_up1_add")([up1_1,bresidual3])

	uconv1_1 = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_upconv1_1", kernel_initializer = 'he_uniform')(add1)
	uconv1_1 = BatchNormalization()(uconv1_1)
	uconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same',name = "hg_upconv1_2", kernel_initializer = 'he_uniform')(uconv1_1)
	uconv1_2 = BatchNormalization()(uconv1_2)
	uconv1_3 = Conv2D(256, (1, 1), activation='relu', padding='same',name = "hg_upconv1_3", kernel_initializer = 'he_uniform')(uconv1_2)
	uconv1_3 = BatchNormalization()(uconv1_3)
	uresidual1 = Add(name = "hg_upblock1_add")([add1,uconv1_3])

	up2_1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name = "hg_up2", kernel_initializer = 'he_uniform')(uresidual1)
	up2_1 = BatchNormalization()(up2_1) #56
	add2 = Add()([up2_1,bresidual2])

	uconv2_1 = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_upconv2_1", kernel_initializer = 'he_uniform')(add2)
	uconv2_1 = BatchNormalization()(uconv2_1)
	uconv2_2 = Conv2D(128, (3, 3), activation='relu', padding='same',name = "hg_upconv2_2", kernel_initializer = 'he_uniform')(uconv2_1)
	uconv2_2 = BatchNormalization()(uconv2_2)
	uconv2_3 = Conv2D(256, (1, 1), activation='relu', padding='same',name = "hg_upconv2_3", kernel_initializer = 'he_uniform')(uconv2_2)
	uconv2_3 = BatchNormalization()(uconv2_3)
	uresidual2 = Add(name = "hg_upblock2")([add2,uconv2_3])

	up3_1 = Conv2DTranspose(256,(2,2),strides = (2,2), activation = 'relu', padding = 'same',name = "hg_up3", kernel_initializer = 'he_uniform')(uresidual2)
	up3_1 = BatchNormalization()(up3_1) #112
	add3 = Add()([up3_1,bresidual1])

	uconv3_1 = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_upconv3_1", kernel_initializer = 'he_uniform')(add3)
	uconv3_1 = BatchNormalization()(uconv3_1)
	uconv3_2 = Conv2D(128, (3, 3), activation='relu', padding='same',name = "hg_upconv3_2", kernel_initializer = 'he_uniform')(uconv3_1)
	uconv3_2 = BatchNormalization()(uconv3_2)
	uconv3_3 = Conv2D(256, (1, 1), activation='relu', padding='same',name = "hg_upconv3_3", kernel_initializer = 'he_uniform')(uconv3_2)
	uconv3_3 = BatchNormalization()(uconv3_3)
	uresidual3 = Add()([add3,uconv3_3])

	out_hg = Conv2D(128, (1, 1), activation='relu', padding='same',name = "hg_out", kernel_initializer = 'he_uniform')(uresidual3)
	Hg = Model(input_layer,out_hg)

	return Hg
##########################################  Decoder   ##################################################
def Decoder(inp):

	up1 = Conv2DTranspose(128,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_1", kernel_initializer = 'he_uniform')(inp[0])
	up1 = BatchNormalization()(up1)
	up1 = concatenate([up1, inp[3]], axis=3)
	Upconv1_1 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_1", kernel_initializer = 'he_uniform')(up1)
	Upconv1_1 = BatchNormalization()(Upconv1_1)
	Upconv1_2 = Conv2D(128, (3, 3), activation='relu', padding='same', name = "Upconv1_2", kernel_initializer = 'he_uniform')(Upconv1_1)
	Upconv1_2 = BatchNormalization()(Upconv1_2)

	up2 = Conv2DTranspose(64,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_2", kernel_initializer = 'he_uniform')(Upconv1_2)
	up2 = BatchNormalization()(up2)
	up2 = concatenate([up2, inp[2]], axis = 3)
	Upconv2_1 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_1", kernel_initializer = 'he_uniform')(up2)
	Upconv2_1 = BatchNormalization()(Upconv2_1)
	Upconv2_2 = Conv2D(64, (3, 3), activation='relu', padding='same', name = "Upconv2_2", kernel_initializer = 'he_uniform')(Upconv2_1)
	Upconv2_2 = BatchNormalization()(Upconv2_2)

	up3 = Conv2DTranspose(16,(3,3),strides = (2,2), activation = 'relu', padding = 'same', name = "upsample_3", kernel_initializer = 'he_uniform')(Upconv2_2)
	up3 = BatchNormalization()(up3)
	up3 = concatenate([up3, inp[1]], axis =3)
	Upconv3_1 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_1", kernel_initializer = 'he_uniform')(up3)
	Upconv3_1 = BatchNormalization()(Upconv3_1)
	Upconv3_2 = Conv2D(16, (3, 3), activation='relu', padding='same', name = "Upconv3_2", kernel_initializer = 'he_uniform')(Upconv3_1)
	Upconv3_2 = BatchNormalization()(Upconv3_2)

	decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same', name = "Output_layer")(Upconv3_2)
	convnet = Model(inp, decoded, name = 'Mask_output')
	return convnet
#########################################################################################################

##########################################'''Model Training and Initialization.'''####################################

# Need to change image shape according to the database.
original_x_shape = 400
original_y_shape = 560
x_shape = 256
y_shape = 320
channels = 1
input_img = Input(shape = (x_shape,y_shape,channels))
print(input_img)

# Encoder initialization.
encoded = Encoder(input_img)	#return encoded representation with intermediate layer Pool3(encoded), Econv1_3, Econv2_3,Econv3_3

# Decoder initialization.
HG_ = Input(shape = (int(x_shape/(2**3)),int(y_shape/(2**3)),128)) #Converted float to int
conv1_l = Input(shape = (x_shape,y_shape,16))
conv2_l = Input(shape = (int(x_shape/(2**1)),int(y_shape/(2**1)),64)) #Made change to int
conv3_l = Input(shape = (int(x_shape/(2**2)),int(y_shape/(2**2)),128)) #to int
decoded = Decoder( [HG_, conv1_l, conv2_l, conv3_l])

# Bottleneck initialization.
Neck_input = Input(shape = (int(x_shape/(2**3)), int(y_shape/(2**3)),128)) #to int
neck = neck(Neck_input)

# Hourglass initialization - number of hourglasses must be initialized based on requirement.
HG_input = Input(shape = (int(x_shape/ (2**3)), int(y_shape/(2**3)) ,128))
Hg_1 = Hourglass(HG_input)
Hg_2 = Hourglass(HG_input)
Hg_3 = Hourglass(HG_input)
Hg_4 = Hourglass(HG_input)

# Change hourglass setting according to number of hourglasses.
output_img = decoded([Hg_4(Hg_3(Hg_2(Hg_1(encoded(input_img)[0])))), encoded(input_img)[1], encoded(input_img)[2], encoded(input_img)[3]])
model= Model(input_img, output_img)
model.summary()
model.compile(optimizer = Adam(0.0005), loss='binary_crossentropy', metrics = ["accuracy"])
model.load_weights('./pretrained_mln.h5')

#########################################################################################################
name = os.listdir("./train/")

input_images = []
output_images = []

print("Loading images..")
count = 0
for i in name:
  i_split = i.split('.')
  j  = i_split[0] + '.bmp'
  if os.path.exists("./train"+i) and os.path.exists("./mask_gtruth"+ j):
    img_x = cv2.imread("./train"+i, 0)
    img_x = cv2.resize(img_x, (y_shape,x_shape), cv2.INTER_AREA)
    img_x = img_x[:,:,np.newaxis]
    input_images.append(img_x)
    img_y = cv2.imread("./mask_gtruth"+ j, 0)
    img_y = cv2.resize(img_y, (y_shape,x_shape))
    img_y = img_y[:,:,np.newaxis]
    output_images.append(img_y)

print("Data splitting..")
X_train,X_test,Y_train,Y_test=train_test_split(input_images,output_images,test_size=0.01)
del input_images
del output_images

X_train = np.asarray(X_train, np.float16)/255
X_test = np.asarray(X_test, np.float16)/255
Y_train = np.asarray(Y_train, np.float16)/255
Y_test = np.asarray(Y_test, np.float16)/255
saveModel = "./trained_mln.h5"
numEpochs = 150
batch_size = 8
num_batches = int(len(X_train)/batch_size)
print("Number of batches: %d\n" % num_batches)
saveDir = './mln/Stats/'
loss=[]
val_loss=[]
acc=[]
val_acc=[]
epoch=0

# Model Training.

number_of_epochs = numEpochs

while epoch <= number_of_epochs :

  history=model.fit(X_train, Y_train, batch_size=batch_size, epochs=1, validation_data=(X_test,Y_test), shuffle=True, verbose=1)

  # Loss and Accuracy Tracking.
  epoch=epoch+1
  print("EPOCH NO. : "+str(epoch)+"\n")
  loss.append(float(history.history['loss'][0]))
  val_loss.append(float(history.history['val_loss'][0]))
  acc.append(float(history.history['accuracy'][0]))
  val_acc.append(float(history.history['val_accuracy'][0]))
  loss_arr=np.asarray(loss)
  e=range(epoch)
  plt.plot(e,loss_arr)
  plt.xlabel('Number of Epochs')
  plt.ylabel('Training Loss')
  plt.savefig('./Model_exp/fvc_2006/Stats_4/Plot'+str(epoch)+'.png')
  plt.close()
  loss1=np.asarray(loss)
  val_loss1=np.asarray(val_loss)
  acc1=np.asarray(acc)
  val_acc1=np.asarray(val_acc)

  np.savetxt('./mln/Stats/Loss.txt',loss1)
  np.savetxt('./mln/Stats/Val_Loss.txt',val_loss1)
  np.savetxt('./mln/Stats/Acc.txt',acc1)
  np.savetxt('./mln/Stats/Val_Acc.txt',val_acc1)

  s=rng.randint(len(X_test))
  x_test=X_test[s,:,:,:]
  x_test=x_test.reshape(1,x_shape,y_shape,1)
  mask_img = model.predict(x_test)
  x_test = x_test.reshape(x_shape,y_shape)
  mask_img = mask_img.reshape(x_shape,y_shape)
  temp = np.zeros([x_shape,y_shape*2])
  temp[:,:y_shape] = x_test[:,:]+mask_img[:,:]
  temp[:,y_shape:y_shape*2] = x_test[:,:]
  temp = temp*255
  mask_img=mask_img*255
  cv2.imwrite('./mln/Validation_Samples/' + str(epoch+1) + ".bmp", temp)
  cv2.imwrite('./mln/Validation_Samples/' + str(epoch+1) + ".png", mask_img)

  model.save_weights(saveModel)
print("training Done.")

# MLN predictions for both train and test data.
test_img = "./data"
mask_visual = "./MLN_output"
files = os.listdir(test_img)
i=0
for file in files:
  i+=1
  print(i)
  path = os.path.join(test_img,file)
  x_test = cv2.imread(path,0)
  x_test = cv2.resize(x_test, (y_shape,x_shape), cv2.INTER_AREA)
  x_test = np.asarray(x_test, np.float16)/255
  print(x_test.shape)

  x_test=x_test.reshape(1,original_x_shape,original_y_shape,1)
  mask_img = model.predict(x_test)
  x_test = x_test.reshape(original_x_shape,original_y_shape)
  mask_img = mask_img.reshape(original_x_shape,original_y_shape)
  temp = np.zeros([original_x_shape,original_y_shape*2])
  temp[:,:original_y_shape] = x_test[:,:]+mask_img[:,:]
  temp[:,original_y_shape:original_y_shape*2] = x_test[:,:]
  temp = temp*255
  mask_img=mask_img*255
  filename = file.split('.')[0]
  cv2.imwrite(mask_visual + filename + ".png", mask_img)

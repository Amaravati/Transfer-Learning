import numpy as np
import os
import time
from vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten
from keras.layers import merge, Input
from keras.models import Model
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from keras import backend as K
K.set_image_dim_ordering('tf')
from decimal_to_bin_v5 import decbin, cnvrt, compl2_frac, compl2_int, bits2double, bits2double_real, decfrac, decint
from histplot import histplot
from keras.optimizers import SGD


img_path = 'elephant.jpg'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
print (x.shape)
x = np.expand_dims(x, axis=0)
print (x.shape)
x = preprocess_input(x)
print('Input image shape:', x.shape)

# Loading the training data
PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

img_data_list=[]

for dataset in data_dir_list:
	img_list=os.listdir(data_path+'/'+ dataset)
	print ('Loaded the images of dataset-'+'{}\n'.format(dataset))
	for img in img_list:
		img_path = data_path + '/'+ dataset + '/'+ img
		img = image.load_img(img_path, target_size=(224, 224))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = preprocess_input(x)
#		x = x/255
		print('Input image shape:', x.shape)
		img_data_list.append(x)

img_data = np.array(img_data_list)
#img_data = img_data.astype('float32')
print (img_data.shape)
img_data=np.rollaxis(img_data,1,0)
print (img_data.shape)
img_data=img_data[0]
print (img_data.shape)

# Define the number of classes
num_classes = 4
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:100]=0
labels[100:200]=1
labels[200:300]=2
labels[300:]=3


names = ['cats','dogs','horses','humans']

def classify_test(model, X_test,y_test):
    Y_pred = model.predict(X_test)
    y_out = np.argmax(Y_pred,axis=1)
    Y_real = np.argmax(y_test, axis=1)
    accuracy_test = np.mean(np.equal(y_out,Y_real))
    return accuracy_test

names = ['cats','dogs','horses','humans']

# convert class labels to on-hot encoding for labels1
#Y = np_utils.to_categorical(labels, num_classes)
# is the labels in one-hot encoded format
Y = np_utils.to_categorical(labels, num_classes)

#Shuffle the dataset
#x,y = shuffle(img_data,Y, random_state=2)
x,y = shuffle(img_data,Y, random_state=2)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

####################################################################################################################

#Training the feature extraction also

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

model.summary()

last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)
custom_vgg_model2 = Model(image_input, out)
print("Model before training is:\n")
custom_vgg_model2.summary()

accuracy_before_train=classify_test(custom_vgg_model2, x,y)

print( "The accuracy before training is {}" .format(accuracy_before_train*100) )

# freeze all the layers except the dense layers
for layer in custom_vgg_model2.layers[:-1]:
	layer.trainable = False

print("The trainable modelis:\n")

custom_vgg_model2.summary()

'''ada_delta optimizer'''

#custom_vgg_model2.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['accuracy'])

'''sgd optimizer'''


custom_vgg_model2.compile( loss='categorical_crossentropy', optimizer=SGD(lr=1e-3), metrics=['accuracy'] )
  
t=time.time()
#	t = now()
num_epoch = 10

#hist_fc3 = custom_vgg_model2.fit(X_train, y_train, batch_size=10, epochs=num_epoch, verbose=1, validation_data=(X_test, y_test))
hist_fc3 = custom_vgg_model2.fit(X_train, y_train, batch_size=10, epochs=num_epoch, verbose=1)
print('Training time: %s' % (t - time.time()))
(loss_tr, accuracy_tr) = custom_vgg_model2.evaluate(X_test, y_test, batch_size=10, verbose=1)


print("[INFO] loss_tr={:.4f}, accuracy_tr: {:.4f}%" .format(loss_tr,accuracy_tr * 100))

#saving weights with only last layer being trainable
'''weights for training the last FC layer'''

#fname="Test-weights-CNN_fc3.hdf5"

'''training last 2 FC layers_ada_delta'''
#fname="Test-weights-CNN_fc2.hdf5"

'''weight for stochastic gradient descent'''

fname="Test-weights-CNN_fc3_sgd.hdf5"

''' no training of last layer '''
#fname="Test-weights-CNN_nt.hdf5"

custom_vgg_model2.save_weights(fname,overwrite=True)

custom_vgg_model2.load_weights(fname)

accuracy_test=classify_test(custom_vgg_model2, X_test,y_test)

print( "The test set accuracy is {}" .format(accuracy_test*100) )

histplot(hist_fc3,num_epoch)


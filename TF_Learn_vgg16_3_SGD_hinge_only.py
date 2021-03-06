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

# used to find the classification accuracy before trainign with random weights

def classify_test(model, X_test,y_test):
    Y_pred = model.predict(X_test)
    y_out = np.argmax(Y_pred,axis=1)
    Y_real = np.argmax(y_test, axis=1)
    accuracy_test = np.mean(np.equal(y_out,Y_real))
    return accuracy_test

names = ['cats','dogs','horses','humans']


Y = np_utils.to_categorical(labels, num_classes)


x,y = shuffle(img_data,Y, random_state=2)


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

####################################################################################################################

#Training the feature extraction also

image_input = Input(shape=(224, 224, 3))

model = VGG16(input_tensor=image_input, include_top=True,weights='imagenet')

print("The original model of VGG-net is\n")

model.summary()

# extracting the last layer weights and 
last_layer = model.get_layer('fc2').output
out = Dense(num_classes, activation='softmax', name='output')(last_layer)


custom_vgg_model3 = Model(image_input, out)
print("Model before training is:\n")
custom_vgg_model3.summary()

accuracy_before_train=classify_test(custom_vgg_model3, x,y)

print( "The accuracy before training with random weights is {}" .format(accuracy_before_train*100) )

''' Testsing program for extracting the output of hidden layers'''

img2 = np.zeros([1,224,224,3])
img2[0,:,:,:]=img_data[0,:,:,:]

# model to generate fc3
test_model = Model(inputs=custom_vgg_model3.input, outputs=custom_vgg_model3.layers[-2].output)
fc3_output = test_model.predict(img2)

# output of fc2 layer
test_model_fc2 = Model(inputs=custom_vgg_model3.input, outputs=custom_vgg_model3.layers[-3].output)
inter_output_fc2 = test_model_fc2.predict(img2)


# output of final layer evaluation (Dense)
test_modelo = Model(inputs=custom_vgg_model3.input, outputs=custom_vgg_model3.layers[-1].output)
final_output = test_modelo.predict(img2)

#softmax evaluation

def softmax(x):
    out=np.zeros(x.shape)
    for i in range(x.shape[0]):
        x1=x[i,:]
        e_x = np.exp(x1 - np.max(x1))
        out[i,:] = e_x / e_x.sum()
    return out


def classify_v2(img, w_fc1, b_fc1, labels):
    inter_output1 = test_model.predict(img)
    l_out1 = np.dot(inter_output1 , w_fc1) + b_fc1
    print("The cross entropy is:{}".format(cross_entropy(l_out1,labels)))
    y_label = np.argmax(l_out1,axis=1)
    y_true = np.argmax(labels,axis=1)
    accuracy = np.equal(y_label, y_true)
    return accuracy, y_label, y_true


def cross_entropy(X,y):
    log_likelihood = np.zeros(X.shape)
    for i in range(X.shape[0]):
        log_likelihood[i,:] = y[i,:]*np.log(X[i,:])
    log_likelihood=np.nan_to_num(log_likelihood)
    log_likelihood[log_likelihood>1e3]=0.9e3
    log_likelihood[log_likelihood<-1e3]=-0.9e3
    loss = np.sum(log_likelihood)/(4*32)
    return loss

def loss_vec(z2,delta,y):
    loss=0
    score=z2[y]
    margins=np.maximum(0,z2-score+delta)
    margins[y]=0
    loss=np.sum(margins)
    return loss  

def loff_final(score,yn):
    [m,]=yn.shape
    loss=0
    for i in range(0,m):
        z2=score[i,:]
        y=yn[i]
        loss+=loss_vec(z2,1,y)
    return loss/len(yn)


def grad_update2(z2,y,index,num_class,num_samples):
    grad_wyi=0
    grad_wyin=np.zeros(num_class)
    grad_f1=np.zeros((num_samples,num_class))
    score=z2[y]
    for i in range(0,num_class):
        if(i==y):
            continue
        grad_wyi+=int((z2[i]-score+1)>0)
        grad_wyin[i]=int((z2[i]-score+1)>0)
        grad_f1[index,i]=grad_wyin[i]

    grad_f1[index,y]=-grad_wyi

    
    return grad_f1   
    
def grad_updatef(score, y, num_class, num_samples):
    grad_f = np.zeros((num_samples,num_class))
    for i in range(num_samples):
        grad_f+=grad_update2(score[i,:], y[i], i, num_class, num_samples)
    return grad_f

# custom_last layer
#%%
w_fc1_init= custom_vgg_model3.layers[-1].get_weights()[0]
b_fc1_init = custom_vgg_model3.layers[-1].get_weights()[1]

'''Initializing with random weights'''

#w_fc1 = np.random.rand(4096,4)
#b_fc1 = np.random.rand(4)

'''storing them in the form of a text file'''

#np.savetxt('./w_fc_store/w_fc2.txt',w_fc1)
#np.savetxt('./w_fc_store/b_fc2.txt',b_fc1)

'''loading them as arrays'''
w_fc2=np.loadtxt('./w_fc_store/w_fc2.txt')
b_fc2=np.loadtxt('./w_fc_store/b_fc2.txt')

#w_fc2 = w_fc1
#b_fc2 = b_fc1

num_epoch = 10

int_w=8
frac_w=8
scalar=1e-3

accuracy    = np.zeros(X_test.shape[0])
label_final = np.zeros(X_test.shape[0])
label_true = np.zeros(X_test.shape[0])
batchsize = 32
num_class=4

acc_list=[]

for ep in range(0, num_epoch):
    
    seed = np.arange(X_train.shape[0])
    np.random.shuffle(seed)
    xt = X_train[seed]
    yt = y_train[seed]

    for j in range(X_train.shape[0]//batchsize):
        
        k= j*batchsize
        l= (j+1)*batchsize
        num_examples = X_train.shape[0]//batchsize
        
        fc3_output1 = test_model.predict(xt[k:l])

        l_out1 = np.dot(fc3_output1 , w_fc2) + b_fc2

        ytn = np.argmax(yt[k:l], axis=1)
        
        grad = grad_updatef(l_out1, ytn, num_class, batchsize)

        g_out = grad

            
        dJ_dw2 = np.dot(fc3_output1.T, g_out)
        db = np.sum(g_out, axis=0, keepdims=True)
        w_fc2 = w_fc2 - (scalar)*dJ_dw2 
        b_fc2 = b_fc2 - (scalar)*db
        

            
    accuracy_ts, label_final_ts, label_true_ts =  classify_v2(X_test, w_fc2, b_fc2, y_test)   


    accuracy_final_ts = np.mean(accuracy_ts)
    acc_list.append(accuracy_final_ts)
        
    print( "The final test case accuracy is {} for {} epoch" .format(accuracy_final_ts * 100, ep+1) )


accuracy_tr, label_final_tr, label_true_tr =  classify_v2(X_train, w_fc2, b_fc2, y_train) 

accuracy_train = np.mean(accuracy_tr)

print( "The training case accuracy is {}" .format( accuracy_train*100) )
        
accuracy_overall = (accuracy_train+accuracy_final_ts)/2

print( "The overall accuracy is {}" .format( accuracy_overall*100) )





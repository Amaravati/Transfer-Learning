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
from out_val import out_val
from process_out import process_out
from dw_out1 import dw_out1
from dw_up import dw_up
from w_up  import w_up


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


custom_vgg_model3 = Model(image_input, out)
print("Model before training is:\n")
custom_vgg_model3.summary()

accuracy_before_train=classify_test(custom_vgg_model3, x,y)

print( "The accuracy before training is {}" .format(accuracy_before_train*100) )

''' Testsing program for extracting the output of hidden layers'''

img2 = np.zeros([1,224,224,3])
img2[0,:,:,:]=img_data[0,:,:,:]

# model to generate fc2
test_model = Model(inputs=custom_vgg_model3.input, outputs=custom_vgg_model3.layers[-2].output)
fc3_output = test_model.predict(img2)

# output of fc2 layer
test_model_fc2 = Model(inputs=custom_vgg_model3.input, outputs=custom_vgg_model3.layers[-3].output)
inter_output_fc2 = test_model_fc2.predict(img2)


# output of final layer evaluation
test_modelo = Model(inputs=custom_vgg_model3.input, outputs=custom_vgg_model3.layers[-1].output)
final_output = test_modelo.predict(img2)


def softmax(x):
    out=np.zeros(x.shape)
    for i in range(x.shape[0]):
        x1=x[i,:]
        e_x = np.exp(x1 - np.max(x1))
        out[i,:] = e_x / e_x.sum()
    return out


def classify(img, w_fc2, b_fc2, labels):
    inter_output1 = test_model.predict(img)
    l_out1 = np.dot(inter_output1 , w_fc2) + b_fc2
    lsm_out1 = softmax(l_out1)
    y_label = np.argmax(lsm_out1)
    y_true = np.argmax(labels)
    accuracy = np.equal(y_label, y_true)
    return accuracy, y_label, y_true


def classify_v1(img, w_fc1, b_fc1, labels):
    inter_output1 = test_model.predict(img)
    l_out1 = np.dot(inter_output1 , w_fc1) + b_fc1
    lsm_out1 = softmax(l_out1)
    print("The cross entropy is:{}".format(cross_entropy(lsm_out1,labels)))
    y_label = np.argmax(lsm_out1,axis=1)
    y_true = np.argmax(labels,axis=1)
    accuracy = np.equal(y_label, y_true)
    return accuracy, y_label, y_true

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
    loss = -np.sum(log_likelihood)/4
    return loss

def loss_vec(z2,delta,y):
    loss=0
    score=z2[y]
    margins=np.maximum(0,z2-score+delta)
#    print(margins)
    margins[y]=0
    loss=np.sum(margins)
    return loss  

def loff_final(score,yn):
    [m,]=yn.shape
    loss=0
    for i in range(0,m):
        z2=score[i,:]
        y=yn[i]
#        print("z2 value is {}".format(z2))
        loss+=loss_vec(z2,1,y)
#        print("Loss value is {}".format(loss))
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
w_fc1 = custom_vgg_model3.layers[-1].get_weights()[0]
b_fc1 = custom_vgg_model3.layers[-1].get_weights()[1]
w_fc1_init= w_fc1
b_fc1_init = b_fc1

w_fc1 = np.random.rand(4096,4)
b_fc1 = np.random.rand(4)
w_fc2 = w_fc1
b_fc2 = b_fc1

num_epoch = 1

int_w=8
frac_w=8
scalar=1e-3

accuracy    = np.zeros(X_test.shape[0])
label_final = np.zeros(X_test.shape[0])
label_true = np.zeros(X_test.shape[0])
batchsize = 32
reg= 1e-9
num_class=4
# code for gradient descent:
#predicting the output
# Load the training image to img2
acc_list=[]
for ep in range(0, num_epoch):
# predict the output for neural net except last layer
    
#    seed = np.arange(X_train.shape[0])
#    np.random.shuffle(seed)
#    xt = X_train[seed]
#    yt = y_train[seed]
    xt = X_train
    yt = y_train
    
    for j in range(1):
#    for j in range(X_train.shape[0]//batchsize):
        
        k= j*batchsize
        l= (j+1)*batchsize
        num_examples = X_train.shape[0]//batchsize
        
        fc3_output1 = test_model.predict(xt[k:l])
            #conversion back to FP (8,8)
# evaluate the output
        l_out1 = np.dot(fc3_output1 , w_fc2) + b_fc2
#softmax non-linearity
#        lsm_out1 = softmax(l_out1)
        
#output gradient
        ytn = np.argmax(yt[k:l], axis=1)
        
        grad = grad_updatef(l_out1, ytn, num_class, batchsize)

        g_out = grad
#        g_out = lsm_out1 - yt[k:l]
#        g_out/=num_examples
        
        # Needs to predict fc1 output as well
#        inter_output_fc2 = test_model_fc2.predict(xt[k:l])
            
        dJ_dw2 = np.dot(fc3_output1.T, g_out)
        db = np.sum(g_out, axis=0, keepdims=True)
        w_fc2 = w_fc2 - (scalar)*dJ_dw2 
        b_fc2 = b_fc2 - (scalar)*db
        
#    scalar=scalar/(1+num_epoch)
        

#    print("The iteration number is: {}".format(ep))
#    print("The cross entropy is:{}".format(cross_entropy(lsm_out1,yt[k:l])))
            
    accuracy, label_final, label_true =  classify_v2(X_test, w_fc2, b_fc2, y_test)   


    accuracy_final = np.mean(accuracy)
    acc_list.append(accuracy_final)
        
    print( "The final accuracy is {}" .format(accuracy_final * 100) )

    
# Saving the txt files for output of fc3, w_fc and bias values
def make_files(fc3_output1, w_fc2, ytn):
    loop_iter = fc3_output1.shape[0]
    ftest = open("./weight/m_files.txt","w")
    
    for i in range(loop_iter):
        ftest.write("fc3_o{}=open('./weight/fc3o{}.list','w')\n".format(i,i))
        
    ftest.write("\n")
    
    loop_wfc = w_fc2.shape[1]
    
    for i in range(loop_wfc):
        ftest.write("f_w2c{}=open('./weight/w2r{}.list','w')\n".format(i,i))
    ftest.write("\n")

    ftest.write("for i in range(len(w_fc2[:,0])):\n")
    for i in range(loop_iter):
        ftest.write("    fc3_o{}.write(str(decbin(fc3_output1[{},i],int_w-1,frac_w) ) )\n".format(i,i))    
        ftest.write("    fc3_o{}.write('\')\n".format(i))
       
    ftest.write("\n")
    
    ftest.write("for i in range(fc3_output1.shape[1]):\n")
    for i in range(loop_wfc):
        ftest.write("    f_w2c{}.write(str(decbin(w_fc2[i,{}],int_w-2,frac_w+4) ) )\n".format(i,i))
        ftest.write("    f_w2c{}.write('\')\n".format(i))
    
    ftest_arr = open("./weight/m_files1.txt","w")
    
    ftest_arr.write("fc3_ot=open('./weight/fc3ot.list','w')\n")
    
    ftest_arr.write("for i in range(fc3_output1.shape[0]):\n")
    ftest_arr.write("    for j in range(fc3_output1.shape[1]):\n")
    ftest_arr.write("        fc3_ot.write(str(decbin(fc3_output1[{},{}],int_w-1,frac_w)))")
    ftest_arr.write("        \n")
    ftest_arr.write("    fc3_ot.write('\')\n")
    ftest_arr.write("\n")
    
    ftest_arr.write("    y_o=open('./weight/ytn.list','w')\n")
    ftest_arr.write("    for i in range(ytn.shape[0]):\n")
    ftest_arr.write("        y_o.write(str(decbin(ytn[i], 3,0)))")
    ftest_arr.write("        y_o.write(\)")

    ftest_arr.close()
    ftest.close()

make_files(fc3_output1, w_fc2, ytn)


def cnvrt_v1(w_fc2, fc3_output1):
    w_fc2n = np.zeros([4096,1])
    w_fc2n[:,0] = w_fc2[:,0]

    int_w=6
    frac_w=6
    
    w_fc2new = cnvrt(w_fc2n, int_w-2, frac_w+4)
    
    print("The maximun and minimum values of the w_fc2new are {} and {}" .format(np.max(np.max(w_fc2new)), np.min(np.min(w_fc2new))))

    
    fc3_output1n = np.zeros([1, 4096])
    
    fc3_output1n[0,:] = fc3_output1[0,:]
    
    fc3_output1new = cnvrt(fc3_output1n, int_w-1, frac_w)
    
    print("The maximun and minimum values of the fc3_out are {} and {}" .format(np.max(np.max(fc3_output1new)), np.min(np.min(fc3_output1new))))

    
    return w_fc2new, fc3_output1new

w_fc2n, fc3_output1n = cnvrt_v1(w_fc2, fc3_output1)
test = np.dot(fc3_output1n,w_fc2n)
test1=1120.90
bini=decbin(test,12,4)
out_quant= bits2double_real(bini,12,4)

print("The original, binary and new values are{}, {} and {}".format(test, bini,out_quant))
 

cnvrt_real = np.dot(fc3_output1,w_fc2)

cnvrt_arr = out_val()

diff_val = cnvrt_real - cnvrt_arr

grad_v1 = grad_updatef(cnvrt_real, ytn, num_class, batchsize)

grad_v2 = grad_updatef(cnvrt_arr, ytn, num_class, batchsize)

grad_rtl=process_out()

rtl_grad_diff = grad_v2 - grad_rtl

dw_out = np.dot(fc3_output1.T, grad_v2)

dw_out_rtl, o0 = dw_out1()

diff_rtl = dw_out - dw_out_rtl

dw_out_1024 = dw_out/2**10
dw_out_1024n = cnvrt(dw_out_1024,2,12)
dw_out_1024_rtl = dw_up()
 
w_fc2new = cnvrt(w_fc2,2,12) - cnvrt(dw_out_1024n,2,12)

w_fc2_rtl=w_up()

    


#
#
#with open('w2r1.list','r') as w:
#    w_contents=w.readlines()
##    print(w_contents)
#
#
#with open('fc3_out.list','r') as fc3:
#    fc3_contents= fc3.readlines()



        





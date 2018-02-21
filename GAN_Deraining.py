from code2 import *
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2


images = []
for i in glob.glob("*.jpg"):
    images.append(cv2.imread(i))

co = 0
rain_im = []
derain_im = []
for i in images:
    co += 1
    col = i.shape[1]
    im = i[:,:col//2,:]
    im2 = i[:,col//2:,:]
    derain_im.append(im)
    rain_im.append(im2)

derain_final = []
for i in derain_im:
    x = cv2.resize(i,(256,256))
    derain_final.append(x)

rain_final = []
for i in rain_im:
    x = cv2.resize(i,(256,256))
    rain_final.append(x)




#with open('rainWale.pkl', 'rb') as f:
#    loadedImages = pickle.load(f)
#Ximgs=np.array(loadedImages)

#with open('binaRainWale.pkl', 'rb') as f:
#    outImages = pickle.load(f)
#Yimgs=np.array(outImages)



#This is the defines the generator, with input as Ximgs(image with rain)

G=Generator()
X=tf.placeholder(tf.float32,shape=[None,256,256,3])
Y=tf.placeholder(tf.float32,shape=[None,256,256,3])
# X=tf.reshape(Ximgs,[-1,256,256,3])
conv1Out=G.addConvLayer(X,3,64)
conv2Out=G.addConvLayer(conv1Out,64,64)
conv3Out=G.addConvLayer(conv2Out,64,64)
conv4Out=G.addConvLayer(conv3Out,64,64)
conv5Out=G.addConvLayer(conv4Out,64,32)
conv6Out=G.addConvLayer(conv5Out,32,1)
deconv1Out=G.addDeConvLayer(conv6Out,1,32)
deconv2Out=G.addDeConvLayer(deconv1Out,32,64) + conv4Out
deconv3Out=G.addDeConvLayer(deconv2Out,64,64)
deconv4Out=G.addDeConvLayer(deconv3Out,64,64) + conv2Out
deconv5Out=G.addDeConvLayer(deconv4Out,64,64)
GX=G.addDeConvLayer(deconv5Out,64,3) + X

##########################################################


#This defines the discriminator, with input as Y(original image without rain)

D=Generator()
Dconv1Out=D.addConvLayer(Y,3,48,strideX=2, strideY=2,filterSize=4,BN=False)
Dconv2Out=D.addConvLayer(Dconv1Out,48,96,strideX=2, strideY=2,filterSize=4)
Dconv3Out=D.addConvLayer(Dconv2Out,96,192,strideX=2, strideY=2,filterSize=4)
Dconv4Out=D.addConvLayer(Dconv3Out,192,384,strideX=1, strideY=1,filterSize=4)
Dconv5Out=D.addConvLayer(Dconv4Out,384,1,strideX=1, strideY=1,filterSize=4,BN=False,PRelu=False)
DY = D.addDeepNet(Dconv5Out)

Le = tf.reduce_mean(tf.squared_difference(GX,Y))
La_Y = -tf.reduce_mean(tf.log(DY))
DX_ = D.forward(GX)
La_X_ = -tf.reduce_mean(tf.log(1-DX_))
La = La_Y + La_X_

# for i in range(100):



#Le = tf.reduce_mean(tf.squared_difference(GX,Y))
#La = 

#finding D of original image
# D.setInput(Y)
# Dori = tf.nn.sigmoid(D.forward(Y))
# Lp = (tf.reduce_mean(tf.squared_difference(DY,Dori)))
L = 0.0066 * La + Le 

solver =  tf.train.AdamOptimizer().minimize(L)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())


for i in range(2):
	print("idhar pohoch gaya ",i)
	finalLoss=sess.run(solver,feed_dict={X:rain_final[:5],Y:derain_final[:5]})
	print(finalLoss)

# img=sess.run(genImg,feed_dict={X:Ximgs[:5]})

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 09:04:22 2018

@author: anvesh
"""
import matplotlib.pyplot as plt



def histplot (hist,num_epoch):
    train_loss=hist.history['loss']
#    val_loss=hist.history['val_loss']
#    train_acc=hist.history['acc']
#    val_acc=hist.history['val_acc']
    xc=range(num_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss,'go-',linewidth=4)
#    plt.plot(xc,train_loss,'bo',linewidth=1)

#    plt.plot(xc,train_loss,'g^')
#    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs epochs')
    plt.grid(True)
    plt.legend(['train'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    plt.style.use(['classic'])
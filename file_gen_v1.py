#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:30:47 2018

@author: anvesh
"""
import numpy as np
int_w=6
frac_w=6
from decimal_to_bin_v5 import decbin, cnvrt, compl2_frac, compl2_int, bits2double, bits2double_real, decfrac, decint
#used for text file generation to run the code

'''
usage save_filesval(fc3_output1, w_fc2, ytn, 6, 6)
'''

def save_filesval(fc3_output1, w_fc2, ytn, int_w, frac_w):
    
   
    f_w2c0=open('./weight/w2r0.list','w')
    f_w2c1=open('./weight/w2r1.list','w')
    f_w2c2=open('./weight/w2r2.list','w')
    f_w2c3=open('./weight/w2r3.list','w')

    for i in range(fc3_output1.shape[1]):
        f_w2c0.write(str(decbin(w_fc2[i,0],int_w-4,frac_w+6) ) )
        f_w2c0.write("\n")
        f_w2c1.write(str(decbin(w_fc2[i,1],int_w-4,frac_w+6) ) )
        f_w2c1.write("\n")
        f_w2c2.write(str(decbin(w_fc2[i,2],int_w-4,frac_w+6) ) )
        f_w2c2.write("\n")
        f_w2c3.write(str(decbin(w_fc2[i,3],int_w-4,frac_w+6) ) )
        f_w2c3.write("\n")
        
        
#    fc3_ot=open("./weight_file/fc3_ot_1.txt","w")
    fc3_ot=open("./weight/fc3_ot_1.txt","w")

    
    for i in range(fc3_output1.shape[0]):
        for j in range(fc3_output1.shape[1]):
            fc3_ot.write(str(decbin(fc3_output1[i,j],int_w-1,frac_w) ))        
        fc3_ot.write("\n")

    fc3_ot.close()
    
    y_o=open('./weight/ytn.list','w')
    for i in range(ytn.shape[0]):
        y_o.write(str(decbin(ytn[i], 3,0)))
        y_o.write("\n")


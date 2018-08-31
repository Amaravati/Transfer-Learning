#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 19:int_w:02 2018

@author: anvesh
"""
import numpy as np
from decimal_to_bin_v5 import decbin, cnvrt, compl2_frac, compl2_int, bits2double, bits2double_real, decfrac, decint


def process_out():
    int_w=8;
    frac_w=0;
    with open('./out_file_v1/gv0_out.txt','r') as f:
        o0_contents=f.readlines()
  
    with open('./out_file_v1/gv1_out.txt','r') as f:
        o1_contents=f.readlines()
    
    with open('./out_file_v1/gv2_out.txt','r') as f:
        o2_contents=f.readlines()
      
    with open('./out_file_v1/gv3_out.txt','r') as f:
        o3_contents=f.readlines()
    
    cnvrt_o0=np.zeros(len(o0_contents))
    cnvrt_o1=np.zeros(len(o1_contents))      
    cnvrt_o2=np.zeros(len(o2_contents))      
    cnvrt_o3=np.zeros(len(o3_contents))      
          
    for i in range(len(o0_contents)):
        a=o0_contents[i]
        b=o1_contents[i]
        c=o2_contents[i]
        d=o3_contents[i]    
        a=a[0:-1]
        b=b[0:-1]
        c=c[0:-1]
        d=d[0:-1]    
        
        cnvrt_o0[i]=bits2double_real(a,int_w,frac_w)
        cnvrt_o1[i]=bits2double_real(b,int_w,frac_w)
        cnvrt_o2[i]=bits2double_real(c,int_w,frac_w)
        cnvrt_o3[i]=bits2double_real(d,int_w,frac_w)
        
    cnvrt_arr = np.array([cnvrt_o0,cnvrt_o1,cnvrt_o2,cnvrt_o3])
    cnvrt_arr = cnvrt_arr.T
    
    return cnvrt_arr
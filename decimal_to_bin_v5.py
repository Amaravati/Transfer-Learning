import numpy as np
#tailPrecision=0;
#bitPrecision=8;

class Int(int):
	def __add__(self, other):
		return "%s%s"%(self, other)

# bits2double() converts the binary input to floating point variable
def bits2double(num,bitPrecision,tailPrecision):
    result = float(int(num, 2));
    result = float(result) * (2** ((-1)*tailPrecision));
    if(num[0] == "1"):
        result = result - (2 ** (bitPrecision-tailPrecision)) ;
    return result;

def bits2double_real(num,int_w,frac_w):
    word=0
    if(num[0]=="0"):
        for i in range(1,int_w):
            word+=(int(num[i])*(2**(int_w-i-1)))
#            print("The word value is {}".format(word))
        for i in range(1,frac_w):
            word+=(int(num[int_w+i-1])*(2**-i))
#            print("The word value is {}".format(word))
    else:
        word=-(2**(int_w-1))
        for j in range(1,int_w):
            word+=(int(num[j])*(2**(int_w-j-1)))
        for j in range(1,frac_w):
            word+=(int(num[int_w+j-1])*(2**-j))
    return word

# decfrac() converts the decimal to binary
def decfrac(num,frac_width):
    cnvrt="";
    l=[]
    for i in range(frac_width):
        num=num*2;
        num1=int(num);
#        print("num value is {}".format(num))
        if(num1==1):
            cnvrt+=str(1);
            num=num-1;
        else:
            cnvrt+=str(0);
    l=cnvrt
    return l

def decint(num,int_width):
    cnvrt=""
    cnvrt_final=""
    for i in range(int_width):
        rem=num%2
        num=num//2
        if(rem==1):
            cnvrt+=str(1)
        else:
            cnvrt+=str(0)
    cnvrt_final=cnvrt[::-1]
    return cnvrt_final


def compl2_int(int_bin):
    int_bin_fl=""
    for i in range(len(int_bin)):
        if int_bin[i]=='0':
            int_bin_fl+=str(1)           
        else:
            int_bin_fl+=str(0)
    return int_bin_fl


def compl2_frac(frac_bin):
    frac_bin_fl=""
    for i in range(len(frac_bin)-1):
        if frac_bin[i]=='0':
            frac_bin_fl+=str(1)           
        else:
            frac_bin_fl+=str(0)
    
    if frac_bin[len(frac_bin)-1]=='0':
        frac_bin_fl+=str(0)
    else:
        frac_bin_fl+=str(1)
    return frac_bin_fl
            
            

def decbin(num,int_width,frac_width):
    int_part=int(num)
    frac_part=abs(num-int_part)
    int_init=""
    #initial bit assignmrnt based on the sign bit
    if str(int_part)[0]=='-':
        int_init+=str(1)
        int_part1=int(str(int_part)[1:])
    else:
        int_init+=str(0)
        int_part1=int(str(int_part)[:])
    #isolating the sign bit
#    int_part1=int(str(int_part)[1:])
#    print("Integer and fractional prts are: {} and {}".format(int_part1,frac_part))
    #decimal to binary conversion of integer part
    int_bin=decint(int_part1,int_width-1)
#    print("Integer binary value is {}".format(int_bin))
    #decimal to integer conversion of fractional part
    frac_bin=decfrac(frac_part,frac_width)
    total_bin=""
    frac_bin_fl=""
    if str(int_part)[0]=='-' or np.signbit(num)==True:
        int_init=str(1)
        int_bin_fl=int_init+compl2_int(int_bin)
        frac_bin_fl=compl2_frac(frac_bin)
#        frac_bin_fl=compl2(frac_bin)
    else:
        int_bin_fl=int_init+int_bin
        frac_bin_fl=frac_bin
#    print("The integer and fractional parts are:{} and {}".format(int_bin_fl,frac_bin_fl))
    total_bin=int_bin_fl+frac_bin_fl
    return total_bin


precision=6
    
def cnvrt(arr,int_w,frac_w):
    vec2=np.zeros(arr.shape)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            test=np.round(arr[i,j],6)
            bini=decbin(test,int_w,frac_w)
            vec2[i,j]= bits2double_real(bini,int_w,frac_w)
#        print("The value of ith vector variable is {}".format(vec2[i]))
    return vec2



# conversion tests
#nums=np.array([[ 3.456, -16.54, -24.56, 30.245, -15.1, -8.3, -20.9]])
#vec2=cnvrt(nums,6,6)
#print(vec2)    

'''
def convertToBinary(n):
   """Function to print binary number
   for the input decimal using recursion"""
   if n > 1:
       convertToBinary(n//2)
#   return n%2    
   print(n % 2,end = '')

'''    
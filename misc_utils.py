import numpy as np
import re

#  array2string(a):
#  array2string(array):
#  SieveOfSundaram(n):
#  square_compliments(N):




#TODO:  generalize so that arrays of arrays becomes lists of lists

# (N,) shape int array becomes list of length 1 (all ints concatenated along axis=1 (into single string))
# (N,1) shape int array becomes list of length N (no concatenation) [PREFERRED]
# (N,2) shape int array becomes list of length N (ints concatenated along axis=1)

def array2string(a):
    a_str = np.array_str(np.expand_dims(a,1))
    return re.sub('|'.join(['\[','\]',' ']), '', a_str).split('\n')

# internally converts input array from shape (N,M) to (N,M+1)
# behavior then follows above description
def array2string(array):
    if array.ndim==1:
        array = np.expand_dims(array,1)
        
    array_str = np.array_str(array)
    return re.sub('|'.join(['\[','\]',' ']), '', array_str).split('\n')

######################  MATH  ###############################

def SieveOfSundaram(n):
    
    nNew = int((n -1 ) / 2)
    marked = [0] * (nNew + 1)
    
    for i in range(1, nNew +1):
        j = i
        while (i + j + 2 * i * j) <= nNew:
            marked[i + j + 2 * i * j] = 1
            j += 1
            
    if n > 2:
        print(2, end = " ")
        
        for i in range(1, nNew + 1):
            if marked[i] == 0:
                print((2 * i + 1), end = " ")
                
get_primes = SieveOfSundaram


def square_compliments(N):
	s = np.arange(N, dtype=np.int)
	s2 = s*s
	a = np.c_[s.T,s2.T]
	a_str = np.array_str(a)
	a_str_list = re.sub('|'.join(['\[','\]',' ']), '', a_str).split('\n')
	A = np.array([len(set(x))/len(list(x)) for x in a_str_list])
	ix = np.argwhere(A.astype(np.int))
	return a[ix[:,0]]

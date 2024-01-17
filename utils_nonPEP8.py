import json
import lxml
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import glob
import re
from scipy import signal
import psutil
import datetime

# for create_a_gif
import imageio
import glob
import IPython

plt.style.use('seaborn')

#  arguments():    #credit to Kelly Yancey (kybyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html)
#  array_info(array, print_info=True, return_info=False, return_info_str=False):
#  autoscale_array(array):
#  autoscale_arrays(A, B):
#  bit_blast(array):
#  bit_hot(array):
#  bits2U8(bit_hot_array):
#  blend(a1,a2,a3=None):
#  bool_hot(array):
#  change_extension(filename, ext):
#  char_to_one_hot(char, minchar=0, maxchar=128):
#  cos1D(pix=512, start=0., stop=2*np.pi, amplitude=1., freq=1., offset=0., phase=0.):
#  cos2D(pix_x=512, start_x=0., stop_x=2*np.pi, freq_x=1., pix_y=512, start_y=0., stop_y=2*np.pi, freq_y=1, offset_z=0.):
#  create_gif(filename_regex, anim_file='new.gif'):
#  cyclic_diff(x,axis=0):
#  ddir(module, x=None, print_only=False):
#  df_get(df,column_name,value):
#  diagonal_slice(N,M, start=0, stop=-1, step=1):
#  diagonal_unwrap(a):
#  diagonal_wrap(x, num_cols=1, clip=False):
#  diagrid(N,M):
#  diff(x, axis=0):
#  encodeDF(df):
#  entropy(array, normalize=True, nbins=100):
#  entropy(array, normalize=True, nbins=100):  #modified 20211115 0220: normalize frequencies
#  find(key, recursive=False):
#  flatten_by_cols(x):
#  flatten_by_rows(x):
#  generate_and_save_images(model, epoch, test_input):
#  get_iso8601(date_and_time):
#  get_iso8601_offset(date_and_time, count, granularity):
#  hex_remove(h, s):
#  hex2string(h):
#  image_distribution(P, bins=255, lo=0, hi=255):
#  image_entropy(array, bins=255, lo=0, hi=255):
#  image_KL(P,Q, bins=255, lo=0, hi=255):
#  image_KL(P,Q, bins=255, lo=0, hi=255):
#  image_xentropy(p, q, bins=255, lo=0, hi=255):
#  invert_as_sparse(a):
#  iseven(x):
#  isnotebook():
#  isodd(x):
#  list_imported_modules():
#  list_imported_modules_(g):
#  log_memory(ref_id):
#  mkpath(path):
#  multiplot(index, list_of_arrays):
#  mv(fpath_from, fpath_to):
#  normalize_array(array):
#  np_fftconvolve(A, B):
#  pad2square(array_1d, dtype=np.int8):
#  path2tuple(path):
#  peek(dictionary, n=3, delim='|'):
#  peek2(dictionary, delim='|'):
#  print_array(array):
#  print_dict_json(dictionary):
#  print_full(x):
#  range_ave(s):
#  readCovidData(filepath):
#  redir(x, expr):
#  remove_newline(s):
#  remove_non_numeric(s):
#  remove_nonascii(string):
#  runtime(f):
#  show(images, n_cols=None):
#  sin1D(pix=512, start=0., stop=2*np.pi, amplitude=1., freq=1., offset=0., phase=0.):
#  sin2D(pix_x=512, start_x=0., stop_x=2*np.pi, freq_x=1., pix_y=512, start_y=0., stop_y=2*np.pi, freq_y=1, offset_z=0.):
#  sparse_diag(column_array, diag_index, R, C):
#  sparse2full(sparse_array, r, c):
#  string_to_one_hot(string, maxchar=128, minchar=0):
#  string2hex(s):
#  strip_html(data):
#  threshold(array, threshold=0):
#  timestamp():
#  view(image, cmap="binary"):
#  view_memory(f):
#  wrap_list(s, n=1):
#  xentropy(P, Q, normalize=True, nbins=100):

def timestamp():
  return datetime.datetime.now().isoformat() #strftime("%Y%m%d_%H%M%S")

def runtime(f):
    '''
    # decorator to measure performance of function f
    # Example:
       
        @runtime
        def test(n):
            j = 0
            for i in range(n):
                j+=1
       
        >>> test(100000)
        runtime: 0.0050120 s

    '''
    def run(*args):
        start = datetime.datetime.now()
        out = f(*args)
        end = datetime.datetime.now()
        elapsed = (end-start).total_seconds()
        print(f'runtime: {elapsed:.7f} s')
        return out
    return run

def log_memory(ref_id):
    pid = os.getpid()
    mem = psutil.Process(pid).memory_info()[0]/float(2**20)
    virt = psutil.virtual_memory()[3]/float(2**20)
    swap = psutil.swap_memory()[1]/float(2**20)

    print(f'[{timestamp()}]  [{ref_id}|{pid}]    rss: {mem:.2f} MB  (virtual: {virt:.2f} MB, swap: {swap:.2f} MB)')

def view_memory(f):
    '''
    >>> @view_memory
        def test(a):
            return 2*a

    >>> l = test(np.ones((1000000,), dtype='uint8'))
    [2022-01-31T23:46:44.906963]  [test|B|32060]    rss: 200.70 MB  (virtual: 15989.61 MB, swap: 33152.62 MB)
    [2022-01-31T23:46:44.908984]  [test|E|32060]    rss: 201.70 MB  (virtual: 15989.46 MB, swap: 33152.77 MB)
    
    ### Note rss increase of 1.0 MB, consistent with instantiating 1,000,000 8-bit integers

    '''
    def run(*args):
        log_memory(f'{f.__name__}|B')
        out = f(*args)
        log_memory(f'{f.__name__}|E')
        return out
    return run

def mv(fpath_from, fpath_to):
    shutil.move(fpath_from, fpath_to)

def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter

def arguments():

    '''
    credit to Kelly Yancey (kybyanc.blogspot.com/2007/07/python-aggregating-function-arguments.html)
    gets only the parameters passed to the function (i.e., no local variables defined within the function only)

    argcount = test_full.__code__.co_argcount
    m = test_full.__code__.co_varnames[:argcount]
    h = arguments()        
    args = [(f'{n}: {h[2][n]}') for n in list(m)]

    '''
    from inspect import getargvalues, stack
    posname, kwname, args = getargvalues(stack()[1][0])[-3:]
    posargs = args.pop(posname, [])
    args.update(args.pop(kwname, []))
    return args, posargs


def list_imported_modules():

    '''
    Calling U.list_imported_modules() will list modules imported by utils.py (not very useful)
    To get list of modules imported in active notebook, define this function locally in the notebook.
    >>> import os
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.preprocessing import StandardScaler
    >>> list_imported_modules()
    ['ModuleType',
     'os',
     'np',
     'plt',
     'StandardScaler']

    '''
    from types import ModuleType

    return [k for k,v in globals().items() if not k.startswith('_') and (type(v) is type(ModuleType) or type(v) is ModuleType)]

def list_imported_modules_(g):

    '''
    Pass globals() as argument to get list of modules imported in the local environment 
    >>> import sys
    >>> utilsdir = r'C:\\Users\\DrD\\JUPYTER\\UTILITIES'
    >>> sys.path.insert(1, utilsdir)
    >>> import utils as U
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt
    >>> from sklearn.preprocessing import StandardScaler
    >>> g = globals()
    >>> U.list_imported_modules_(g)
    ['sys',
     'U',
     'np',
     'plt',
     'StandardScaler',
     'ModuleType']
     
    '''
    from types import ModuleType

    return [k for k,v in g.items() if not k.startswith('_') and (type(v) is type(ModuleType) or type(v) is ModuleType)]



def change_extension(filename, ext):
    return '.'.join(filename.split('.')[:-1] + [ext])

def path2tuple(path):
    '''    
    recursively call os.path.split 
    return path components as tuple, preserving hierarchical order

    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> path2tuple(newdir)
    ('C:\\', 'temp', 'subdir0', 'subdir1', 'suubdir2')
          

    '''
    (a,b) = os.path.split(path)
    if b == '':
        return a,
    else:
        return *path2tuple(a), b

def mkpath(path):
    '''
    Similar to os.mkdir except mkpath also creates implied directory structure as needed.

    For example, suppose the directory "C:\\temp" is empty. Build the hierarchy "C:\\temp\\subdir0\\subdir1\\subdir2" with single call:
    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> mkpath(newdir)
        

    '''
    u = list(path2tuple(path))    
    pth=u[0]

    for i,j in enumerate(u, 1):
        if i < len(u):
            pth = os.path.join(pth,u[i])
            if not os.path.isdir(pth):
                os.mkdir(pth)


def mkpath_test(dir_path):
    '''
    Similar to os.mkdir except mkpath also creates parent directories if they don't exist
    For example, if the directory "C:\\temp" is empty, build "C:\\temp\\subdir0\\subdir1\\subdir2" with single call:
    >>> newdir = r'C:\\temp\\subdir0\\subdir1\\subdir2'
    >>> mkpath(newdir)

    '''
    
    print(f'call to function mkpath with argument {dir_path}')
    u = path2tuple(dir_path)
    print(f'[mkpath] u = {u}')
    u=list(u)
    print(f'[mkpath] list(u) = {u}')
    pth=u[0]
    print(f'[mkpath]: initial pth = u[0] = {pth}')
    for i,j in enumerate(u, 1):
        if i < len(u):
            pth = os.path.join(pth,u[i])
            print(f'[mkpath]: pth = {pth}')
            if not os.path.isdir(pth):
                print(f'===> {pth} IS NOT a directory')
            else:
                print(f'===> {pth} IS a directory')

def ddir(module, types=None, print_only=True, nunder=True):
    '''
    Enhanced summary of module contents based on dir().
    
    Example:

    >>> q = U.ddir(tf.keras.metrics.Accuracy, types=['method', 'function'], print_only=False)
    >>> print(q)
    {'function': ['add_loss', 'add_metric', 'add_update', ..., 'set_weights', 'update_state'], 'method': ['from_config', 'with_name_scope']}
    
    >>> ddir(tf.keras.metrics.Accuracy)

    :::::::::::::::: property ::::::::::::::::::

    activity_regularizer
    compute_dtype
     ...
    variables
    weights

    :::::::::::::::: function ::::::::::::::::::

    add_loss
    add_metric
    add_update
      ...
    set_weights
    update_state

    :::::::::::::::: method ::::::::::::::::::

    from_config
    with_name_scope

    >>> ddir(tf.keras.metrics.Accuracy, types=['method', 'function'])

    :::::::::::::::: method ::::::::::::::::::

    from_config
    with_name_scope

    :::::::::::::::: function ::::::::::::::::::

    add_loss
    add_metric
    add_update
       [...]
    set_weights
    update_state

    >>> ddir(tf.keras.metrics.Accuracy, types=['method', 'function'], nunder=False)

    :::::::::::::::: method ::::::::::::::::::

    from_config
    with_name_scope

    :::::::::::::::: function ::::::::::::::::::

    __call__
    __delattr__
    __getstate__
    __init__
    __new__
      ...
    add_loss
    add_metric
    add_update
      ...
    set_weights
    update_state

    '''
    tmp = {}
    for c in dir(module):
        if c == '':
            continue
        if nunder & (str(c)[0]=='_'):
            continue
        t = type(getattr(module, c))
        t = str(t).split("\'")[1]
        if t in tmp.keys():
            tmp[t].append(c)
        else:
            tmp[t] = [c]

    if types is not None:
        if ('function' in types) | ('method' in types):
            if 'builtin_function_or_method' not in types:
                types.append('builtin_function_or_method')
        if not print_only:
            return {key: value for key, value in tmp.items() if key in types}
        for x in types:
            if x in tmp.keys():
                if print_only:
                    print(f'\n:::::::::::::::: {x} ::::::::::::::::::\n')
                    _ = [print(w) for w in tmp[x]]                    
    else:
        if not print_only:
            return tmp
        for k,v in tmp.items():
            print(f'\n:::::::::::::::: {k} ::::::::::::::::::\n')
            _ = [print(w) for w in v]
        print()
  

def multiplot(index, list_of_arrays):
    
    q = tuple()
    for t in list_of_arrays:
            q = q + (index,t)
            
    _ = plt.plot(*q)

def redir(x, expr):
    ''' return items from dir(x) containing "expr" string '''
    return [tmp for tmp in dir(x) if expr in tmp]



def iseven(x):
    return x%2==0

def isodd(x):
    return x%2==1

def remove_nonascii(string):
    return string.encode('ascii', 'ignore').decode('utf-8')

def remove_non_numeric(s):
    """ remove all but the 12 numeric characters .-0123456789
    ['.', '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'] 
    """
    pattern = r'([^\.\-0-9]*)'
    return re.sub(pattern, '', s)

def string2hex(s):
    """ convert string characters to hexadecimal representation """
    return s.encode('utf-8').hex()

def hex2string(h):
    """ convert characters in hexadecimal representation to string ('utf-8')
    # h must have form:  b'\x7c'
    """
    return h.decode('utf-8')

def hex_remove(h, s):
    """ remove character h (specified in hexadecimal form) from string s
    # h must have form:  b'\x00\x00'
    """
    pattern = re.compile(h)
    return re.sub(pattern, '', s)

def strip_html(data):
    text = lxml.html.fromstring(str(data)).text_content()
    return text.encode('ascii', 'ignore').decode('utf-8')

# unwrap text (i.e., remove newlines introduced by line wrapping)
def remove_newline(s):
    t = s.replace('\n', ' ')
    pattern = r'(^\s)(.*)(\s$)'
    t = re.sub(pattern, r'\2', t)
    t = re.sub(b'\x0a'.decode('utf-8'), '', t)
    return re.sub('  ', ' ', t)

def wrap_list(s, n=1):
    return [s[q*n:(q+1)*n] for q in range(int(len(s)/n + 0.5))]


def char_to_one_hot(char, minchar=0, maxchar=128):
    """  [[2]] == [[0 1 2 3]] ---->> [[False False True False]]"""
    ascii = np.array([ord(char)-minchar])
    return np.array(ascii[:,None] == np.arange(maxchar-minchar)[None, :], dtype=int)

def string_to_one_hot(string, maxchar=128, minchar=0):
    """Converts an ASCII string to a one-of-k encoding.
    Use minchar to shift ord() values down if not using chars like ^@, ^A, ^B, ..."""
    ascii = np.array([(ord(c)-minchar) for c in string]).T  # transpose unnecessary?
    return np.array(ascii[:,None] == np.arange(maxchar-minchar)[None, :], dtype=int)


def range_ave(s):
    """ create numerical values from string, replacing value ranges with the range average """
    r = 0
    s = remove_non_numeric(str(s))
    x = s.split("-")
    if len(x) == 1:
        if len(x[0]) > 0:
            r = float(x[0])
    elif len(x) > 1:
        if len(x[0]) * len(x[1]) > 0:
            r = (float(x[0]) + float(x[1])) / 2
        elif len(x[0]) + len(x[1]) > 0:
            r = float(x[0] + x[1])

    return r


def autoscale_array(array):

    #array = array.astype(np.float32)
    lo = array.flatten().min()
    array -= lo    
    hi = array.flatten().max()
    try:
        assert hi > 0, f'autoscale_array cannot map null array to interval [0,1]'
    except AssertionError as msg:
        print(msg)
        return array

    array = array / hi
 
    return array.astype(np.float32)

def autoscale_arrays(A, B):

    lo = np.hstack([A,B]).flatten().min()
    
    A = A - lo
    B = B - lo

    hi = np.hstack([A,B]).flatten().max()
    A = A / hi
    B = B / hi
    return A, B


#TODO:  Merge entropy and image_entropy functions 

def image_entropy(array, bins=255, lo=0, hi=255):

    if array.dtype.name[:5] == 'float':
        array = (array * 255).astype(np.uint8)
    
    counts = np.histogram(a,bins=bins, range=(lo,hi))[0]
    frequencies = counts / counts.sum() + 1e-12
    return (-1* np.dot(frequencies, np.log2(frequencies)))


def image_xentropy(p, q, bins=255, lo=0, hi=255):
    if p.dtype.name[:5] == 'float':
        p = (p * 255).astype(np.uint8)

    if q.dtype.name[:5] == 'float':
        q = (q * 255).astype(np.uint8)
    
    counts_p = np.histogram(p,bins=bins, range=(lo,hi))[0]
    frequencies_p = counts_p / counts_p.sum() + 1e-12

    counts_q = np.histogram(q,bins=bins, range=(lo,hi))[0]
    frequencies_q = counts_q / counts_q.sum() + 1e-12

    return (-1* np.dot(frequencies_p, np.log2(frequencies_q)))



def image_KL(P,Q, bins=255, lo=0, hi=255):
    return image_xentropy(P,Q, bins=lo, lo=0, hi=hi) - image_entropy(P, bins=bins, lo=lo, hi=hi)

def image_KL(P,Q, bins=255, lo=0, hi=255):

    if p.dtype.name[:5] == 'float':
        p = (p * 255).astype(np.uint8)

    if q.dtype.name[:5] == 'float':
        q = (q * 255).astype(np.uint8)
    
    counts_p = np.histogram(p,bins=bins, range=(lo,hi))[0]
    frequencies_p = counts_p / counts_p.sum() + 1e-12

    counts_q = np.histogram(q,bins=bins, range=(lo,hi))[0]
    frequencies_q = counts_q / counts_q.sum() + 1e-12

    xentropy_pq = (-1* np.dot(frequencies_p, np.log2(frequencies_q)))
    entropy_p = (-1* np.dot(frequencies_p, np.log2(frequencies_p)))

    return xentropy_pq - entropy_p



def image_distribution(P, bins=255, lo=0, hi=255):

    if p.dtype.name[:5] == 'float':
        p = (p * 255).astype(np.uint8)
        
    counts_p = np.histogram(p,bins=bins, range=(lo,hi))[0]
    frequencies_p = counts_p / counts_p.sum() + 1e-12

    return 



def entropy(array, normalize=True, nbins=100):
    a = array.flatten().astype(np.float32)
    
    if normalize:
        a = normalize_array(a)
        lo = 0.
        hi = 1.
    else:
        lo = min(a) - 0.00001
        hi = max(a) + 0.00001
        
    n_bins = complex(0,nbins+1) # "nbins+1" edges is "nbins" bins
    bins = np.r_[lo:hi:n_bins]
    hist = np.histogram(a,bins=bins)
    counts = hist[0]
    frequencies = counts / counts.sum() + 1e-12
    return (-1* np.dot(frequencies, np.log2(frequencies)))

def xentropy(P, Q, normalize=True, nbins=100):
    p = P.flatten().astype(np.float32)
    q = Q.flatten().astype(np.float32)
    
    if normalize:
        p = normalize_array(p)
        q = normalize_array(q)
        lo = 0.
        hi = 1.
    else:
        lo = min(np.hstack([q,p])) - 0.00001
        hi = max(np.hstack([q,p])) + 0.00001
        
    n_bins = complex(0,nbins)
    bins = np.r_[lo:hi:n_bins]
    hist_p = np.histogram(p,bins=bins)
    counts_p = hist_p[0][hist_p[0]>0]
    frequencies_p = counts_p / counts_p.sum()

    hist_q = np.histogram(q,bins=bins)
    counts_q = hist_q[0][hist_q[0]>0]
    frequencies_q = counts_q / counts_q.sum()

    return (-1* np.dot(frequencies_p, np.log2(frequencies_q)))
def print_array(array):
    """ print array to console. formatted for copy/paste into excel (brackets, whitespace, etc removed)"""
    if array.ndim == 1:
        array = array[:,None]
        
    rows = array.shape[0]
    
    for row in range(rows):
        print(str(array[row,:]).replace('[ ','').replace('[','').replace(']','').replace('  ',' ').replace(' ',','))

def pad2square(array_1d, dtype=np.int8):
    N = np.max(array_1d.shape)
    N_ix = np.argmax(array_1d.shape)    
    array_2d = np.zeros((N,N), dtype=dtype)
    
    if array_1d.ndim==1:
        array_2d[0,:] = array_1d[:]
    elif array_1d.ndim==2:
        if N_ix == 0:
            array_2d[0,:] = array_1d[:,0]
        else:
            array_2d[:,0] = array_1d[0,:]
    else:
        print(f'ERROR: Input array has {array_1d.ndim} dimensions.  Cannot pad it to make a square array')
        return np.array([])
        
    return array_2d

def blend(a1,a2,a3=None):
    n, m = a1.shape
    if a3 is None:
        a3 = np.zeros((m,n))
        
    q = np.zeros((n,m,3))
    q[:,:,0] = a1
    q[:,:,1] = a2
    q[:,:,2] = a3
    
    return q

def diff(x, axis=0):
    if axis==0:
        return x[1:,:]-x[:-1,:]
    else:
        return x[:,1:]-x[:,:-1]


def cyclic_diff(x,axis=0):
    if axis==0:
        return x[0,:]-x[-1,:]
    else:
        return (x[:,0]-x[:,-1])[None,:].T


def flatten_by_cols(x):
    return x.T.reshape(np.prod(x.shape), -1).flatten()

def flatten_by_rows(x):
    return x.reshape(-1, np.prod(x.shape)).flatten()

def sparse_diag(column_array, diag_index, R, C):
    ''' Returns a sparse matrix with elements "column_array" assigned to the idx-th diagonal'''
    a = column_array.copy()
    idx = np.abs(diag_index)
    N = a.shape[0]
    row_idx = np.arange(N)[None,:].T
    col_idx = np.arange(N)[None,:].T

    if diag_index > 0:
        col_idx -= idx
    elif diag_index < 0:
        row_idx += idx
        
    w = np.hstack([row_idx, col_idx]).astype(np.int16)
    q = (row_idx<R)*(col_idx>=0)

    w = w[q[:,0]]
    return tuple(w.T.tolist()), a[q[:,0]].flatten()

def sparse2full(sparse_array, r, c):
    try:
        assert len(sparse_array)==2, "Incompatible input shape"
    except AssertionError as msg:
        print(msg)
        return None
    rows = max(r,1+max(sparse_array[0][0]))
    cols = max(c,1+max(sparse_array[0][1]))
    full_array = np.zeros((rows, cols),dtype=sparse_array[1].dtype)
    full_array[sparse_array[0]] = sparse_array[1]
    return full_array

from numpy.fft import fft2, ifft2

def np_fftconvolve(A, B):
    return np.real(ifft2(fft2(A)*fft2(B)))


############# array diagonal index manipulation


def diagrid(N,M):
    '''
    wrap integer sequence {0,..,N*M-1} diagonally into N x M numpy array 

    Example:

    >>> diagrid(4,5)

        array([[ 0,  2,  5,  9, 13],
               [ 1,  4,  8, 12, 16],
               [ 3,  7, 11, 15, 18],
               [ 6, 10, 14, 17, 19]])
    '''
    a = np.zeros((N,M), dtype='int')
    n = 0

    for i in range(N):
        for k in range(min(i+1,M)):
          a[i-k,k] = n
          n+=1
    if N < M:
        for j in np.arange(0,M-1):
          for k in np.arange(0,min(M-j-1,N)):
            a[(N-1)-k, 1+j+k] = n
            n+=1
    else:
        for j in np.arange(0,M-1):
          for k in np.arange(0,M-j-1):
            a[(N-1)-k, 1+j+k] = n
            n+=1
    return a


def diagonal_wrap(x, num_cols=1, clip=False):
    '''
    creates 2-tuple of index lists (i.e., ([x],[y])) used to slice a diagonally unwound 2d numpy array 

    Example:

    >>> diagrid(3,2)

        array([[0, 2],
               [1, 4],
               [3, 5]])
    >>> diagonal_slice(3,2)
    
        ([0, 1, 0, 2, 1, 2], [0, 0, 1, 0, 1, 1])
    
    >>> diagrid(3,2)[diagonal_slice(3,2)]
    
        array([0, 1, 2, 3, 4, 5])

    '''

    max_stop = N*M
    if stop == -1:
      stop = max_stop
    else:
      stop = min(stop, max_stop)

    try:
      assert start < stop, f'Error: diagonal_slice({N},{M}, start={start}, stop={stop}, step={step}) ==> start must be less than stop. Applying override: start = 0.'
    except AssertionError as msg:
      print(msg)
      start = 0
    try:
      assert (step > 0) & (step < max_stop), f'Error: diagonal_slice({N},{M}, start={start}, stop={stop}, step={step}) ==> step must be less than N*M. Applying override: step = 1'
    except AssertionError as msg:
      print(msg)      
      step = 1

    x = np.zeros((N*M,), dtype=int)
    y = np.zeros((N*M,), dtype=int)
    n = 0

    for i in range(N):
        for k in range(min(i+1,M)):
          x[n] = i-k
          y[n] = k
          n+=1
    if N < M:
        for j in np.arange(0,M-1):
          for k in np.arange(0,min(M-j-1,N)):
            x[n] = (N-1)-k
            y[n] = 1+j+k
            n+=1
    else:
        for j in np.arange(0,M-1):
          for k in np.arange(0,M-j-1):
            x[n] = (N-1)-k
            y[n] = 1+j+k
            n+=1

    x = x[start:stop:step]
    y = y[start:stop:step]

    return tuple(np.vstack([x,y]).tolist())


def diagonal_wrap(x, num_cols=1, clip=False):
    '''
    wraps 1d array into 2d array along diagonals with lower left to upper right chirality

    Example:

    >>> x = np.array([20, 30, 40, 50, 60, 70, 80])
    >>> diagonal_wrap(x,3)

        array([[20, 40, 70],
               [30, 60,  0],
               [50, 80,  0]])

    >>> diagonal_wrap(x,3, clip=True)

        array([[20, 40],
               [30, 60],
               [50, 70]])

    >>> diagonal_wrap(x,2, clip=True)

        array([[20, 40, 60],
               [30, 50, 70]])

    '''
    if clip:
        num_rows = int(len(x)/num_cols)
        a = np.zeros((num_cols,num_rows), dtype=x.dtype)
        a[diagonal_slice(num_cols,num_rows)] = x[:num_cols*num_rows]

    else:
        num_rows = round(len(x)/num_cols + 0.499)
        a = np.zeros((num_cols,num_rows), dtype=x.dtype)
        a[diagonal_slice(num_cols,num_rows, stop=len(x))] = x


    return a  

def diagonal_unwrap(a):
    '''
    inverse action of diagonal_wrap(x):
        Example:

    >>> x = np.array([20, 30, 40, 50, 60, 70, 80])
    >>> y = diagonal_wrap(x,3)
    >>> print(y)

        array([[20, 40, 70],
               [30, 60,  0],
               [50, 80,  0]])

    >>> diagonal_unwrap(y)

        array([20, 30, 40, 50, 60, 70, 80,  0,  0])

    '''
    return a[diagonal_slice(*tuple(a.shape))]    

######################  NUMPY  ###############################

#np.genfromtxt(filepath, delimiter=',', dtype=None, names=True, encoding='utf-8')

# binary representation (decompose int -> {0,1} array)
# def bit_hot(n):
#     binary_array = np.array([int(x) for x in bin(n)[2:]])
#     out = np.zeros((8,), dtype=np.uint8)
#     out[:binary_array.shape[0]] = binary_array
#     return out

# #  binary representation (decompose int -> bool array)
# def bool_hot(n):
#     binary_array = np.array([int(x) for x in bin(n)[2:]], dtype='bool')
#     out = np.zeros((8,), dtype=np.bool)
#     out[:binary_array.shape[0]] = binary_array
#     return out


######################SCIPY#####################################

from scipy.sparse.linalg import spilu
from scipy.sparse import csc_matrix

def invert_as_sparse(a):
    s = spilu(csc_matrix(a))
    return s.solve(np.eye(a.shape[0]))


#################### DIRECTORY / FILE SEARCH ####################

def find(key, recursive=False):
    return [x for x in glob.glob(''.join(['*',key,'*']), recursive=recursive)];

 
#################### DATES ####################

def convert_iso8601(iso_time, date=True, time=False):  # 20220616 added output format/type options
    '''iso_time: integer, 10-digit number of seconds since 1970-01-01 00:00:00 (i.e., date & time in iso8601 format)
       returns: 
                string:
                  w/format:
                    YYYY-MM-DD            date=True, time=False (default)
                    YYYY-MM-DD HH:mm:s    date=True, time=True
                    HH:mm:ss              date=False, time=True
            or
                object:
                    datetime.datetime
    '''
    formt = []
    if date:
        formt.append(r'%Y-%m-%d')
    if time:
        formt.append(r'%H:%M:%S')

    string_format = ' '.join(formt)
    
    if len(string_format) > 0:
        return datetime.datetime.utcfromtimestamp(iso_time).strftime(string_format)
    else:
        return datetime.datetime.utcfromtimestamp(iso_time)

        #return datetime.datetime.utcfromtimestamp(iso_time).strftime('%Y-%m-%d')


# def get_iso8601(date_and_time):
#     '''  d must be a string with format YYYY-MM-DD HH:mm:ss 
#          returns date & time in iso8601 format (i.e., number of seconds since 1970-01-01 00:00:00)
#     '''
#     date, time = date_and_time.split(' ')
#     year, month, day = date.split('-')
#     hour, minute, second = time.split(':')
#     #print(f'{year} {month} {day} {hour} {minute} {second}')
    
#    return 86400*(datetime.date(int(year), int(month), int(day)).toordinal() - datetime.date(1970,1,1).toordinal()) + 3600*int(hour) + 60*int(minute) + int(second)

def get_iso8601(date_and_time): # 20220616 added support for date-only input
    '''  d must be a string with format YYYY-MM-DD HH:mm:ss or YYYY-MM-DD
         returns date & time in iso8601 format (i.e., number of seconds since 1970-01-01 00:00:00)
    '''

    hour, minute, second = ('00','00','00')

    if ' ' in date_and_time:
        date, time = date_and_time.split(' ')
        if ':' in time:
            hour, minute, second = time.split(':')
    else:
        date = date_and_time.strip()

    year, month, day = date.split('-')

    #print(f'{year} {month} {day} {hour} {minute} {second}')
    
    return 86400*(datetime.date(int(year), int(month), int(day)).toordinal() - datetime.date(1970,1,1).toordinal()) + 3600*int(hour) + 60*int(minute) + int(second)

def get_iso8601_offset(date_and_time, count, granularity):
    ''' date: must be in YYYY-MM-DD HH:mm:ss string format
        count: number of intervals
        granularity: number of seconds per interval

        returns date & time, offset by (count*granularity) seconds, in iso8601 format and YYYY-MM-DD HH:mm:ss string format 
        use negative count (or negative granularity) to offset backwards in time
    '''
    d_iso8601 = get_iso8601(date_and_time)
    d_iso8601_offset = d_iso8601 + count*granularity
    return d_iso8601_offset, pd.to_datetime(d_iso8601_offset, unit='s').__str__()
    
#################### ARRAY DECOMPOSITION TO BIT LAYERS ####################

def bit_hot(array):
    return np.unpackbits(array).reshape(*array.shape, 8)

def bool_hot(array):
    return np.unpackbits(array).reshape(*array.shape, 8).astype(np.bool)

def bit_blast(array):
    return np.unpackbits(array).reshape(8, *array.shape)

def bits2U8(bit_hot_array):
    base2=2**np.arange(8, dtype=np.uint8)[::-1]
    return np.dot(bit_hot_array, base2)

def threshold(array, threshold=0):
    A = array.copy()
    A[A<threshold] = 0
    A[A!=0] = 1
    return A

def peek(dictionary, n=3, delim='|'):
    [print(i,delim,dictionary[i]) for i in list(dictionary.keys())[:n]]

def peek2(dictionary, delim='|'):
    {print(k,delim,v) for k,v in dictionary.items()}

def print_dict_json(dictionary):
    print(json.dumps(dictionary, indent=4, sort_keys=True))


def normalize_array(array):
    if array.ndim==3:
        array_ = array - array.min(axis=2).min(axis=1).min(axis=0)
        return array_ / array_.max(axis=2).max(axis=1).max(axis=0)
    
    if array.ndim==2:
        array_ = array - array.min(axis=1).min(axis=0)
        return array_ / array_.max(axis=1).max(axis=0)

    if array.ndim==1:
        array_ = array - array.min(axis=0)
        return array_ / array_.max(axis=0)




# def array_info(array, print_info=True, return_info=False, return_info_str=False):

#     info = {}
#     info['dtype'] = array.dtype
#     info['ndim'] = array.ndim
#     info['shape'] = array.shape
#     info['max'] = array.max()
#     info['min'] = array.min()
#     info['mean'] = array.mean()
#     info['std'] = array.std()
#     info['size'] = array.size
#     info['nonzero'] = np.count_nonzero(array)
#     info['layer_variation'] = 0
#     info['entropy'] = entropy(array)

#     if array.ndim > 2:
#         info['layer_variation'] = array.std(axis=array.ndim-1).mean()

#     info['pct'] = 100 * info['nonzero'] / info['size']

#     if print_info:
#         print('{dtype}  {shape}'.format(**info))
#         print('nonzero: {nonzero} / {size}  ({pct:.1f} %)'.format(**info))
#         print('min:  {min:.2f}   max: {max:.2f}'.format(**info))
#         print('mean: {mean:.2f}   std: {std:.2f}'.format(**info), end="")
#         if array.ndim > 2:
#             print('     layer_variation: {layer_variation:.2f}'.format(**info))
#         else:
#             print('\n')

#         print('entropy: {entropy:.2f}'.format(**info), end="")

#     out = []
#     if return_info:
#         out.append(info)
#     if return_info_str:
#         info_str = f'shape: {info["shape"]}\n'
#         info_str += f'size: {info["size"]}\nnonzero: {info["nonzero"]}  ({info["pct"]:.4f} %)\n'
#         info_str += f'min: {info["min"]}    max: {info["max"]}\n'
#         info_str += f'mean: {info["mean"]:.4f}    std: {info["std"]:.4f}\n'
#         if array.ndim > 2:
#             info_str += f'layer_variation: {info["layer_variation"]:.4f}\n'
#         else:
#             print('\n')
            
#         info_str += f'entropy: {info["entropy"]:.4f}\n'

#         out.append(info_str)
        
#     if return_info or return_info_str:
#         return out

def array_info(array, print_info=True, return_info=False, return_info_str=False, name=None):
    '''

    Calculate array properties
    Print formatted string [Optional]  (Default)

    Returns:            
        Return info as dictionary (return_info=True) [Optional]
        Return info as formatted string (return_info_str=True) [Optional]
        
    Example:
    >>> x = np.random.randint(0,255,size=(1080, 1920, 3), dtype=np.uint8)
    >>> array_info(x, name='x')

        **********************************
        **********************************
            x
        **********************************

        bytes: 6220944
        uint8  (1080, 1920, 3)
        nonzero: 6196350 / 6220800  (99.6)
        min:  0.00   max: 254.00
        mean: 127.01   std: 73.63     layer_variation: 54.83
        entropy: 7.99
        **********************************

    '''
    info = {}
    info['name'] = str(name)
    info['bytes'] = sys.getsizeof(array)
    info['dtype'] = array.dtype
    info['ndim'] = array.ndim
    info['shape'] = array.shape
    info['max'] = array.max()
    info['min'] = array.min()
    info['mean'] = array.mean()
    info['std'] = array.std()
    info['size'] = array.size
    info['nonzero'] = np.count_nonzero(array)
    info['layer_variation'] = 0
    info['entropy'] = entropy(array)

    if array.ndim > 2:
        info['layer_variation'] = array.std(axis=array.ndim-1).mean()

    info['pct'] = 100 * info['nonzero'] / info['size']

    if print_info:
        if info["name"] is not None:
            print(f'\n**********************************\n**********************************\n    {info["name"]}\n**********************************\n')
        print(f'bytes: {info["bytes"]}')
        print(f'{info["dtype"]}  {info["shape"]}')
        print(f'nonzero: {info["nonzero"]} / {info["size"]}  ({info["pct"]:.1f})')
        print(f'min:  {info["min"]:.2f}   max: {info["max"]:.2f}')
        print(f'mean: {info["mean"]:.2f}   std: {info["std"]:.2f}', end="")
        if info["ndim"] > 2:
            print(f'     layer_variation: {info["layer_variation"]:.2f}')
        else:
            print('\n')

        print(f'entropy: {info["entropy"]:.2f}')#, end="")
        print(f'**********************************\n')    
    out = []
    if return_info:
        out.append(info)
    if return_info_str:
        info_str = f'name: {info["name"]}\n'
        info_str += f'bytes: {info["bytes"]}\n'
        info_str += f'shape: {info["shape"]}\n'
        info_str += f'size: {info["size"]}\nnonzero: {info["nonzero"]}  ({info["pct"]:.4f} %)\n'
        info_str += f'min: {info["min"]}    max: {info["max"]}\n'
        info_str += f'mean: {info["mean"]:.4f}    std: {info["std"]:.4f}\n'
        if array.ndim > 2:
            info_str += f'layer_variation: {info["layer_variation"]:.4f}\n'
        else:
            print('\n')
            
        info_str += f'entropy: {info["entropy"]:.4f}\n'

        out.append(info_str)
 
    if return_info or return_info_str:
        if len(out)==1:
            return out[0]
        else:
            return out

def print_array_info(info):
    '''
    Print items in info

    info: dictionary [Note:  Must be created by or equivalent to the dictionary returned by array_info(array, return_info=True)]

    Example:

    >>> x = np.random.randint(0,255,size=(1080, 1920, 3), dtype=np.uint8)
    >>> array_info(x, name='x')

        **********************************
        **********************************
            x
        **********************************

        bytes: 6220944
        uint8  (1080, 1920, 3)
        nonzero: 6196350 / 6220800  (99.6)
        min:  0.00   max: 254.00
        mean: 127.01   std: 73.63     layer_variation: 54.83
        entropy: 7.99
        **********************************

    >>> y = array_info(x, name='x', return_info=True, print_info=False)
    >>> print_array_info(y)

        **********************************
        **********************************
            x
        **********************************

        bytes: 6220944
        uint8  (1080, 1920, 3)
        nonzero: 6196350 / 6220800  (99.6)
        min:  0.00   max: 254.00
        mean: 127.01   std: 73.63     layer_variation: 54.83
        entropy: 7.99
        **********************************

    '''

    if info["name"] is not None:
        print(f'\n**********************************\n**********************************\n    {info["name"]}\n**********************************\n')
    print(f'bytes: {info["bytes"]}')
    print(f'{info["dtype"]}  {info["shape"]}')
    print(f'nonzero: {info["nonzero"]} / {info["size"]}  ({info["pct"]:.1f})')
    print(f'min:  {info["min"]:.2f}   max: {info["max"]:.2f}')
    print(f'mean: {info["mean"]:.2f}   std: {info["std"]:.2f}', end="")
    if info["ndim"] > 2:
        print(f'     layer_variation: {info["layer_variation"]:.2f}')
    else:
        print('\n')

    print(f'entropy: {info["entropy"]:.2f}')#, end="")
    print(f'**********************************\n')    


#######################  PANDAS  ####################################

def df_get(df,column_name,value):
    """ returns the dataframe row(s) with matching 'value' in 'column_name' """
    return df[df[column_name]==value].T


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 2000)
    pd.set_option('display.float_format', '{:20,.2f}'.format)
    pd.set_option('display.max_colwidth', None)
    print(x)
    pd.reset_option('display.max_rows')
    pd.reset_option('display.max_columns')
    pd.reset_option('display.width')
    pd.reset_option('display.float_format')
    pd.reset_option('display.max_colwidth')

def encodeDF(df):
    column_unique_values = {}
    column_value_indices = {}
    numeric_columns = []
    non_numeric_columns = []
    for col in df.columns:
        if df[col].dtype == 'object':
            column_value_indices[col], column_unique_values[col] = df[col].factorize()
            non_numeric_columns.append(col)
        else:
            column_value_indices[col] = df[col].copy()
            numeric_columns.append(col)
    
    headers = {'non_numeric':non_numeric_columns, 'numeric':numeric_columns }
    return headers, column_unique_values, pd.DataFrame(column_value_indices)


def readCovidData(filepath):
    lines = []
    with open(filepath) as f: # r'D:\DATA\CORONAVIRUS\covid_NYC_20201007.txt'
            for line in f:
                s0 = '|'.join(line.strip().split('\t'))
                s1 = '|'.join(next(f).strip().split('\t'))
                s2 = '|'.join(next(f).strip().split('\t'))
                lines.append('|'.join([s0, s1, s2]))
                
    data = pd.DataFrame(lines)
    names = data[0].iloc[0].split('|')
    names = ['ZIP, BOROUGH', 'HOOD'] + names
    data = pd.DataFrame(data[0].iloc[1:].str.split('|').tolist(), columns=names)

    for name in names[2:]:
        data[name] = data[name].apply(lambda x: int(x.replace(',','')) if x.replace(',','').isdigit() else -1)

    return data

########################  IMAGE PROCESSING  #############################

def view(image, cmap="binary"):
    plt.figure()
    plt.subplot(1,1,1)
    plt.imshow(image, cmap)
    plt.show()

def show(images, n_cols=None):
    n_cols = n_cols or len(images)
    n_rows = (len(images) - 1) // n_cols + 1
    if images.shape[-1] == 1:
        images = np.squeeze(images, axis=-1)
    plt.figure(figsize=(n_cols, n_rows))
    for index, image in enumerate(images):
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(image, cmap="binary")
        plt.axis("off")

def sin1D(pix=512, start=0., stop=2*np.pi, amplitude=1., freq=1., offset=0., phase=0.):
    axis_range = (start, stop)
    x = np.linspace(*axis_range, num=pix)
    y = amplitude * np.sin(freq * x + phase) + offset
    return x, y
    #>>>  x, y = sin1D()
    #>>>  plt.plot(x, y)

def cos1D(pix=512, start=0., stop=2*np.pi, amplitude=1., freq=1., offset=0., phase=0.):
    axis_range = (start, stop)
    x = np.linspace(*axis_range, num=pix)
    y = amplitude * np.cos(freq * x + phase) + offset
    return x, y
    #>>>  x, y = cos1D()
    #>>>  plt.plot(x, y)

def sin2D(pix_x=512, start_x=0., stop_x=2*np.pi, freq_x=1., pix_y=512, start_y=0., stop_y=2*np.pi, freq_y=1, offset_z=0.):
    axis_range_x = (start_x, stop_x)
    axis_range_y = (start_y, stop_y)
    x = np.linspace(*axis_range_x, num=pix_x)
    y = np.linspace(*axis_range_y, num=pix_y)
    
    #### equivalent, using np.arange() ####
    #axis_range = (-np.pi, np.pi)
    #pixels = 512
    #length = np.abs(np.array([*axis_range])).sum()
    #step_size = length / pixels
    #x = np.arange(0, 2*np.pi, step=step_size)

    Ax = np.sin(freq_x * x)
    Ay = np.sin(freq_y * y)
    
    return offset_z + np.outer(Ax, Ay)
    #>>>  plt.imshow(sin2D())

def cos2D(pix_x=512, start_x=0., stop_x=2*np.pi, freq_x=1., pix_y=512, start_y=0., stop_y=2*np.pi, freq_y=1, offset_z=0.):
    axis_range_x = (start_x, stop_x)
    axis_range_y = (start_y, stop_y)
    x = np.linspace(*axis_range_x, num=pix_x)
    y = np.linspace(*axis_range_y, num=pix_y)
    
    #### equivalent, using np.arange() ####
    #axis_range = (-np.pi, np.pi)
    #pixels = 512
    #length = np.abs(np.array([*axis_range])).sum()
    #step_size = length / pixels
    #x = np.arange(0, 2*np.pi, step=step_size)

    Ax = np.cos(freq_x * x)
    Ay = np.cos(freq_y * y)
    
    return offset_z + np.outer(Ax, Ay)
    #>>>  plt.imshow(cos2D())


## Source https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif
def generate_and_save_images(model, epoch, test_input):
    # Notice 'training' is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)
    
    fig = plt.figure(figsize=(10,10))
    
    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='binary')
        plt.axis('off')
        
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()

## Source: https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif

def create_gif(filename_regex, anim_file='new.gif'):
#anim_file = 'dcgan.gif'

    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(filename_regex)
        filenames = sorted(filenames)
        last = -1
        for i,filename in enumerate(filenames):
            frame = 2*(i**0.5)
            if round(frame) > round(last):
                last = frame
            else:
                continue
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)

    return anim_file



# def entropy(array, normalize=True, nbins=100):  #modified 20211115 0220: normalize frequencies
#     a = array.flatten()
#     a = a.astype(np.float32)

#     if normalize:
#         a = normalize_array(a)
#         lo = 0.
#         hi = 1.
#     else:
#         lo = min(a) - 0.001
#         hi = mx(a) + 0.001
        
#     n_bins = complex(0,nbins)
#     bins = np.r_[lo:hi:n_bins]  #np.arange(a.min()-1,a.max())+0.5
#     hist = np.histogram(a,bins=bins) #, density=True)
#     h01 = hist[0] / np.prod(a.shape)
#     #h01 = normalize_array(hist[0])
#     h = h01[h01>0]
#     return (-1* h* np.log2(h)).sum()


# def array_info(array, print_info=True, return_info=False):
#     a ={}
#     a['dtype'] = array.dtype
#     a['ndim'] = array.ndim
#     a['shape'] = array.shape
#     a['max'] = array.max()
#     a['min'] = array.min()
#     a['mean'] = array.mean()
#     a['std'] = array.std()
#     a['size'] = array.size
#     a['nonzero'] = np.count_nonzero(array)
#     a['layer_variation'] = 0
#     a['entropy'] = entropy(array)
    
#     a['pct'] = 100 * a['nonzero'] / a['size']
    
#     if print_info:
#         print('{dtype}  {shape}'.format(**a))
#         print('nonzero: {nonzero} / {size}  ({pct:.1f} %)'.format(**a))
#         print('min:  {min:.2f}   max: {max:.2f}'.format(**a))
#         if array.ndim > 2:
#             print('mean: {mean:.2f}   std: {std:.2f}'.format(**a), end="")
#             a['layer_variation'] = array.std(axis=array.ndim-1).mean()
#             print('     layer_variation: {layer_variation:.2f}'.format(**a))
#         else:
#             print('mean: {mean:.2f}   std: {std:.2f}'.format(**a))

#         print('entropy: {entropy:.2f}'.format(**a), end="")

#     # if print_info:
#     #     print('%(dtype)s  %(shape)s' %a)
#     #     print('nonzero: %(nonzero)i / %(size)i'%a)
#     #     print('min:  %(min)s   max: %(max)s' %a)
#     #     print('mean: %(mean).2f   std: %(std).2f    layer_variation: %(layer_variation).2f' %a)
    
#     if return_info:
#         return a
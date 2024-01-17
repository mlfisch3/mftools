import numpy as np

#  add_layers(image, layers):    
#  add_row(row_index, column_indices):
#  bigrid(a):
#  blend(image1, image2, stride=2):
#  concat_coords(a, b):
#  concat_coords_np(a,b):
#  coord_grid(array):
#  coord_grid(array, stride=2):
#  coord_grid_even(array, stride=2):
#  coord_grid_odd(array, stride=2):
#  cos2D(pix_x=512, start_x=0., stop_x=2*np.pi, freq_x=1., pix_y=512, start_y=0., stop_y=2*np.pi, freq_y=1, offset_z=0.):
#  create_even_and_odd_coordinates(x):
#  gradient(size=(100,100)):
#  index_array(N,M):
#  mask(array, mask_width=2,intensity=(0,255)):
#  pad_array(input_array, pad_rows=1, pad_cols=1, intensity=255, rgb=(255,255,255)):
#  radial_gradient_layers(N, step=1,center='light', noisy=False):
#  shift_coords(coords,dx,dy):
#  shift_indices(indices,dx,dy, inplace=False):
#  sin2D(pix_x=512, start_x=0., stop_x=2*np.pi, freq_x=1., pix_y=512, start_y=0., stop_y=2*np.pi, freq_y=1, offset_z=0.):
#  topairs(coords):
#  topairs_np(coords):


def index_array(N,M):

    '''
    create 2d array with elements equal to their [row, column] index pairs
    (i.e., array A with shape (N,M,2) and A[i,j,:] = [i,j]  for (0 ≤ i < N) and (0 ≤ j < M))

    example:

    >>> index_array(3,2)

    array([[[0., 0.],
            [0., 1.]],

           [[1., 0.],
            [1., 1.]],

           [[2., 0.],
            [2., 1.]]])    

    >>> A = index_array(10,10)
    >>> a[7,3]
    array([7., 3.])

    '''
    aa = np.arange(N)[:,None]*np.ones((M))
    bb = np.arange(M)[:,None]*np.ones(N)
    bb = bb.T

    return np.append(aa[:,:,None], bb[:,:,None], axis=2)


def bigrid(a):

    MAX = a.ravel().max()
    A = np.array(a[:,:,None]==np.arange(MAX)[None,:], dtype='int')
    
    return A * a[:,:,None]

def pad_array(input_array, pad_rows=1, pad_cols=1, intensity=255, rgb=(255,255,255)):

    z = np.array(input_array.shape)
    if input_array.ndim == 2:
        z = z + [2*pad_rows, 2*pad_cols]
        padded_array = np.ones(tuple(z), dtype=input_array.dtype) * int(intensity)
    elif input_array.ndim == 3:
        z = z + [2*pad_rows, 2*pad_cols, 0]
        padded_array = np.ones(tuple(z), dtype=input_array.dtype) * np.array(rgb, dtype=input_array.dtype)

    padded_array[pad_rows:-pad_rows, pad_cols:-pad_cols] = input_array

    return padded_array
    
# def pad_array(input_array, pad_rows=1, pad_cols=1, intensity=255, rgb=(255,255,255)):
#     z = np.array(input_array.shape)
#     if input_array.ndim == 2:
#         z = z + [2*pad_rows, 2*pad_cols]
#         padded_array = np.ones(tuple(z)) * intensity
#     elif input_array.ndim == 3:
#         z = z + [2*pad_rows, 2*pad_cols, 0]
#         padded_array = np.ones(tuple(z)) * np.array(rgb)

#     padded_array[pad_rows:-pad_rows, pad_cols:-pad_cols] = input_array

#     return padded_array

def create_even_and_odd_coordinates(x):
    coords_even = tuple(np.array([x[::2],y[::2]]).tolist())
    coords_odd = tuple(np.array([x[::2],y[::2]]).tolist())
    return coords_even, coords_odd

def add_row(row_index, column_indices):
    y = column_indices.copy()
    x = np.ones(y.shape[0], dtype=np.int16) * row_index
    coordinates = np.array([x,y])
    #return x.flatten(), y.flatten()
    return tuple(coordinates.tolist())

def concat_coords(a, b):
    x = a[0] + b[0]
    y = a[1] + b[1]
    return x,y

def concat_coords_np(a,b):
    return tuple(np.hstack([np.array(a),np.array(b)]).tolist())

def topairs(coords):
    pairs = []
    for i,j in zip(*coords):
        pairs.append([i,j])
        return pairs
    
def topairs_np(coords):
    pairs = np.array(coords).T
    return pairs.tolist()

# def coord_grid(array):
    
#     '''
#     example use: assign value to every other array element
    
#     >>> x = np.zeros((200,200))
#     >>> x[coord_grid(x)]=255
#     >>> img.imshow(x)

#     example use #2: create hybrid image
    
#     >>> x = np.zeros(image1.shape)
#     >>> grid = coord_grid(x)
#     >>> image_blend = image1.copy()
#     >>> image_blend[grid]=image2[grid]
#     >>> img.imshow(image_blend)

#     '''

#     M, N =array.shape
#     index = np.arange(0,N)
#     even = index[::2]
#     odd = index[1::2]
    
#     coords = ([-1],[-1])
#     for i in list(range(M))[::2]:
#         a = add_row(i, even)
#         b = add_row(i+1, odd)
#         c =  concat_coords(a,b)
#         coords = concat_coords(coords,c)
        
#     coords = (coords[0][1:], coords[1][1:])
#     return coords


def coord_grid(array, stride=2):
    
    '''
    example use: assign value to every other array element
    >>> x = np.zeros((200,200))
    >>> x[coord_grid(x)]=255
    >>> img.imshow(x)

    example use #2: create hybrid image
    >>> x = np.zeros(image1.shape)
    >>> grid = coord_grid(x)
    >>> image_blend = image1.copy()
    >>> image_blend[grid]=image2[grid]
    >>> img.imshow(image_blend)

    '''

    M, N =array.shape
    index = np.arange(0,N)
    even = index[::stride]
    odd = index[1::stride]
    
    coords = ([-1],[-1])
    for i in list(range(M))[::stride]:
        a = add_row(i, even)
        if i < M-1:
            b = add_row(i+1, odd)
            c =  concat_coords(a,b)
            coords = concat_coords(coords,c)
        else:
            coords = concat_coords(coords,a)
            
    coords = (coords[0][1:], coords[1][1:])
    return coords

def coord_grid_even(array, stride=2):
    
    M, N =array.shape
    index = np.arange(0,N)
    even = index[::stride]
    #odd = index[1::stride]
    
    coords = ([-1],[-1])
    for i in list(range(M))[::stride]:
        a = add_row(i, even)
        #b = add_row(i+1, odd)
        #c =  concat_coords(a,b)
        coords = concat_coords(coords,a)
        
    coords = (coords[0][1:], coords[1][1:])
    return coords

def coord_grid_odd(array, stride=2):
    
    M, N =array.shape
    index = np.arange(0,N)
    #even = index[::stride]
    odd = index[1::stride]
    
    coords = ([-1],[-1])
    for i in list(range(M))[::stride]:
     #   a = add_row(i, even)
        b = add_row(i+1, odd)
      #  c =  concat_coords(a,b)
        coords = concat_coords(coords,b)
        
    coords = (coords[0][1:], coords[1][1:])
    return coords

def shift_coords(coords,dx,dy):
    '''
    example use with coord_grid_even():
    
    >>> fig = plt.figure(figsize=(10,10))
    >>> q = np.zeros((20,20),dtype='uint8')
    >>> coords = tiles.coord_grid_even(q, stride=4)
    >>> coords=shift_coords(coords, 1, 1)
    >>> q[coords]=255
    >>> coords=shift_coords(coords, 1, 0)
    >>> q[coords]=64
    >>> coords=shift_coords(coords, 0, 1)
    >>> q[coords]=128
    >>> coords=shift_coords(coords, -1, 0)
    >>> q[coords]=32

    >>> plt.imshow(q)

    '''
    index_coords = np.array(list(coords))
    index_coords += np.array([[dx],[dy]])
    return tuple(index_coords.tolist())



def shift_indices(indices,dx,dy, inplace=False):
    ''' index-equivalent to shift_coords() ''' 
    if inplace:
        #shifts indices IN PLACE 
        indices += np.array([[dx],[dy]])
        return indices
    
    return indices + np.array([[dx],[dy]])

def gradient(size=(100,100)):
    f = np.zeros(size)

    rows = f.shape[0]
    for i in list(range(rows)):
        f[:,i] = i  / rows
    return f

def blend(image1, image2, stride=2):
    try:
        assert image1.shape == image2.shape, f'input array shape mismatch'
    except AssertionError as msg:
        print(msg)
        
    x = np.zeros(image1.shape)
    grid = coord_grid(x, stride=stride)
    image_blend = image1.copy()
    image_blend[grid]=image2[grid]
    return image_blend    

def mask(array, mask_width=2,intensity=(0,255)):
    
    N = array.shape[0]
    b = mask_width
    n_pad = int((N-b)/2)
    image_tmp = np.ones((b,b), dtype=np.uint8)
    if isinstance(intensity, type((1,1))):
        lo = intensity[0]
        hi = intensity[1]
        image_tmp = image_tmp * np.random.randint(lo, high=hi, size=image_tmp.shape)
    else:
        image_tmp = image_tmp*intensity

    image = array.copy()
    image[n_pad:n_pad+b,n_pad:n_pad+b] = image_tmp

    return image

# def mask(array, mask_width=2,intensity=255):
    
#     N = array.shape[0]
#     b = mask_width
#     n_pad = int((N-b)/2)
#     image_tmp = np.ones((b,b))*intensity
#     image = array.copy()
#     image[n_pad:n_pad+b,n_pad:n_pad+b] = image_tmp

#     return image
    
def radial_gradient_layers(N, step=1,center='light', noisy=False):
    ''' returns customizable "layers" parameter for add_layers '''
    layers=[]
    if noisy:
        if center == 'dark':
            for i in list(range(N)[2::step]):
                layers.append((N-i,(0,N-i)))

        if center == 'light':
            for i in list(range(N)[2::step]):
                layers.append((N-i,(0,i)))

    else:
        if center == 'dark':
            for i in list(range(N)[2::step]):
                layers.append((N-i,N-i))

        if center == 'light':
            for i in list(range(N)[2::step]):
                layers.append((N-i,i))

    return layers
    

def add_layers(image, layers):    
    ''' overwrites input image array according to layers
    layers is a list of 2-tuples like [(a0,b0),(a1,b1),...]
    ai sets the ith layer to have shape (ai,ai)
    bi sets the ith layer's intensity:
        - bi is an integer in range 0-255 --> uniform intensity
        - bi is a 2-tuple (x,y)  --> intensity made from random numbers in range [x,y]

    Examples:
    ------------------------

    >>> layers=[(46,255),(40,128),(20,(0,255)),(10,10)]
    >>> y = add_layers(np.zeros((50,50)),layers)
    >>> vis_utils.imshow(y)#, normalize=False)

    >>> N=256
    >>> layers=radial_gradient_layers(N, center='light', step=32, noisy=True)
    >>> y = add_layers(np.zeros((N,N)),layers)
    >>> vis_utils.imshow(y)

    >>> N=256
    >>> layers=radial_gradient_layers(N, center='light', step=32)
    >>> y = add_layers(np.zeros((N,N)),layers)
    >>> vis_utils.imshow(y)


    Fun array of alternating light and dark tiles:

        N=256
        layers=radial_gradient_layers(N, center='dark', step=32)
        y_dark = add_layers(np.zeros((N,N)),layers)
        layers=radial_gradient_layers(N, center='light', step=32)
        y_light = add_layers(np.zeros((N,N)),layers)

        R=3
        g = np.hstack([y_light,y_dark])
        for i in range(R):
            g = np.hstack([g,y_light,y_dark])
            
        h = np.hstack([y_dark, y_light])
        for i in range(R):
            h = np.hstack([h,y_dark, y_light])
            
        z = np.vstack([g,h])
        for i in range(R):
            z = np.vstack([z,g,h])
            
        fig = plt.figure(figsize=(30,30))
        plt.axis("off")
        vis_utils.imshow(z)


    '''
    x = image.copy()
    N = x.shape[0]
    for layer in layers:
        L=layer[0]  # length of side
        intensity=layer[1]
        if isinstance(intensity, type((1,1))):
            lo = intensity[0]
            hi = intensity[1]
            tmp = np.random.randint(lo, high=hi, size=(L,L))
        else:
            tmp = np.ones((L,L))*intensity

        n_pad = int((N-L)/2)
        x[n_pad:n_pad+L,n_pad:n_pad+L] = tmp

    return x   

########## Duplicated in utils.py  ######################

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

#########################################################

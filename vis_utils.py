import gc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
from PIL import Image
from scipy import signal
import ffmpeg

#  affine(array, new_min=0., new_max=1.):
#  affine_(array, idx_1=0, value_1=0., idx_2=0, value_2=1.):
#  array_level_view(x, size=10, L=(0,0), color_scale='default', show_axes=True, show_grid=False):
#  autoF32(array, return_params=False):
#  autoU8(array, return_params=False):
#  bit_blast(array):
#  bit_split(array):
#  bits2U8(bit_hot_array, mask=None):
#  bw2rgb(array):
#  config_grid(major_cadence=4, minor_cadence=2, color='#CCCCCC', major_linestyle='--', minor_linestyle=':'):
#  convolve(image, kernel=None):
#  crop(array):
#  crop2file(path_in, path_out):
#  crop_video(video_filename, center_v, center_h, height, width, output_filename):
#  fft2view(fft, log_scale=True, normalize=False, log_multiplier=20, shift=True):
#  float32_to_uint8(array):
#  fourier1D(array):
#  fourier2D(array):
#  get_video_frame(video_filename, frame_index=0):
#  ifourier(fft_shifted):
#  imread(filename):
#  imresize(image, scale=-1, size=(-1,-1)):
#  imshow(array, normalize=True):
#  pil(path_in, path_out):
#  random_color_samples(array, N=4, M=4, scale=20):
#  rgb():
#  scale_max(array, max=1.):
#  scale_mid(array, mid=0.5):
#  scale_min(array, min=0.):
#  show_frame_array(frame_array, num_cols=-1, size=20, titles=None, normalize=True, show_grid=False, show_axes=False, tight_layout=True):
#  show_frame_histograms(frame_array, titles, N, M):
#  show_image_array(figs, titles, num_cols=1, size=30, normalize=True, show_axes=False, tight_layout=True):  ''' 20220709: functionality added: improved automatic spacing 
#  show_image_histograms(figs, titles, N, M):
#  sign2rgb(c, L=(0,0), color_scale='default'):
#  soften_image(image, k=5, w=0.5):
#  spectrum2D(array, log_scale=True, normalize=False, log_multiplier=20, shift=True):
#  uint8_to_float32(array):
#  video_to_array(video_filename):
#  view_bit_layers(bit_layers, figsize=(20, 10)):
#  view_bit_layers_fft(bit_layers, figsize=(20, 10), normalize=False):
#  view_bit_layers_fft_rgb(bit_layers, figsize=(20, 10), normalize=False):
#  view_bit_layers_rgb(rgb_bit_layers, figsize=(15, 20), filepath=None, dpi=96):   
#  view_color_scales():

def bw2rgb(array):
    ''' "3d-ify" input 2d array, shape (N,M) --> (N,M,3) '''
    return np.tile(array.T, (3,1,1)).T

# def imresize(image, scale=-1, size=(-1,-1)):
#     im = Image.fromarray(image.astype(np.uint8))
#     if scale > 0:
#         width, height = im.size
#         newsize = (int(width*scale), int(height*scale))
#         return np.array(im.resize(newsize))
    
#     return np.array(im.resize(size))

def imresize(image, scale=-1, size=(-1,-1)):
    ''' image: numpy array with shape (n, m) or (n, m, 3)
       scale: mulitplier of array height & width (if scale > 0)
       size: (num_rows, num_cols) 2-tuple of ints > 0 (only used if scale <= 0)'''
    
    if (image.shape == size) | (scale == 1):
        return image

    dtype = image.dtype
    if dtype == 'float64':
        dtype = 'float32'

    if image.ndim==2:
        im = Image.fromarray(image)
        if scale > 0:
            width, height = im.size
            newsize = (int(width*scale), int(height*scale))
        else:
            newsize = (size[1],size[0])    #         numpy index convention is reverse of PIL

        return np.array(im.resize(newsize), dtype=dtype)

    if scale > 0:
        height = np.max([1,image.shape[0]])
        width = np.max([1,image.shape[1]])
        newsize = (int(width*scale), int(height*scale))
    else:
        newsize = (size[1],size[0])    

    tmp = np.zeros((newsize[1],newsize[0],3), dtype=dtype)
    for i in range(3):
        im = Image.fromarray(image[:,:,i])
        tmp[:,:,i] = np.array(im.resize(newsize), dtype=dtype)

    return tmp

def imshow(array, normalize=True, show_grid=False, show_axes=False):
    # should probably rewrite this function 

    def normalize_image(array):

        lo = array.ravel().min()
        array -= lo    
        hi = array.ravel().max()
        try:
            assert hi > 0, f'normalize_image cannot map null array to interval [0,1]'
        except AssertionError as msg:
            print(msg)
            return array

        array = array/hi  # "true divide" `array/=hi` causes error (broadcasting issue?) ["TypeError: No loop matching the specified signature and casting was found for ufunc true_divide"]
 
        return array

    if array.ndim==3:
        if array.std(axis=2).ravel().sum() == 0:
            array = array[:,:,0]
        else:
            if normalize:
                plt.imshow(normalize_image(array))
            else:
                plt.imshow(array)
    if array.ndim==2:
        if normalize:
            a = normalize_image(array)
            plt.imshow(a, cmap='gray', vmin=0, vmax=1)
        else:
            if (array.dtype == np.float32) | (array.dtype == np.float64):
                plt.imshow(array, cmap='gray', vmin=0, vmax=1)
            else:
                plt.imshow(array, cmap='gray', vmin=0, vmax=255)

    if not show_axes:
        plt.axis("off")
    if not show_grid:
        plt.grid(visible=False)

def imread(filename):
    frame = img.imread(filename)
    # if frame.ndim == 3:
    #     #frame = frame[:,:,:2]
    #     if frame.std(axis=2).sum()==0.:
    #         frame = frame[:,:,0]

    # print(frame.shape)

    # return np.uint8(frame * 255 / frame.max())
    return frame

def convolve(image, kernel=None):

    if kernel is None:
        kernel = np.ones((2,2), dtype='uint8')

    sigma_v, sigma_h = kernel.shape

    try:
        assert sigma_v%2==1, f'Warning: sigma_v should be odd. Using sigma_v = {sigma_v + 1}.'
    except AssertionError as warning:
        sigma_v += 1
        print(f'***** {warning} *****')
    try:
        assert sigma_h%2==1, f'Warning: sigma_h should be odd. Using sigma_h = {sigma_h + 1}.'
    except AssertionError as warning:
        sigma_h += 1
        print(f'***** {warning} *****')
    
    n_pad = [int(sigma_v/2), int(sigma_h/2)]
    
    
    convolution = signal.convolve(image, kernel, method='fft')[n_pad[0]:n_pad[0]+image.shape[0],n_pad[1]:n_pad[1]+image.shape[1]]
    
    
    return convolution

def rotate(image, direction='cw'):
    N, M, _ = image.shape
    if direction=='cw':
        rgb = tuple(list(map(lambda  x: x[:,:,0].T[:,::-1,None], np.split(image, [1,2], axis=2))))
    else:
        rgb = tuple(list(map(lambda  x: x[:,::-1,0].T[:,:,None], np.split(image, [1,2], axis=2))))

    return np.dstack(rgb)
    # RGB = np.zeros((M,N,3), dtype=np.uint8)
    # for i in range(3):
    #     RGB[:,:,i] = rgb[i][:,:,0]
        
    # return RGB

def soften_image(image, k=5, w=0.5):
    tmp = signal.convolve(image, kernel=np.ones((k,k), dtype='uint8')/(k**2))
    return w*image + (1-w)*tmp

def float32_to_uint8(array):
    array = (255 * array).astype(np.uint8)
    return array

def uint8_to_float32(array):
    array = array.astype(np.float32) / np.float32(255)
    return array

def crop2file(path_in, path_out):
    pic = img.imread(path_in)
    pic = cv.crop(pic)
    img.imsave(path_out, pic)

def pil(path_in, path_out):
    pic = PIL.Image.open(path_in)
    pic = cv.cropframe(pic)
    pic.save(path_out)

def bit_blast(array):
    return np.unpackbits(array).reshape(8, *array.shape)

def bit_split(array):
    return np.unpackbits(array).reshape(*array.shape, 8)

def view_bit_layers(bit_layers, figsize=(20, 10)):
    n_layers = bit_layers.shape[2]   
    fig = plt.figure(figsize=figsize)
    for i in range(n_layers):
        plt.subplot(1, n_layers, i+1)
        plt.imshow(bit_layers[:,:,i], cmap='binary')
        plt.axis("off")
    plt.show()

def view_bit_layers_fft(bit_layers, figsize=(20, 10), normalize=False):
    n_layers = bit_layers.shape[2]   
    fig = plt.figure(figsize=figsize)
    for i in range(n_layers):
        plt.subplot(1, n_layers, i+1)
        plt.imshow(spectrum2D(bit_layers[:,:,i], normalize=normalize), cmap='binary')
        plt.axis("off")
    plt.show()

def view_bit_layers_rgb(rgb_bit_layers, figsize=(15, 20), filepath=None, dpi=96):   
    fig = plt.figure(figsize=figsize, dpi=dpi)
    n_layers = rgb_bit_layers.shape[3]    
    for j in range(n_layers):
        rgb_layers = rgb_bit_layers[:,:,:,j]
        n_colors = rgb_layers.shape[2]
        for i in range(n_colors):
            plt.subplot(n_layers, n_colors, j*n_colors + i+1)
            plt.imshow(rgb_layers[:,:,i], cmap='binary')
            plt.axis("off")
    fig.tight_layout()
    if filepath is not None:
        plt.savefig(filepath, bbox_inches='tight')

# def view_bit_layers_rgb(bit_layers, figsize=(20, 10)):
#     n_colors = bit_layers.shape[2]    
#     for j in range(n_colors):
#         view_bit_layers(bit_layers[:,:,j,:], figsize=figsize)

def view_bit_layers_fft_rgb(bit_layers, figsize=(20, 10), normalize=False):
    n_colors = bit_layers.shape[2]    
    for j in range(n_colors):
        view_bit_layers_fft(bit_layers[:,:,j,:], figsize=figsize, normalize=normalize)

def bits2U8(bit_hot_array, mask=None):
    base2=2**np.arange(8, dtype=np.uint8)[::-1]
    if mask is not None:
    	base2=base2*mask
    return np.dot(bit_hot_array, base2)

# def bits2U8(bit_hot_array):
#     base2=2**np.arange(8, dtype=np.uint8)[::-1]
#     return np.dot(bit_hot_array, base2)

def fft2view(fft, log_scale=True, normalize=False, log_multiplier=20, shift=True):
	f = fft.copy()
	if shift:
		f = np.fft.fftshift(f)

	f = np.abs(f)

	if log_scale:
		f = log_multiplier * np.log(f)

	if normalize:
		f = autoU8(f)

	return (255-f)

def spectrum2D(array, log_scale=True, normalize=False, log_multiplier=20, shift=True):
    array_fft = np.fft.fft2(array)
    return fft2view(array_fft, log_scale=log_scale, \
    	normalize=normalize, log_multiplier=log_multiplier, shift=shift)


def fourier2D(array):
    array_fft = np.fft.fft2(array)
    array_ffts = np.fft.fftshift(array_fft)
    array_ffts_real = np.real(array_ffts)
    array_ffts_imag = np.imag(array_ffts)
    array_ffts_abs = np.abs(array_ffts)
    array_ffts_abs_20log = 20*np.log(array_ffts_abs)
    
    return (array_fft, array_ffts, array_ffts_real, \
            array_ffts_imag, array_ffts_abs, array_ffts_abs_20log)

def fourier1D(array):
    array_fft = np.fft.fft(array)
    array_ffts = np.fft.fftshift(array_fft)
    array_ffts_real = np.real(array_ffts)
    array_ffts_imag = np.imag(array_ffts)
    array_ffts_abs = np.abs(array_ffts)
    array_ffts_abs_20log = 20*np.log(array_ffts_abs)
    
    return (array_fft, array_ffts, array_ffts_real, \
            array_ffts_imag, array_ffts_abs, array_ffts_abs_20log)

def ifourier(fft_shifted):
    fft = np.fft.ifftshift(fft_shifted)
    ifft = np.fft.ifft2(fft)
    ifft_magnitude = np.abs(ifft)
    return fft, ifft, ifft_magnitude

def scale_max(array, max=1.):
    a = array.astype(np.float64)
    a_max = a.max()
    a_min = a.min()
    width_initial = a_max - a_min
    width_target = max - a_min
    a *= width_target / width_initial
    a += (max - a.max())
    return a

def scale_min(array, min=0.):
    a = array.astype(np.float64)
    a_max = a.max()
    a_min = a.min()
    width_initial = a_max - a_min
    width_target = a_max - min

    a *= width_target / width_initial
    a += (min - a.min())
    return a

def scale_mid(array, mid=0.5):
    a = array.astype(np.float64)
    a_max = a.max()
    a_min = a.min()
    a_mid = (a_min + a_max) / 2.
    d = max([abs(a_max-a_mid), abs(a_mid-a_min)])

    a = (mid / d) * (a + d - a_mid)
    return a

def affine_(array, idx_1=0, value_1=0., idx_2=0, value_2=1.):
    if idx_1 < 0:
        idx_1 = 0

    if idx_2 == 0 or idx_2 >= min(array.shape):
        idx_2 = min(array.shape) - 1

    x1 = array[idx_1]
    x2 = array[idx_2]

    scale = (value_2 - value_1) / (x2 - x1)
    shift = (value_1 * x2 - value_2 * x1) / (x2 - x1)

    return scale * array + shift


def affine(array, new_min=0., new_max=1.):

    x_min = array.min()
    x_max = array.max()

    if x_max <= x_min:
        x_max = x_min + 1

    scale = (new_max - new_min) / (x_max - x_min)
    shift = (new_min * x_max - new_max * x_min) / (x_max - x_min)

    return scale * array + shift


def crop(array):
	#same as openCV_utilities.crop() except color -> b&w via np.sum(axis=2)
    if array.ndim==3:
        frame=array.sum(axis=2)
    else:
    	frame=array

    frame_std_col= np.std(frame,axis=0)
    for ileft, std in enumerate(frame_std_col):
        if std > 0:
            break
    for iright, std in enumerate(reversed(frame_std_col)):
        if std > 0:
            break
    frame_std_row= np.std(frame,axis=1)
    for itop, std in enumerate(frame_std_row):
        if std > 0:
            break
    for ibot, std in enumerate(reversed(frame_std_row)):
        if std > 0:
            break    
            
    return array[itop:-(ibot+1),ileft:-(iright+1)]

def autoU8(array, return_params=False):
    
    if str(array.dtype)[0]=='c':
        a = np.abs(array)
    else:
        a = array.copy()
    
    offset = a.min()
    a = a - offset
    a_max = a.max()
    if a_max:
        a = a * (255.0 / a_max)

    a = a.astype(np.uint8)
    if return_params:
    	return a, a_max, offset

    return a

def autoF32(array, return_params=False):
    
    if str(array.dtype)[0]=='c':
        a = np.abs(array)
    else:
        a = array.copy()

    a = a.astype(np.float32)
    offset = a.min()
    a = a - offset
    a_max = a.max()
    if a_max:
        a = a * (1. / a_max)
    
    if return_params:
    	return a, a_max, offset

    return a


def config_grid(major_cadence=4, minor_cadence=2, color='#CCCCCC', major_linestyle='--', minor_linestyle=':'):
    '''
        activate major and minor grid lines 

        Example:
            >>> x = np.arange(0,256,dtype=np.uint8).reshape(16,16)
            >>> fig, ax = plt.subplots(figsize=(5,5))
            >>> plt.imshow(x)
            >>> ax.set_xlim(0,x.shape[0]-1)
            >>> ax.set_ylim(0,x.shape[1]-1)
            >>> config_grid()
            >>> plt.show()
    '''

    ax.xaxis.set_major_locator(MultipleLocator(major_cadence))
    ax.yaxis.set_major_locator(MultipleLocator(major_cadence))
    ax.xaxis.set_minor_locator(AutoMinorLocator(minor_cadence))
    ax.yaxis.set_minor_locator(AutoMinorLocator(minor_cadence))

    ax.grid(which='major', color=color, linestyle=major_linestyle)
    ax.grid(which='minor', color=color, linestyle=minor_linestyle)


def video_to_array(video_filename):
    '''
    Stack video frames
    Returns 4-d numpy array:
        axis-0: video frame number
        axis-1: pixel row index
        axis-2: pixel column index
        axis-3: rgb color channel


    Example use:
        >>> frame_array = video_to_array('D:\VIDEOS\movie_clip.mp4')
        >>> frame_array.shape
        (1001, 1080, 1920, 3)
    '''

    video = cv2.VideoCapture(video_filename)
    
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    #initialize frame stack
    frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)

    i = 0
    while video.isOpened():

        success, frame = video.read()
        if not success:
            break

        frames[i,:,:,:] = frame[:,:,::-1]
        i += 1

    video.release()
    
    return frames


def show_frame_array(frame_array, num_cols=-1, size=20, frame_indices=None, range_start=0, range_end=-1, step=1, titles=None, normalize=True, show_grid=False, show_axes=False, tight_layout=True, return_selected=False):
  '''
    Create image array from video frame stack

    Example use:
        >>> frame_array = video_to_array('D:\VIDEOS\movie_clip.mp4')
        >>> num_frames = frame_array.shape[0]
        >>> show_frame_array(frame_array, num_cols=5, step=int(num_frames/20)) # create 4 x 5 array of images using frames 0, 50, 100, 150, ..., 950, 1000

    Choose overall figure size and number of image columns.
    Automatically formats layout to maximize size of individual images (i.e., minimize gaps)

    tight_layout     (bool, Default=False) [Optional]: Further reduce spacing between images

  '''

  num_frames_total = frame_array.shape[0]
  height = frame_array.shape[1]
  width = frame_array.shape[2]  

  if titles is not None:
       Q = num_frames_total - len(titles)
       if Q > 0:
           _ = [titles.append("") for i in range(Q)]

  else:
      titles = np.arange(num_frames_total)

  if range_end == -1:
      range_end = num_frames_total

  if frame_indices is None:
      frame_indices = range(range_start, range_end, step)

  num_frames_selected = len(frame_indices)

  if num_cols == -1:
      num_cols = min(4, num_frames_selected)

  frame_array_ = frame_array[frame_indices]
  titles_ = titles[frame_indices]

  M = num_cols
  #R = round(num_frames_selected/M + 0.5) # how many rows are needed, in order to have no more than M images per row
  N = round(M * width / (1.2 * height) + 0.5) # given M & individual image dimensions, N is the number of rows corresponding to filled cells. 
  # if N < R, split figures into groups of at most N*M each.
  B = int(num_frames_selected/(N*M)) # number of groups
  x = num_frames_selected/(N*M) - B
  if x > 0:
    B += 1
  
  for k in range(B):
    fig = plt.figure(figsize=(size,size))
    lower_index = k*M*N
    upper_index = min((k+1)*M*N, num_frames_selected)
    for j, subplot_index in enumerate(np.arange(lower_index, upper_index)):
        plt.subplot(N,M,j+1)
        imshow(frame_array_[subplot_index], normalize=normalize)
        
        if show_axes:
            plt.axis("on")
        else:
            plt.axis("off")

        plt.grid(visible=show_grid)


        if titles is not None:
            plt.title(str(titles_[subplot_index]))
        
    if tight_layout:  
      fig.tight_layout()
      
    plt.show()

  if return_selected:
      return frame_array_, titles_

  del frame_array_
  gc.collect()


def show_image_array(figs, num_cols=-1, size=20, titles=None, normalize=True, show_grid=False, show_axes=False, tight_layout=True):
  ''' 20220709: functionality added: improved automatic spacing 
    Choose overall figure size and number of image columns.
    Automatically formats layout to maximize size of individual images (i.e., minimize gaps)

    tight_layout     (bool, Default=False) [Optional]: Further reduce spacing between images

  '''
  if type(figs) != type([]):
      figs = [figs]         # if figs is/contains only one image, not necessary to pass as list (i.e., 'figs' can be either single array OR list of array(s))
  
  if num_cols == -1:
      num_cols = min(4, len(figs))

  M = num_cols
  
  height = figs[0].shape[0]
  width = figs[0].shape[1]
  
  #R = round(len(figs)/M + 0.5) # how many rows are needed, in order to have no more than M images per row
  N = round(M * width / (1.2 * height) + 0.5) # given M & individual image dimensions, N is the number of rows corresponding to filled cells. 
  # if N < R, split figures into groups of at most N*M each.
  B = int(len(figs)/(N*M)) # number of groups
  x = len(figs)/(N*M) - B
  if x > 0:
    B += 1
  
  if titles is None:
      for k in range(B):
        fig = plt.figure(figsize=(size,size))
        for j, figure in enumerate(figs[k*M*N:min((k+1)*M*N, len(figs))]):        
            plt.subplot(N,M,j+1)
            imshow(figure, normalize=normalize)
            [plt.axis("off"),plt.axis("on")][show_axes]
            plt.grid(visible=show_grid)

        if tight_layout:  
          fig.tight_layout()
          
        plt.show()
  else:
      for k in range(B):
        fig = plt.figure(figsize=(size,size))
        for j, (figure, title) in enumerate(zip(figs[k*M*N:min((k+1)*M*N, len(figs))], titles[k*M*N:min((k+1)*M*N, len(figs))])):        
            plt.subplot(N,M,j+1)
            imshow(figure, normalize=normalize)            
            [plt.axis("off"),plt.axis("on")][show_axes]
            plt.grid(visible=show_grid)

            if title is not None:
                plt.title(title)
            
        if tight_layout:  
          fig.tight_layout()
          
        plt.show()




def show_frame_histograms(frame_array, titles, N, M):
    num_frames = frame_array.shape[0]

    if titles is not None:
        Q = num_frames - len(titles)
        if Q > 0:
            _ = [titles.append("") for i in range(Q)]

    fig = plt.figure(figsize=(20,20))
    for j in range(num_frames):
        plt.subplot(N,M,j+1)
        _ = plt.hist(frame_array[j].ravel(),bins=256, log=True)
        #plt.axis("off")
        plt.title(titles[j])
    plt.show()


def show_image_histograms(figs, titles, N, M):

    fig = plt.figure(figsize=(20,20))
    for j, (figure, title) in enumerate(zip(figs, titles)):
        plt.subplot(N,M,j+1)
        _ = plt.hist(figure.ravel(),bins=100, log=True)
        #plt.axis("off")
        plt.title(title)
    plt.show()

def rgb():
    return tuple(np.random.randint(256, size=3, dtype=np.uint8).tolist())

def sign2rgb(c, L=(0,0), color_scale='default'):
    
    color_presets = {}
    color_presets['default'] = {'lo':(0,0,0), 'mid':(255,255,255),'hi':(0,0,255)} #{'lo':(255,0,0), 'mid':(0,255,0), 'hi':(0,0,255)}
    color_presets['byg_light'] = {'lo':(173,213,240), 'mid':(228,240,131), 'hi':(205,220,193)}
    color_presets['orbrv'] = {'lo':(222,111,0),'mid':(123,45,234),'hi':(87,31,44)}
    
    if type(color_scale)==str:
        if color_scale in color_presets.keys():
            colors = color_presets[color_scale]
        else:
            colors = color_presets['default']
    elif type(color_scale)==dict:
        colors = color_scale
    
    def set_colors(p, condition, layer):
        p[:,:,0] = np.where(condition,colors[layer][0],p[:,:,0])
        p[:,:,1] = np.where(condition,colors[layer][1],p[:,:,1])
        p[:,:,2] = np.where(condition,colors[layer][2],p[:,:,2])
        
        return p
        
    h,w = c.shape
    p = np.zeros((h,w,3), dtype=np.uint8)    
    p = set_colors(p, c<L[0], 'lo')
    p = set_colors(p, (c>=L[0])&(c<=L[1]),'mid')
    p = set_colors(p, c>L[1], 'hi')    
    
    return p

def array_level_view(x, size=10, L=(0,0), color_scale='default', show_axes=True, show_grid=False):
    if type(x) == np.ndarray:
        show_image_array(sign2rgb(x, L=L, color_scale=color_scale), size=size, show_axes=show_axes, show_grid=show_grid)
    else:
        from scipy.sparse.dia import dia_matrix

        if type(x) == dia_matrix:
            show_image_array(sign2rgb(x.todense(), L=L, color_scale=color_scale), size=size, show_axes=show_axes, show_grid=show_grid)

def random_color_samples(array, N=4, M=4, scale=20):
    '''    [In]:   s = random_color_samples(c, N=8, M=8, scale=20) '''     
    def sign2rgb(c, L=(0,0), color_scale=0):

        color_presets = {}
        color_presets[0] = {'lo':(255,0,0), 'mid':(0,255,0), 'hi':(0,0,255)}
        color_presets[1] = {'lo':(46,155,223),'mid':(239,5,97),'hi':(181,209,176)}
        color_presets[2] = {'lo':(176,66,129),'mid':(249,153,81),'hi':(13,22,209)}
        color_presets[3] = {'lo':(70,22,68),'mid':(142,209,137),'hi':(198,68,86)}
        color_presets[4] = {'lo':(218,210,197),'mid':(55,55,60),'hi':(24,179,9)}
        color_presets[5] = {'lo':(46,83,140),'mid':(245,70,109),'hi':(56,204,61)}
        color_presets[6] = {'lo':(169,141,31),'mid':(246,175,57),'hi':(75,45,250)}
        color_presets[7] = {'lo':(115,146,105),'mid':(252,123,82),'hi':(9,41,124)}
        color_presets[8] = {'lo':(118,145,175),'mid':(182,230,74),'hi':(220,164,206)}
        color_presets[9] = {'lo':(173,213,240), 'mid':(228,240,131), 'hi':(205,220,193)}
        color_presets[10] = {'lo':(222,111,0),'mid':(123,45,234),'hi':(87,31,44)}

        if type(color_scale)==dict:
            colors = color_scale
        else:
            colors = color_presets[color_scale]

        def set_colors(p, condition, layer):
            p[:,:,0] = np.where(condition,colors[layer][0],p[:,:,0])
            p[:,:,1] = np.where(condition,colors[layer][1],p[:,:,1])
            p[:,:,2] = np.where(condition,colors[layer][2],p[:,:,2])

            return p

        h,w = c.shape
        p = np.zeros((h,w,3), dtype=np.uint8)    
        p = set_colors(p, c<L[0], 'lo')
        p = set_colors(p, (c>=L[0])&(c<=L[1]),'mid')
        p = set_colors(p, c>L[1], 'hi')    

        return p

    def rgb():
        return tuple(np.random.randint(256, size=3, dtype=np.uint8).tolist())

    if M==N:
        figsize=(scale,scale)
    else:
        aspect_ratio = int(M/N) + int(N/M)
        if M<N:
            aspect_ratio += int((N%M)>0)
            figsize=(scale, scale*aspect_ratio)
        elif N<M:
            aspect_ratio += int((M%N)>0)
            figsize=(scale*aspect_ratio, scale)
            
    fig = plt.figure(figsize=figsize)
    samples=[]
    for i in range(N):
        for j in range(M):
            color_scale={}
            color_scale['lo'] = rgb()
            color_scale['mid'] = rgb()
            color_scale['hi'] = rgb()
            samples.append(color_scale)
            plt.subplot(N,M,i*M+j+1)
            plt.imshow(sign2rgb(array,L=(0,0), color_scale=color_scale))
            plt.title(f'({i},{j})  index:{i*M+j}')
            #plt.title(f'{i*N+j}')
            plt.axis("off")
            

    plt.show()

    for k, h in enumerate(samples):
        print(f'{k}:   {h}')
    
    return samples

def view_color_scales():
    mozaic = np.zeros((1,3,3))
    for i in range(11):
        mozaic = np.vstack([mozaic, sign2rgb(np.array([[-1,0,1]]), color_scale=i)])
    mozaic = mozaic[1:,:,:]
    fig = plt.figure(figsize=(40,20))
    imshow(mozaic)




import cv2

def select_video_region(video_filename, start_v, end_v, start_h, end_h, output_filename):
    # Read the video file
    video = cv2.VideoCapture(video_filename)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    try:
        assert all([width > end_h, height > end_v]), f'region dimensions exceed frame dimensions {height} x {width}'
    except AssertionError as msg:
        print(msg)

        return None

    # Check if the video is opened correctly
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    # Create a VideoWriter object for saving the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, video.get(cv2.CAP_PROP_FPS), (end_h-start_h, end_v-start_v))

    
    # Loop over the frames of the video
    while video.isOpened():
        # Read a frame
        success, frame = video.read()

        # Stop looping if the frame is not read correctly
        if not success:
            break

        # Crop the frame and save it to the output video
        out.write(frame[start_v:end_v, start_h:end_h])

    # Release the VideoCapture and VideoWriter objects
    video.release()
    out.release()


def crop_video(video_filename, center_v, center_h, height, width, output_filename):
    # Read the video file
    video = cv2.VideoCapture(video_filename)

    # Check if the video is opened correctly
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    # Create a VideoWriter object for saving the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, video.get(cv2.CAP_PROP_FPS), (width, height))

    # Loop over the frames of the video
    while video.isOpened():
        # Read a frame
        success, frame = video.read()

        # Stop looping if the frame is not read correctly
        if not success:
            break

        # Crop the frame and save it to the output video
        out.write(frame[center_v-height//2:center_v+height//2, center_h-width//2:center_h+width//2])

    # Release the VideoCapture and VideoWriter objects
    video.release()
    out.release()


def get_video_frame(video_filename, frame_index=0):
    # Read the video file
    video = cv2.VideoCapture(video_filename)
    
    # Check if the video is opened correctly
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    # Get the width and height of the video
#    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
 #   height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_index = max(min(num_frames-1, frame_index),0)
  #  frames = np.zeros((num_frames, height, width, 3), dtype=np.uint8)
    # Loop over the frames of the video
    i = 0
    while video.isOpened():
        # Read a frame
        success, frame = video.read()
        if not success:
            break
        
        if i==frame_index:
            break
        else:
            i+=1

# Release the VideoCapture and VideoWriter objects
    video.release()
        
    return frame[:,:,::-1]





def video_adjust_resolution(video_in, height, width, video_out=None):

    size = f'{width}x{height}'
    if video_out is None:
        video_out = '.'.join([video_in.split('.')[:-1]])
        video_out = f'{video_out}_{size}.avi'

    ffmpeg.input(video_in).output(video_out, s=size).run()


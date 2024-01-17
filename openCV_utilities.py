import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import sys, os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

#from UTILITIES import vis_utils as vis
import vis_utils as vis
from IPython.display import display, clear_output

#  crop(array):
#  cvauto(array):
#  cvauto_(array):
#  cvautoRGB(array):
#  cvBIMEF(image_path_in, image_path_out, scale_factor=1, preview=False):
#  cvBIMEF_(image_path_in, image_path_out):
#  cvread(filename, scale_factor=1, window_size=None, grayscale=False, mono=False, transpose=False):
#  cvshow(array, scale_factor=1, window_size=None, title=None, mono=False, transpose=False, output=False):
#  cvVideo(filename):
#  cvVideoBIMEF(video_path_in, video_path_out):
#  cvVideoBIMEF(video_path_in, video_path_out, show_progress=False):
#  cvVideoWrite(filename, frames, fps):
#  cvview(filename, scale_factor=1, title=None, grayscale=False, mono=False, transpose=False):
#  get_fps(filename):
#  get_frame(filename, index):
#  get_frame_size(filename):
#  get_frames(filename):
#  get_video_parameters(filename):



#######################################################################

def cvBIMEF_(image_path_in, image_path_out):
    dPath=r'D:\BACKUPS\NOTE8\PHONE\SCOS'
    fName=r'SCOS_20201107_153553106'
    fExt=r'.jpg'
    finName=''.join([fName, fExt])
    finPath=os.path.join(dPath, finName)
    q = cvread(finPath, scale_factor=1)
    q_ = cv2.intensity_transform.BIMEF(q)

    foutName=''.join([fName, '_', r'.jpg'])
    foutPath=os.path.join(dPath, foutName)
    img.imsave(foutPath, q_[:,:,[2,1,0]])

#######################################################################

def cvBIMEF(image_path_in, image_path_out, scale_factor=1, preview=False):
    q = cvread(image_path_in, scale_factor)
    q_ = cv2.intensity_transform.BIMEF(q)
    img.imsave(image_path_out, q_[:,:,[2,1,0]])
    if preview:
        cvshow(q_)

#######################################################################
def crop(array):
    if len(array.shape)==3:
        frame=cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    else:
        frame = array

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

#######################################################################
def cvauto(array):
    
    if str(array.dtype)[0]=='c':
        a = np.abs(array)
    else:
        a = array.copy()
    
    if a.ndim==3:
        a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    a = a - a.min()
    a_max = a.max()
    if a_max:
        a = a * (255.0 / a_max)
    
    return a.astype(np.uint8)

#######################################################################
def cvautoRGB(array):
    a=np.zeros_like(array, dtype=np.uint8)
    for i in range(3):
        a[:,:,i]=cvauto(array[:,:,i])
    
    return a

#######################################################################
def cvauto_(array):
    
    if str(array.dtype)[0]=='c':
        a = np.abs(array)
    else:
        a = array.copy()
    
    if a.ndim==3:
        a=cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)

    a = a.astype(np.float32)
    offset = a.min()
    a = a - offset
    a_max = a.max()
    if a_max:
        a = a * (1. / a_max)
    
    return a, a_max, offset

#######################################################################
def cvview(filename, scale_factor=1, title=None, grayscale=False, mono=False, transpose=False):
    # cv2.IMREAD_UNCHANGED
    # cv2.IMREAD_GRAYSCALE
    # cv2.IMREAD_COLOR

    if grayscale:
        frame = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    else:
        frame = cv2.imread(filename)
        
    if mono:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if transpose:
        frame = frame.T
        
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#######################################################################	
def cvread(filename, scale_factor=1, window_size=None, grayscale=False, mono=False, transpose=False):
    # cv2.IMREAD_UNCHANGED
    # cv2.IMREAD_GRAYSCALE
    # cv2.IMREAD_COLOR
    if not os.path.isfile(filename):
        print(f'file not found:   {filename}')
        return None

    if grayscale:
        frame = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        mono = False  # cannot have both grayscale and mono True.  mono requires 3 channels, but grayscale conversion leaves only 2 channels
    else:
        frame = cv2.imread(filename)
        
    if mono:
        frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
        
    if transpose:
        frame = frame.T
    
    return frame

#######################################################################	
def cvshow(array, scale_factor=1, window_size=None, title=None, mono=False, transpose=False, output=False):

    frame = np.copy(array)
    if mono:
        if len(frame.shape)==3:
            frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if transpose:
        frame = frame.T
        	
    cv2.imshow(title, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if output:
        return frame

####################################################################### 
def get_frames(filename):
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            yield frame
        else:
            break
    video.release()
    yield None

def cvVideo(filename):
    for frame in get_frames(filename):
        if frame is None:
            break
        cv2.imshow('frame', frame)
        if cv2.waitKey(10) == 27:
            break
        cv2.destroyAllWindows()


####################################################################### 
def get_frame(filename, index):
    counter = 0
    video = cv2.VideoCapture(filename)
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            if counter == index:
                return frame
            counter += 1
        else:
            break
    video.release()
    return None

####################################################################### 
def get_frame_size(filename):
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        return (-1, -1)
    
    frame_width = int(video.get(3))
    frame_height = int(video.get(4))
    video.release()
    
    return (frame_width, frame_height)

####################################################################### 
def get_fps(filename):
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        return -1
    
    fps = int(video.get(5))
    video.release()
    
    return fps

#######################################################################    
def cvVideoWrite(filename, frames, fps):
    frame_width = frames[0].shape[1]
    frame_height = frames[0].shape[0]
    frame_size = (frame_width, frame_height)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # specify video codec
    video_out = cv2.VideoWriter(filename, fourcc, fps, frame_size)

    counter = 0
    for frame in frames:
        if frame is None:
            break
        #cv2.putText(frame, text=str(counter), org=(100,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0,255,0), thickness=3)
        video_out.write(frame)
        counter += 1
        
    video_out.release()
    print(f'{counter} frames written to {filename}')


####################################################################### 
def get_video_parameters(filename):
    video = cv2.VideoCapture(filename)
    if not video.isOpened():
        return -1
    
    dic = {
            'current_frame_msec' : int(video.get(0)),
            'next_frame_index' : int(video.get(1)),
            'progress' : float(video.get(2)),
            'frame_width' : int(video.get(3)),
            'frame_height' : int(video.get(4)),
            'frames_per_second' : float(video.get(5)),
            'fourcc' : int(video.get(6)),
            'number_of_frames' : int(video.get(7)),
            'format_' : int(video.get(8)),
            'capture_mode' : int(video.get(9)),
            'brightness' : int(video.get(10)),
            'contrast' : int(video.get(11)),
            'saturation' : int(video.get(12)),
            'hue' : int(video.get(13)),
            'gain' : int(video.get(14)),
            'exposure' : int(video.get(15)),
            'convert_rgb' : int(video.get(16)),
            'white_balance' : int(video.get(17)),
            'rectification' : int(video.get(18))
            }


    video.release()
    
    return dic


# Parameter Numbering                          EXAMPLE USE:
#                                              video = cv2.VideoCapture(filename)
#                                              frame_width = int(video.get(3))
#                                              video.release()

# 0. CV_CAP_PROP_POS_MSEC Current position of the video file in milliseconds.

# 1. CV_CAP_PROP_POS_FRAMES 0-based index of the frame to be decoded/captured next.

# 2. CV_CAP_PROP_POS_AVI_RATIO Relative position of the video file

# 3. CV_CAP_PROP_FRAME_WIDTH Width of the frames in the video stream.

# 4. CV_CAP_PROP_FRAME_HEIGHT Height of the frames in the video stream.

# 5. CV_CAP_PROP_FPS Frame rate.

# 6. CV_CAP_PROP_FOURCC 4-character code of codec.

# 7. CV_CAP_PROP_FRAME_COUNT Number of frames in the video file.

# 8. CV_CAP_PROP_FORMAT Format of the Mat objects returned by retrieve() .

# 9. CV_CAP_PROP_MODE Backend-specific value indicating the current capture mode.

# 10. CV_CAP_PROP_BRIGHTNESS Brightness of the image (only for cameras).

# 11. CV_CAP_PROP_CONTRAST Contrast of the image (only for cameras).

# 12. CV_CAP_PROP_SATURATION Saturation of the image (only for cameras).

# 13. CV_CAP_PROP_HUE Hue of the image (only for cameras).

# 14. CV_CAP_PROP_GAIN Gain of the image (only for cameras).

# 15. CV_CAP_PROP_EXPOSURE Exposure (only for cameras).

# 16. CV_CAP_PROP_CONVERT_RGB Boolean flags indicating whether images should be converted to RGB.

# 17. CV_CAP_PROP_WHITE_BALANCE Currently unsupported

# 18. CV_CAP_PROP_RECTIFICATION Rectification flag for stereo cameras (note: only supported by DC1394 v 2.x backend currently)

#######################################################################    
def cvVideoBIMEF(video_path_in, video_path_out):
    params = get_video_parameters(video_path_in)
    frame_size = (params['frame_width'], params['frame_height'])
    fps = params['frames_per_second']
    number_of_frames = params['number_of_frames']

    frames = get_frames(video_path_in)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # specify video codec
    video_out = cv2.VideoWriter(video_path_out, fourcc, fps, frame_size)

    counter = 0
    for frame in frames:
        if frame is None:
            break
        frame = cv2.intensity_transform.BIMEF(frame)    
        video_out.write(frame)
        counter += 1
        
    video_out.release()
    print(f'{counter} frames written to {video_path_out}')

# Version with option to show progress in Jupyter/IPython cell:

def cvVideoBIMEF(video_path_in, video_path_out, show_progress=False):

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

    notebook = isnotebook()

    if show_progress & notebook:
        params = get_video_parameters(video_path_in)
        frame_size = (params['frame_width'], params['frame_height'])
        fps = params['frames_per_second']
        number_of_frames = params['number_of_frames']

    frames = get_frames(video_path_in)
    fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # specify video codec
    video_out = cv2.VideoWriter(video_path_out, fourcc, fps, frame_size)

    counter = 0
    for frame in frames:
        if frame is None:
            break
        frame = cv2.intensity_transform.BIMEF(frame)    
        video_out.write(frame)
        counter += 1
        if show_progress & notebook:
            clear_output(wait=True)
            display(f'Frames processed: {counter} / {number_of_frames}')
        
    video_out.release()
    print(f'{counter} frames written to {video_path_out}')

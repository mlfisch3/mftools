import cv2
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import image as img
import sys, os
from UTILITIES import vis_utils as vis
from UTILITIES import openCV_utilities as cvu

#  array_viewer(array, scale_factor=1, window_size=None):
#  array_viewer_1ch(array, scale_factor=1, window_size=None):
#  array_viewer_BIMEF(array, scale_factor=1, window_size=None):
#  array_viewer_BIMEF2(array, scale_factor=1, window_size=None):
#  colorViewer():
#  draw_():
#  drawOnArray(*args):
#  drawOnArray1Ch(*args):
#  fft_studio(array, scale_factor=1, window_size=None):
#  fft_viewer(array, scale_factor=1, window_size=None):
#  fft_viewer_(array):
#  fft_viewer_z(array, scale_factor=1, window_size=None):
#  sampling_demo(M=512, tick_size=1, sample_rate=800, rate_max=1000, tick_size_max=100, freq=300, record_length=100000):
#  showFilter(array, scale_factor=1, output=False):
#  showFilter2(array, scale_factor=1, output=False):

sys.path.insert(1, os.path.join(sys.path[0], '..'))


def array_viewer_BIMEF2(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.dtype != np.uint8:
        frame = np.uint8(frame * 255/frame.max())
    #     frame = np.abs(frame).sum(axis=2)
    #     frame = frame / frame.max()
    # else:
    #     frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))
    limits = '_,_'
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    # cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('exposure ratio', 'parameters', 0, 10000, nothing)
    cv2.createTrackbar('enhancement ratio', 'parameters', 0, 20000, nothing)
    cv2.createTrackbar('a-parameter', 'parameters', 0, 10000, nothing)
    cv2.createTrackbar('b-parameter', 'parameters', 0, 20000, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)

    # window_init = 100
    # level_init = 50
    # window = window_init
    # level = level_init
    # adjust = 0
    reset = 0
    exposure_init = 5000
    enhancement_init = 5000
    a_init = 3293
    b_init = 11258
    exposure = exposure_init
    enhancement = enhancement_init
    a_param = a_init
    b_param = b_init
    # cv2.setTrackbarPos('window', 'parameters', window_init)
    # cv2.setTrackbarPos('level', 'parameters', level_init)
    # cv2.setTrackbarPos('adjust', 'parameters', adjust)
    cv2.setTrackbarPos('exposure ratio', 'parameters', exposure_init)
    cv2.setTrackbarPos('enhancement ratio', 'parameters', enhancement_init)
    cv2.setTrackbarPos('a-parameter', 'parameters', a_init)
    cv2.setTrackbarPos('b-parameter', 'parameters', b_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('parameters', parameter_text)
     #   cv2.putText(parameter_text, limits, (50, 150), font, 4, (0, 0, 255))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            # cv2.setTrackbarPos('window', 'parameters', window_init)
            # cv2.setTrackbarPos('level', 'parameters', level_init)
            # cv2.setTrackbarPos('adjust', 'parameters', 0)
            cv2.setTrackbarPos('exposure ratio', 'parameters', exposure_init)
            cv2.setTrackbarPos('enhancement ratio', 'parameters', enhancement_init)
            cv2.setTrackbarPos('a-parameter', 'parameters', a_init)
            cv2.setTrackbarPos('b-parameter', 'parameters', b_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame
                       

        if any(
                    [
                        exposure != cv2.getTrackbarPos('exposure ratio', 'parameters'),
                        enhancement != cv2.getTrackbarPos('enhancement ratio', 'parameters'),
                        a_param != cv2.getTrackbarPos('a-parameter', 'parameters'),
                        b_param != cv2.getTrackbarPos('b-parameter', 'parameters')
                    ]
                ):
           
            exposure = cv2.getTrackbarPos('exposure ratio', 'parameters')
            enhancement = cv2.getTrackbarPos('enhancement ratio', 'parameters')
            a_param = cv2.getTrackbarPos('a-parameter', 'parameters')
            b_param = cv2.getTrackbarPos('b-parameter', 'parameters')

            frame_show = cv2.intensity_transform.BIMEF2(
                frame, k=exposure/1000, mu=enhancement/10000, a=-1*a_param/10000, b=b_param/10000)
            # if adjust:
            #     if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'),

            #         window = cv2.getTrackbarPos('window', 'parameters')
            #         level = cv2.getTrackbarPos('level', 'parameters')

            #         upper = min([100, level + window / 2]) / 100
            #         lower = max([0, level - window /2]) / 100
            #         frame_show = vis_utils.affine(frame, lower, upper)

            #         limits = ','.join([str(lower), str(upper)])
           

    cv2.destroyAllWindows()
    
    return frame_show

#######################################################################
def array_viewer_BIMEF(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    if frame.dtype != np.uint8:
        frame = np.uint8(frame * 255/frame.max())
    #     frame = np.abs(frame).sum(axis=2)
    #     frame = frame / frame.max()
    # else:
    #     frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))
    limits = '_,_'
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    # cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('exposure ratio', 'parameters', 0, 300, nothing)
    cv2.createTrackbar('enhancement ratio', 'parameters', 0, 300, nothing)
    cv2.createTrackbar('a-parameter', 'parameters', 0, 300, nothing)
    cv2.createTrackbar('b-parameter', 'parameters', 0, 300, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)

    # window_init = 100
    # level_init = 50
    # window = window_init
    # level = level_init
    # adjust = 0
    reset = 0
    exposure_init = 50
    enhancement_init = 100
    a_init = 33
    b_init = 11
    exposure = exposure_init
    enhancement = enhancement_init
    a_param = a_init
    b_param = b_init
    # cv2.setTrackbarPos('window', 'parameters', window_init)
    # cv2.setTrackbarPos('level', 'parameters', level_init)
    # cv2.setTrackbarPos('adjust', 'parameters', adjust)
    cv2.setTrackbarPos('exposure ratio', 'parameters', exposure_init)
    cv2.setTrackbarPos('enhancement ratio', 'parameters', enhancement_init)
    cv2.setTrackbarPos('a-parameter', 'parameters', a_init)
    cv2.setTrackbarPos('b-parameter', 'parameters', b_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('parameters', parameter_text)
     #   cv2.putText(parameter_text, limits, (50, 150), font, 4, (0, 0, 255))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            # cv2.setTrackbarPos('window', 'parameters', window_init)
            # cv2.setTrackbarPos('level', 'parameters', level_init)
            # cv2.setTrackbarPos('adjust', 'parameters', 0)
            cv2.setTrackbarPos('exposure ratio', 'parameters', exposure_init)
            cv2.setTrackbarPos('enhancement ratio', 'parameters', enhancement_init)
            cv2.setTrackbarPos('a-parameter', 'parameters', a_init)
            cv2.setTrackbarPos('b-parameter', 'parameters', b_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame
                       

        if any(
                    [
                        exposure != cv2.getTrackbarPos('exposure ratio', 'parameters'),
                        enhancement != cv2.getTrackbarPos('enhancement ratio', 'parameters'),
                        a_param != cv2.getTrackbarPos('a-parameter', 'parameters'),
                        b_param != cv2.getTrackbarPos('b-parameter', 'parameters')
                    ]
                ):
           
            exposure = cv2.getTrackbarPos('exposure ratio', 'parameters')
            enhancement = cv2.getTrackbarPos('enhancement ratio', 'parameters')
            a_param = cv2.getTrackbarPos('a-parameter', 'parameters')
            b_param = cv2.getTrackbarPos('b-parameter', 'parameters')

            frame_show = cv2.intensity_transform.BIMEF2(
                frame, k=exposure/100, mu=enhancement/100, a=-1*a_param/100, b=b_param/100)
            # if adjust:
            #     if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'),

            #         window = cv2.getTrackbarPos('window', 'parameters')
            #         level = cv2.getTrackbarPos('level', 'parameters')

            #         upper = min([100, level + window / 2]) / 100
            #         lower = max([0, level - window /2]) / 100
            #         frame_show = vis_utils.affine(frame, lower, upper)

            #         limits = ','.join([str(lower), str(upper)])
           

    cv2.destroyAllWindows()
    
    return frame_show

#######################################################################
def array_viewer_1ch(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)

    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==3:
        frame = np.abs(frame).sum(axis=2)
        frame /= frame.max()
    else:
        frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))
    limits = '_,_'
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)
    #cv2.createTrackbar('colormap', 'parameters', 0, 20, nothing)

    window_init = 100
    level_init = 50
    window = window_init
    level = level_init
    reset = 0

    cv2.setTrackbarPos('window', 'parameters', window_init)
    cv2.setTrackbarPos('level', 'parameters', level_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('parameters', parameter_text)
     #   cv2.putText(parameter_text, limits, (50, 150), font, 4, (0, 0, 255))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            cv2.setTrackbarPos('window', 'parameters', window_init)
            cv2.setTrackbarPos('level', 'parameters', level_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame

        if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'):
            window = cv2.getTrackbarPos('window', 'parameters')
            level = cv2.getTrackbarPos('level', 'parameters')

            upper = min([100, level + window / 2]) / 100
            lower = max([0, level - window /2]) / 100
            frame_show = vis.affine(frame, lower, upper)

            limits = ','.join([str(lower), str(upper)])
           

    cv2.destroyAllWindows()
    
    return frame_show

#######################################################################
def array_viewer(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==3:
        frame = np.abs(frame).sum(axis=2)
        frame = frame / frame.max()
    else:
        frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))
    limits = '_,_'
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)

    window_init = 100
    level_init = 50
    window = window_init
    level = level_init
    reset = 0

    cv2.setTrackbarPos('window', 'parameters', window_init)
    cv2.setTrackbarPos('level', 'parameters', level_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('parameters', parameter_text)
     #   cv2.putText(parameter_text, limits, (50, 150), font, 4, (0, 0, 255))
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            cv2.setTrackbarPos('window', 'parameters', window_init)
            cv2.setTrackbarPos('level', 'parameters', level_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame

        if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'):
            window = cv2.getTrackbarPos('window', 'parameters')
            level = cv2.getTrackbarPos('level', 'parameters')

            upper = min([100, level + window / 2]) / 100
            lower = max([0, level - window /2]) / 100
            frame_show = vis.affine(frame, lower, upper)

            limits = ','.join([str(lower), str(upper)])
           

    cv2.destroyAllWindows()
    
    return frame_show

#######################################################################
def fft_viewer(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==3:
        frame = np.abs(frame).sum(axis=2)
        frame /= frame.max()
    else:
        frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('fft')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))

    cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)

    window_init = 100
    level_init = 50
    window = window_init
    level = level_init
    reset = 0

    cv2.setTrackbarPos('window', 'parameters', window_init)
    cv2.setTrackbarPos('level', 'parameters', level_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    fft = np.fft.fft2(frame_show)
    fft_shift = np.fft.fftshift(fft)
    fft_show = cvauto(np.abs(fft_shift))

    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('fft', fft_show)
        cv2.imshow('parameters', parameter_text)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            cv2.setTrackbarPos('window', 'parameters', window_init)
            cv2.setTrackbarPos('level', 'parameters', level_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame
            fft = np.fft.fft2(frame_show)
            fft_shift = np.fft.fftshift(fft)
            fft_show = cvauto(np.abs(fft_shift))

        if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'):
            window = cv2.getTrackbarPos('window', 'parameters')
            level = cv2.getTrackbarPos('level', 'parameters')

            upper = min([100, level + window / 2]) / 100
            lower = max([0, level - window /2]) / 100
            frame_show = vis.affine(frame, lower, upper)
            fft = np.fft.fft2(frame_show)
            fft_shift = np.fft.fftshift(fft)
            fft_show = cvauto(np.abs(fft_shift))

    cv2.destroyAllWindows()
    
    return frame_show, fft

#######################################################################
def fft_viewer_z(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==3:
        frame = np.abs(frame).sum(axis=2)
        frame /= frame.max()
    else:
        frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('fft_abs')
    cv2.namedWindow('fft_real')
    cv2.namedWindow('fft_imag')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))

    cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)

    window_init = 100
    level_init = 50
    window = window_init
    level = level_init
    reset = 0

    cv2.setTrackbarPos('window', 'parameters', window_init)
    cv2.setTrackbarPos('level', 'parameters', level_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    fft = np.fft.fft2(frame_show)
    fft_shift = np.fft.fftshift(fft)
    fft_show_abs = cvauto(np.abs(fft_shift))
    fft_show_real = cvauto(np.real(fft_shift))
    fft_show_imag = cvauto(np.imag(fft_shift))

    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('fft_abs', fft_show_abs)
        cv2.imshow('fft_abs', fft_show_real)
        cv2.imshow('fft_abs', fft_show_imag)
        cv2.imshow('parameters', parameter_text)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            cv2.setTrackbarPos('window', 'parameters', window_init)
            cv2.setTrackbarPos('level', 'parameters', level_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame
            fft = np.fft.fft2(frame_show)
            fft_shift = np.fft.fftshift(fft)
            fft_show_abs = cvauto(np.abs(fft_shift))
            fft_show_real = cvauto(np.real(fft_shift))
            fft_show_imag = cvauto(np.imag(fft_shift))

        if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'):
            window = cv2.getTrackbarPos('window', 'parameters')
            level = cv2.getTrackbarPos('level', 'parameters')

            upper = min([100, level + window / 2]) / 100
            lower = max([0, level - window /2]) / 100
            frame_show = vis.affine(frame, lower, upper)
            fft = np.fft.fft2(frame_show)
            fft_shift = np.fft.fftshift(fft)
            fft_show_abs = cvauto(np.abs(fft_shift))
            fft_show_real = cvauto(np.real(fft_shift))
            fft_show_imag = cvauto(np.imag(fft_shift))

    cv2.destroyAllWindows()
    
    return frame_show, fft

#######################################################################
def fft_studio(array, scale_factor=1, window_size=None):
    def nothing(x):
        pass

    frame = array.copy()
    if window_size is not None:
        scale_factor = window_size / max(frame.shape)
    elif scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    if array.ndim==3:
        frame = np.abs(frame).sum(axis=2)
        frame /= frame.max()
    else:
        frame = frame / frame.max()   
   
    
    cv2.namedWindow('image')
    cv2.namedWindow('fft')
    cv2.namedWindow('parameters')
    parameter_text = np.zeros((2, 256))

    cv2.createTrackbar('window', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('level', 'parameters', 0, 100, nothing)
    cv2.createTrackbar('reset', 'parameters', 0, 1, nothing)

    window_init = 100
    level_init = 50
    window = window_init
    level = level_init
    reset = 0

    cv2.setTrackbarPos('window', 'parameters', window_init)
    cv2.setTrackbarPos('level', 'parameters', level_init)
    cv2.setTrackbarPos('reset', 'parameters', reset)

    frame_show = frame
    fft = np.fft.fft2(frame_show)
    fft_shift = np.fft.fftshift(fft)
    fft_show = cvauto(np.abs(fft_shift))

  #  ifft = np.fft.ifft2(fft_)

    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('fft', fft_show)
        cv2.imshow('parameters', parameter_text)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

        if cv2.getTrackbarPos('reset', 'parameters'):
            cv2.setTrackbarPos('window', 'parameters', window_init)
            cv2.setTrackbarPos('level', 'parameters', level_init)
            cv2.setTrackbarPos('reset', 'parameters', 0)
            frame_show = frame
            fft = np.fft.fft2(frame_show)
            fft_shift = np.fft.fftshift(fft)
            fft_show = cvauto(np.abs(fft_shift))

        if window != cv2.getTrackbarPos('window', 'parameters') or level != cv2.getTrackbarPos('level', 'parameters'):
            window = cv2.getTrackbarPos('window', 'parameters')
            level = cv2.getTrackbarPos('level', 'parameters')

            upper = min([100, level + window / 2]) / 100
            lower = max([0, level - window /2]) / 100
            frame_show = affine(frame, lower, upper)
            fft = np.fft.fft2(frame_show)
            fft_shift = np.fft.fftshift(fft)
            fft_show = cvauto(np.abs(fft_shift))

    cv2.destroyAllWindows()
    
    return frame_show, fft

#######################################################################
def fft_viewer_(array):
    def nothing(x):
        pass

    switch = '0 : OFF\n 1 : ON'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)


    if array.ndim==3:
        frame = np.abs(array).sum(axis=2)
        frame /= frame.max()
    else:
        frame = array / array.max()

    fft = np.fft.fft2(frame)
    fft_shift = np.fft.fftshift(fft)
    fft_shift_real = np.real(fft_shift)
    fft_shift_imag = np.imag(fft_shift)
    fft_shift_abs = np.abs(fft_shift)
    
    #row1 = np.hstack([frame, fft_shift_abs])
    #row2 = np.hstack([fft_shift_real, fft_shift_imag])
    #frame_show = cvauto(np.vstack([row1,  row2]))
    
    frame_show = frame
    fft_show = cvauto(np.abs(fft))
    
    cv2.namedWindow('image')
    cv2.namedWindow('fft')

    while True:
        cv2.imshow('image', frame_show)
        cv2.imshow('fft', fft_show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break

    cv2.destroyAllWindows()
    
    return fft_shift

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
r,g,b = 0,0,0

drawing = False # true if mouse is pressed
mode = True # if True, draw rectangle. Press 'm' to toggle to curve
ix,iy = -1,-1
intensity = 0

#######################################################################
def colorViewer():

    def nothing(x):
        pass

    img = np.zeros((300, 512, 3), np.uint8)
    cv2.namedWindow('image')

    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('R', 'image', 0, 255, nothing)
    cv2.createTrackbar('grayscale', 'image', 0, 255, nothing)

    switch = '0 : GRAY\n 1 : BGR'
    cv2.createTrackbar(switch, 'image', 0, 1, nothing)

    while(1):
        cv2.imshow('image', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break
            
        b = cv2.getTrackbarPos('B', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        r = cv2.getTrackbarPos('R', 'image')
        gray = cv2.getTrackbarPos('grayscale', 'image')
        s = cv2.getTrackbarPos(switch, 'image')
        
        if s == 0:
            img[:] = [gray, gray, gray]
        else:
            img[:] = [b, g, r]
            
    cv2.destroyAllWindows()

    #return img

#######################################################################
def drawOnArray(*args):
    global mode    
   
    def nothing(x):
        pass

    cv2.namedWindow('image')
   # cv2.createTrackbar('Intensity', 'image', 0, 255, nothing)
    cv2.createTrackbar('Radius', 'image', 1, 100, nothing)
    cv2.createTrackbar('B', 'image', 0, 255, nothing)
    cv2.createTrackbar('G', 'image', 0, 255, nothing)
    cv2.createTrackbar('R', 'image', 0, 255, nothing)

    if len(args):
        frame = args[0].copy() 
        if args[0].ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            # frame_max = frame.max()
            # if frame_max:
            #     max_rgb = args[0].max()
            #     frame *= max_rgb / frame_max                
        frame = np.uint8(frame)                
    else: 
        frame = np.zeros((512,512,3), np.uint8)
        

    # mouse callback function
    def draw_circle(event,x,y,flags,param): #, ix=-1, iy=-1, mode=True, drawing=False):
        global ix,iy,drawing,radius,b,g,r # intensity,
        if event == cv2.EVENT_LBUTTONDOWN:
            # intensity = cv2.getTrackbarPos('Intensity', 'image')
            radius = cv2.getTrackbarPos('Radius', 'image')
            b = cv2.getTrackbarPos('B', 'image')
            g = cv2.getTrackbarPos('G', 'image')
            r = cv2.getTrackbarPos('R', 'image')
            drawing = True
            ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.circle(frame,(x,y),radius,(b,g,r),-1)
                else:
                    cv2.rectangle(frame,(ix,iy),(x,y),(b,g,r),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.circle(frame,(x,y),radius,(b,g,r),-1)
            else:
                cv2.rectangle(frame,(ix,iy),(x,y),(b,g,r),-1)
                
    cv2.setMouseCallback('image',draw_circle)
    
    while(1):
        cv2.imshow('image',frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):           # toggle circle or rectangle shape
            mode = not mode
        elif k == 27:
            break
    
    cv2.destroyAllWindows()

    return frame

#######################################################################
def drawOnArray1Ch(*args):
    global mode    
   
    if len(args):
        frame = args[0].copy() 
        if args[0].ndim == 3:
            frame = args[0].sum(axis=2)
            frame_max = frame.max()
            if frame_max:
                max_rgb = args[0].max()
                frame *= max_rgb / frame_max                
        frame = np.uint8(frame)                
    else: 
        frame = np.zeros((512,512), np.uint8)
        
    def nothing(x):
        pass

    cv2.namedWindow('image')
    cv2.createTrackbar('Intensity', 'image', 0, 255, nothing)
    cv2.createTrackbar('Radius', 'image', 1, 100, nothing)
    
    # mouse callback function
    def draw_circle(event,x,y,flags,param): #, ix=-1, iy=-1, mode=True, drawing=False):
        global ix,iy,drawing,intensity,radius
        if event == cv2.EVENT_LBUTTONDOWN:
            intensity = cv2.getTrackbarPos('Intensity', 'image')
            radius = cv2.getTrackbarPos('Radius', 'image')
            drawing = True
            ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.circle(frame,(x,y),radius,(intensity),-1)
                else:
                    cv2.rectangle(frame,(ix,iy),(x,y),(intensity),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.circle(frame,(x,y),radius,(intensity),-1)
            else:
                cv2.rectangle(frame,(ix,iy),(x,y),(intensity),-1)
                
    cv2.setMouseCallback('image',draw_circle)
    
    while(1):
        cv2.imshow('image',frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):           # toggle circle or rectangle shape
            mode = not mode
        elif k == 27:
            break
    
    cv2.destroyAllWindows()

    return frame

#######################################################################
def draw_():
    drawing = False # true if mouse is pressed
    mode = True # if True, draw rectangle. Press 'm' to toggle to curve
    ix,iy = -1,-1
    # mouse callback function
    def draw_circle(event,x,y,flags,param): #, ix=-1, iy=-1, mode=True, drawing=False):
        global ix,iy,drawing,mode
        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            ix,iy = x,y
        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing == True:
                if mode == True:
                    cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),-1)
                else:
                    cv2.circle(frame,(x,y),5,(0,0,255),-1)
        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            if mode == True:
                cv2.rectangle(frame,(ix,iy),(x,y),(0,255,0),-1)
            else:
                cv2.circle(frame,(x,y),5,(0,0,255),-1)
                
    frame = np.zeros((512,512,3), np.uint8)
    cv2.namedWindow('image')
    cv2.setMouseCallback('image',draw_circle)

    while(1):
        cv2.imshow('image',frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('m'):
            mode = not mode
        elif k == 27:
            break
    cv2.destroyAllWindows()

    return frame

#######################################################################
def sampling_demo(M=512, tick_size=1, sample_rate=800, rate_max=1000, tick_size_max=100, freq=300, record_length=100000):
    def nothing(x):
        pass    

    
    cv2.namedWindow('viewer')
    cv2.createTrackbar('freq', 'viewer', 1, rate_max, nothing)
    cv2.createTrackbar('sample_rate', 'viewer', 1, rate_max, nothing)
    cv2.createTrackbar('tick_size', 'viewer', 1, tick_size_max, nothing)
    cv2.setTrackbarPos('freq', 'viewer', freq)
    cv2.setTrackbarPos('sample_rate', 'viewer', sample_rate)
    cv2.setTrackbarPos('tick_size', 'viewer', tick_size)
    
    sample_interval = int(rate_max/sample_rate)
    a = np.zeros((2*M,2*M), dtype=np.uint8)
    
    y_signal = np.full((record_length,), M)
    y_sample = np.full((record_length,), M)
    
    n = 0
    n_updates = 0
    y_index = int(M/2)
    x_index = M
    
    while n < record_length:
        cv2.imshow('viewer', a)
        k = cv2.waitKey(tick_size) & 0xFF
        if k == 27:
            break
    
        if freq != cv2.getTrackbarPos('freq', 'viewer'):
            freq = cv2.getTrackbarPos('freq', 'viewer')
            n = 0
            n_updates = 0
            
        if sample_rate != cv2.getTrackbarPos('sample_rate', 'viewer'):
            sample_rate = cv2.getTrackbarPos('sample_rate', 'viewer')
            sample_interval = int(rate_max/sample_rate)
            n = 0
            n_updates = 0
        
        if tick_size != cv2.getTrackbarPos('tick_size', 'viewer'):
            tick_size = cv2.getTrackbarPos('tick_size', 'viewer')
            n = 0
            n_updates = 0
        
        
        m = n % int(sample_interval)
        #a[y_index, x_index] = 0
        if m:
            intensity = int(255/np.sqrt(m))
            cv2.line(a, (M,M), (x_index, y_index), intensity, 3)
        else:
            cv2.line(a, (M,M), (x_index, y_index), 0, 3)
            y_index = int(M * (1 - 0.5 * np.cos(2 * np.pi * (freq/rate_max) * n)) )
            x_index = int(M * (1 + 0.5 * np.sin(2 * np.pi * (freq/rate_max) * n)) )
            #a[y_index, x_index] = 255
            cv2.line(a, (M,M), (x_index, y_index), 255, 3)
            n_updates += 1
           # print(f'n: {n} updates: {n_updates}')
            
        y_signal[n] = M * (1 - 0.5 * np.cos(2 * np.pi * (freq/rate_max) * n))
        y_sample[n] = y_index
        
        n+=1
        
    cv2.destroyAllWindows()

    return y_signal[:n], y_sample[:n]

#######################################################################    
def showFilter(array, scale_factor=1, output=False):
    def nothing(x):
        pass

    frame = array.copy()
    frame1 = None
    frame2 = None

    if scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    f_init = np.fft.fft2(frame)
    
    cv2.namedWindow('image')
    N_max = 10
    N_step = 10000
    effect_max = 100
    cv2.createTrackbar('N', 'image', 0, N_max, nothing)
    cv2.createTrackbar('effect', 'image', 0, effect_max, nothing)
    #autorange_state = '0 : OFF\n 1 : ON'
    cv2.createTrackbar('autorange', 'image', 0, 1, nothing)
    
    frame_show = frame
    
    N = cv2.getTrackbarPos('N', 'image')
    effect = cv2.getTrackbarPos('effect', 'image')
    autorange = cv2.getTrackbarPos('autorange', 'image')

    while(1):
        cv2.imshow('image', frame_show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break
        
        N_new = cv2.getTrackbarPos('N', 'image')
        if N_new != N:
            print(f'{N} --> {N_new}')
            N = N_new
            f = f_init.copy()
            x = np.random.randint(f.shape[0],size=(N*N_step, 1))
            y = np.random.randint(f.shape[1],size=(N*N_step, 1))
            idx=np.hstack([x,y])
            f[tuple(idx.T.tolist())]=0
            ifft = np.fft.ifft2(f)
            frame1 = np.abs(ifft).astype(np.uint8)
            blur = cv2.pyrDown(frame1)
            blur = cv2.pyrDown(blur)
            blur = cv2.pyrUp(blur)
            blur = cv2.pyrUp(blur)
            blur = blur[:frame.shape[0],:frame.shape[1]]
            effect = cv2.getTrackbarPos('effect', 'image')
            frame2 = frame-np.uint8(effect/effect_max *blur)
            if autorange:
                frame_show = cvauto(frame2)
            else:
                frame_show = frame2

        #    frame_show = np.minimum(frame,frame_show)

        if effect != cv2.getTrackbarPos('effect', 'image'):
            effect = cv2.getTrackbarPos('effect', 'image')
            frame2 = frame-np.uint8(effect*blur)
            if autorange:
                frame_show = cvu.cvauto(frame2)
            else:
                frame_show = frame2

        #    frame_show = np.minimum(frame,frame_show)
        
        if autorange != cv2.getTrackbarPos('autorange', 'image'):
            autorange = cv2.getTrackbarPos('autorange', 'image')
            if autorange:
                if frame2 is None:
                    frame2 = frame
                frame_show = cvauto(frame2)
            else:
                frame_show = frame2

         #   frame_show = np.minimum(frame,frame_show)

    cv2.destroyAllWindows()
    if output:
        return frame_show

#######################################################################
def showFilter2(array, scale_factor=1, output=False):
    def nothing(x):
        pass

    frame = array.copy()
    frame1 = None
    frame2 = None

    if scale_factor != 1:
        width = int(frame.shape[0] * scale_factor)
        height = int(frame.shape[1] * scale_factor)
        dim = (height, width)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)


    f_init = np.fft.fft2(frame)
    
    frame_show = frame
    fft_show = f_init 

    cv2.namedWindow('image')
    cv2.namedWindow('fft')
    N_max = 10
    N_step = 10000
    effect_max = 100
    cv2.createTrackbar('N', 'image', 0, N_max, nothing)
    cv2.createTrackbar('effect', 'image', 0, effect_max, nothing)
    #autorange_state = '0 : OFF\n 1 : ON'
    cv2.createTrackbar('autorange', 'image', 0, 1, nothing)
    
    N = cv2.getTrackbarPos('N', 'image')
    effect = cv2.getTrackbarPos('effect', 'image')
    autorange = cv2.getTrackbarPos('autorange', 'image')

    while(1):
        cv2.imshow('image', frame_show)
        cv2.imshow('fft', fft_show)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:  # ESC key
            break
        
        N_new = cv2.getTrackbarPos('N', 'image')
        if N_new != N:
            print(f'{N} --> {N_new}')
            N = N_new
            f = f_init.copy()
            x = np.random.randint(f.shape[0],size=(N*N_step, 1))
            y = np.random.randint(f.shape[1],size=(N*N_step, 1))
            idx=np.hstack([x,y])
            f[tuple(idx.T.tolist())]=0
            ifft = np.fft.ifft2(f)
            frame1 = np.abs(ifft).astype(np.uint8)
            blur = cv2.pyrDown(frame1)
            blur = cv2.pyrDown(blur)
            blur = cv2.pyrUp(blur)
            blur = cv2.pyrUp(blur)
            blur = blur[:frame.shape[0],:frame.shape[1]]
            effect = cv2.getTrackbarPos('effect', 'image')
            frame2 = frame-np.uint8(effect/effect_max *blur)
            if autorange:
                frame_show = cvauto(frame2)
            else:
                frame_show = frame2

        #    frame_show = np.minimum(frame,frame_show)

        if effect != cv2.getTrackbarPos('effect', 'image'):
            effect = cv2.getTrackbarPos('effect', 'image')
            frame2 = frame-np.uint8(effect*blur)
            if autorange:
                frame_show = cvauto(frame2)
            else:
                frame_show = frame2

        #    frame_show = np.minimum(frame,frame_show)
        
        if autorange != cv2.getTrackbarPos('autorange', 'image'):
            autorange = cv2.getTrackbarPos('autorange', 'image')
            if autorange:
                if frame2 is None:
                    frame2 = frame
                frame_show = cvauto(frame2)
            else:
                frame_show = frame2

         #   frame_show = np.minimum(frame,frame_show)

    cv2.destroyAllWindows()
    if output:
        return frame_show

#######################################################################

# 0 - Original
import numpy as np
import cv2
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

# 1 - Reading and processing one frame at a time
import numpy as np
import cv2

def video_slice_time_series(video_filename, row0, row1, col0, col1):
    video = cv2.VideoCapture(video_filename)
    
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    slice_values = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        slice_values.append(frame[row0:row1, col0:col1,:].max(axis=2).mean(axis=0))

    video.release()
    return np.array(slice_values)


# 1 - Reading and processing one frame at a time
import numpy as np
import cv2

def video_pixel_time_series(video_filename, row, col, channel=0):
    video = cv2.VideoCapture(video_filename)
    
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    channel_values = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        channel_values.append(frame[row, col, channel])

    video.release()
    return np.array(channel_values)


# 2a - Using moviepy 

from moviepy.video.io.VideoFileClip import VideoFileClip
import numpy as np

def video_to_array_moviepy(video_filename):
    video = VideoFileClip(video_filename)
    frame_num = video.fps * video.duration
    frames = np.zeros((frame_num, video.h, video.w, 3))
    for i, frame in enumerate(video.iter_frames()):
        frames[i] = frame
    return frames


# # 2b  - Using pyav

# import av
# import numpy as np

# def video_to_array(video_filename):
#     container = av.open(video_filename)
#     frames = np.zeros((container.streams.video[0].frames, container.streams.video[0].format.height, container.streams.video[0].format.width, 3))
#     for i, packet in enumerate(container.demux()):
#         for frame in packet.decode():
#             if frame.type != 'video':
#                 continue
#             frames[i] = frame.to_rgb().to_ndarray()
#             i += 1
#     return frames


# 3 - Using multi-threading or multi-processing:

import cv2
from multiprocessing import Process, Queue

def process_frame(queue, frame, row, col, channel=0):
    value = frame[row, col, channel]
    queue.put(value)

def video_pixel_time_series_multiprocess(video_filename, row, col, channel=0):
    video = cv2.VideoCapture(video_filename)
    
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    processes = []
    queue = Queue()
    values = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break
        
        p = Process(target=process_frame, args=(queue, frame))
        processes.append(p)
        p.start()
        
    for p in processes:
        p.join()
        
    while not queue.empty():
        values.append(queue.get())

    video.release()
    return np.array(values)



# 4 - Using built-in python libraries such as imageio

import imageio
import numpy as np

def video_pixel_time_series_imageio(video_filename, row, col, channel=0):
    with imageio.get_reader(video_filename) as video:
        metadata = video.get_meta_data()
        num_frames = int(metadata['nframes'])
        values = []
        for i in range(num_frames):
            frame = video.get_data(i)
            values.append(frame[row, col, channel])
    return np.array(values)


# 5 - It is possible to use CUDA (Compute Unified Device Architecture) to improve the performance of the code, specifically to accelerate the computation-intensive parts of the code such as image processing. 
# Here's an example of how you can use CUDA to extract the values of a particular pixel over time:

import numpy as np
import cv2
import cupy as cp

def video_pixel_time_series_cupy(video_filename, row, col, channel=0):
    video = cv2.VideoCapture(video_filename)
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    values = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        frame_cp = cp.array(frame)
        value = frame_cp[row, col, channel]
        values.append(value.get())

    video.release()
    return np.array(values)


# 6 - Using CUDA: This function loads the entire video to GPU memory and calculate a histogram of pixel intensities for each frame using GPU parallelization

import numpy as np
import cv2
import cupy as cp

def video_to_histograms_cuda(video_filename):
    video = cv2.VideoCapture(video_filename)
    if not video.isOpened():
        raise IOError("Cannot open the input video file")

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = cp.zeros((num_frames, height, width, 3), dtype=np.uint8)
    histograms = cp.zeros((num_frames, 256), dtype=np.uint32)

    for i in range(num_frames):
        success, frame = video.read()
        if not success:
            break

        frames[i] = cp.asarray(frame)
        histograms[i] = cp.histogram(frames[i], bins=256, range=(0,256))[0]

    video.release()
    return cp.asnumpy(histograms)



# Efficient downsampling using opencv

import cv2

def downsample_video(input_filename, output_filename):
    video_capture = cv2.VideoCapture(input_filename)
    width = int(video_capture.get(3))
    height = int(video_capture.get(4))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_filename, fourcc, video_capture.get(5), (width//2, height//2), True)

    while True:
        success, frame = video_capture.read()
        if not success:
            break
        frame = cv2.resize(frame, (width//2, height//2))
        video_writer.write(frame)

    video_capture.release()
    video_writer.release()


# Efficient downsampling using ffmpeg

import subprocess

def downsample_video(input_filename, output_filename):
    subprocess.run(["ffmpeg", "-i", input_filename, "-vf", "scale=iw/2:ih/2", output_filename])


########################
### NOT TESTED YET:  ###
########################

from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from numpy.random import random

# Generate random data for 500 histograms
#data = [random(size=256) for _ in range(500)]
def plot_histograms(histograms, ncols=10):
    # Create a list of figures for each histogram
    nhistograms = len(histograms)
    nrows = int(nhistograms/ncols)
    figures = [figure(title=f"Histogram {i}", tools="pan,box_zoom,reset") for i in range(nhistograms)]

    # Add a histogram to each figure
    for i, fig in enumerate(figures):
        source = ColumnDataSource(data={'x': histograms[i]})
        fig.quad(bottom=0, top=1, left='x', right='x+0.01', source=source,
                 fill_color="navy", line_color="white", alpha=0.5)
        fig.xaxis.axis_label = 'x'
        fig.yaxis.axis_label = 'Frequency'

    # Create a grid plot of the figures
    p = gridplot(figures, ncols=ncols)

    # Show the plot
    show(p)


from bokeh.layouts import gridplot
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from numpy.random import random

# Generate random data for 500 histograms
data = [random(size=256) for _ in range(500)]

def view_histogram_box_plot(histograms):
    # Create a figure
    p = figure(title='Histograms', tools='box_zoom,reset')

    # Add a box plot of the histograms to the figure
    for i, histogram in enumerate(histograms):
        source = ColumnDataSource(data={'x': histogram, 'y': [i]*len(histogram)})
        p.segment(x0='x', y0='y', x1='x', y1='y+1', source=source, line_color='navy')
        p.circle(x='x', y='y', source=source, size=3, color='navy', alpha=0.5)

    p.xaxis.axis_label = 'Value'
    p.yaxis.axis_label = 'Histogram'
    p.ygrid.grid_line_color = None

    # Show the plot
    show(p)

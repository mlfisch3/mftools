import pandas as pd
import numpy as np
from matplotlib import image as img
from matplotlib import pyplot as plt
import sys, os
from UTILITIES import vis_utils as vu
import copy
from UTILITIES import image_stat_utils as isu

#  flip_faces(facialpoints_df):
#  image_similarities(facialpoints_df, j, idxs):
#  overlay_marks(facialpoints_df, n):
#  show_face(facialpoints_df, n):
#  show_faces(figs, numbers, N, M, autoscale=True):


sys.path.insert(1, os.path.join(sys.path[0], '..'))
#def mirror_horizontal

def show_face(facialpoints_df, n):
    plt.imshow(facialpoints_df.iloc[n,30], cmap='gray')
    for i in range(1,31,2):
        plt.plot(facialpoints_df.iloc[n,i-1], facialpoints_df.iloc[n,i], 'rx')
    plt.title(f'{n}')

def overlay_marks(facialpoints_df, n):
    for i in range(1,31,2):
        plt.plot(facialpoints_df.iloc[n,i-1], facialpoints_df.iloc[n,i], 'rx')


def show_faces(figs, numbers, N, M, autoscale=True):
    ''' 
    example: show 8 x 8 array of randomly chosen faces

    >>> show_faces(facialpoints_df, np.random.randint(1,len(facialpoints_df), size=(64,)), 8,8)
    '''

    fig = plt.figure(figsize=(20,20))
    for j, num in enumerate(numbers):
        plt.subplot(N,M,j+1)
        vu.imshow(figs.iloc[num].Image, autoscale=autoscale)
        for j in range(1,31,2):
            plt.plot(figs.iloc[num,j-1], figs.iloc[num,j], 'rx')
        plt.title(f'{num}')
        plt.axis("off")
    plt.show()

def flip_faces(facialpoints_df):
    fp_h_df = copy.copy(facialpoints_df)
    fp_v_df = copy.copy(facialpoints_df)
    V=facialpoints_df.iloc[0,-1].shape[0]
    H=facialpoints_df.iloc[0,-1].shape[1]

    fp_h_df.iloc[:,:-1:2] = H-fp_h_df.iloc[:,:-1:2]
    fp_h_df.Image = fp_h_df.Image.apply(lambda x: np.flip(x, axis=1))
    fp_v_df.iloc[:,1:-1:2] = V-fp_v_df.iloc[:,1:-1:2]
    fp_v_df.Image = fp_v_df.Image.apply(lambda x: np.flip(x, axis=0))

    return fp_h_df, fp_v_df


entropies = [isu.entropy(facialpoints_df.iloc[i].Image) for i in girl_idxs]

girl_idxs_ent = [girl_idxs[i] for i in np.argsort(entropies).tolist()]

def image_similarities(facialpoints_df, j, idxs):
    # apply autoscaling?
    kl=[]
    xent=[]
    joint=[]
    cond=[]
    mut=[]
    var=[]
    nvar=[]
    for i in idxs:
        kl.append(isu.KL(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        xent.append(isu.xentropy(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        joint.append(isu.joint_entropy(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        cond.append(isu.conditional_entropy(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        mut.append(isu.mutual_information(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        var.append(isu.variation_of_information(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        nvar.append(isu.normalized_variation_of_information(facialpoints_df.iloc[j].Image, facialpoints_df.iloc[i].Image))
        
    return kl, xent, joint, cond, mut, var, nvar

# example: plot similarities to image 20
# stats =  image_similarities(facialpoints_df, 20, idxs)
# u.multiplot(np.arange(len(idxs)), list(stats))


# blur face
# kernel = np.ones((5,5), dtype='uint8')
# tmp = vu.convolve(facialpoints_df.iloc[0,30], kernel=kernel)
# tmp = vu.float32_to_uint8(u.autoscale_array(tmp))
# vu.imshow(tmp)
# fu.overlay_marks(facialpoints_df, n)

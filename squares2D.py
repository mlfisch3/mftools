import numpy as np

#  add_layers(image, layers):    
#  mask(array, mask_width=2,intensity=255):


def mask(array, mask_width=2,intensity=255):
    
    N = array.shape[0]
    b = mask_width
    n_pad = int((N-b)/2)
    image_tmp = np.ones((b,b))*intensity
    image = array.copy()
    image[n_pad:n_pad+b,n_pad:n_pad+b] = image_tmp

    return image
    
def add_layers(image, layers):    
    
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

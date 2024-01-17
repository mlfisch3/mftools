import numpy as np

  
def bits2U16(bit_hot_array, mask=None):
    base2=2**np.arange(16, dtype=np.uint16)[::-1]
    if mask is not None:
        base2=base2*mask
    return np.dot(bit_hot_array, base2)


def bit_split(array):
    return np.unpackbits(array).reshape(*array.shape, 8)


def bit_concat(p, q):
  if p.dtype != 'uint8':
      if p.dtype.name[:5] == 'float':
          p = 255 * p
      p = p.astype(np.uint8)
  if q.dtype != 'uint8':
      if q.dtype.name[:5] == 'float':
          q = 255 * q
      q = q.astype(np.uint8)
      
  p_ = bit_split(p.flatten())
  q_ = bit_split(q.flatten())
  pq = np.hstack([p_,q_])
  return bits2U16(pq)


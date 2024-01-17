import numpy as np
from scipy.sparse import find
from scipy.sparse.linalg import cg, spsolve, spilu, LinearOperator, use_solver


#  match_z(dict_z, coor_z):
#  solve_sparse_system(A, B, method='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, x0=None):
#  sparse_diag(column_array, diag_index, R, C):
#  sparse_indexionary(sparse_array):
#  sparse2full(sparse_array, r, c):
#  tuplify_inner(x):
#  z_sparse(s, shift=0):

def solve_sparse_system(A, B, method='cg', CG_prec='ILU', CG_TOL=0.1, LU_TOL=0.015, MAX_ITER=50, FILL=50, x0=None):
    """
    
    Solves for x = b/A  [[b is vector(B)]]
    A can be sparse (csc or csr) or dense
    b must be dense
    
   """    

    r, c = B.shape
    
    b = B.ravel(order='F').astype(np.float32)
    
    N = A.shape[0]
    if method == 'cg': 
        if CG_prec == 'ILU':
            A_ilu = spilu(A.tocsc(), drop_tol=LU_TOL, fill_factor=FILL)
            M = LinearOperator(shape=(N, N), matvec=A_ilu.solve, dtype='float32')
        else:
            M = None
        if x0 is None:
            x0 = b  # initial guess. correct only if A is identity.  if x is expected to resemble b, it's a better guess than a draw from any distribution
        return cg(A, b, x0=x0, tol=CG_TOL, maxiter=MAX_ITER, M=M)[0].astype(np.float32).reshape(r,c, order='F')

    elif method == 'direct':
        use_solver( useUmfpack = False ) # use single precision
        return spsolve(A, b).astype(np.float32).reshape(r,c, order='F')


###### additional functions for manipulating and comparing sparse matrices   ##################

complex_vec = np.vectorize(np.complex)

def z_sparse(s, shift=0):
    s_ = find(s)
    return complex_vec(shift+s_[0], shift+s_[1]), s_[2]

def match_z(dict_z, coor_z):
    if len(coor_z)>0:
        val = dict_z.get(coor_z)
        if val is not None:
            return coor_z, val

match_z_vec = np.vectorize(match_z)


def sparse_indexionary(sparse_array):
    ill_coors_1, ill_vals = z_sparse(sparse_array, shift=1)  # NOTE:  shift indices to count from 1 instead of 0
    ill_coors_1_set = set(ill_coors_1)
    ill = {k:v for (k,v) in zip(ill_coors_1,ill_vals)}


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


def tuplify_inner(x):
    return list(map(tuple, x.tolist()))

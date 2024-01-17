
#  multiplot(index, list_of_arrays):
#  save_proc_images(dpath, out, run_id):
#  show_image_array(figs, titles, N, M, autoscale=True):
#  show_image_histograms(figs, titles, N, M):
#  show_progress(figs, titles):
#  test_cg(illumination_map, b, CG_TOL=0.1, LU_TOL=0.05, MAX_ITER=50, FILL=50):


figs = [[image,image_maxRGB],[dt0_v, dt0_h], [kernel_v, kernel_h], [kernel_v_abs, kernel_h_abs], [wx, wy], [wx1, wy1],[wx2, wy2]] titles = [['image','image_maxRGB'],['dt0_v','dt0_h'],['kernel_v','kernel_h'],['kernel_v_abs','kernel_h_abs'],['wx','wy'],['wx1','wy1'],['wx2','wy2']]

def multiplot(index, list_of_arrays):
    
    q = tuple()
    for t in list_of_arrays:
            q = q + (index,t)
            
    _ = plt.plot(*q)


def show_image_array(figs, titles, N, M, autoscale=True):

    fig = plt.figure(figsize=(20,20))
    for j, (figure, title) in enumerate(zip(figs, titles)):
        plt.subplot(N,M,j+1)
        imshow(figure, autoscale=autoscale)
        plt.axis("off")
        plt.title(title)
    plt.show()

def show_image_histograms(figs, titles, N, M):

    fig = plt.figure(figsize=(20,20))
    for j, (figure, title) in enumerate(zip(figs, titles)):
        plt.subplot(N,M,j+1)
        _ = plt.hist(figure.ravel(),bins=100, log=True)
        #plt.axis("off")
        plt.title(title)
    plt.show()


    
def show_progress(figs, titles):
    
    for i in range(len(figs)):
        fig = plt.figure(figsize=(20,20))
        for j, (figure, title) in enumerate(zip(figs[i], titles[i])):
            plt.subplot(2,4,j+1)
            imshow(figure, autoscale=False)
            plt.axis("off")
            plt.title(title)
        for j, (figure, title) in enumerate(zip(figs[i], titles[i])):
            plt.subplot(2,4,j+3)
            imshow(figure, autoscale=True)
            plt.axis("off")
            plt.title(title+' [[auto]]')
        plt.show()


dpath = r'C:\GIT_REPOS\BIMEF\EXPORTED'

def save_proc_images(dpath, out, run_id):
    
    for i in range(len(out)):
        if type(out[i]) == type(np.ones(1)):
            if out[i].ndim==1:
                continue
            if out[i].dtype == 'bool':
                imout = out[i].astype(np.float32)
            else:
                imout = out[i]

            name = names[i].replace(':','_')
            fname = run_id + '--' + name + '.jpg'
            fpath = os.path.join(dpath, fname)
            img.imsave(fpath, np.where(imout>1,1,imout))
            fname =  run_id + '--' + name + '_auto.jpg'
            fpath = os.path.join(dpath, fname)
            img.imsave(fpath, normalize_array(imout))       




CG_TOL=0.1
LU_TOL=0.05
INIT='uniform'
MAX_ITER=50
FILL=50


if INIT=='normal':
    x0 = np.random.randn(N)
elif INIT=='uniform':
    x0 = np.random.random(N)

def test_cg(illumination_map, b, CG_TOL=0.1, LU_TOL=0.05, MAX_ITER=50, FILL=50):
    times=[]
    times.append(datetime.now())
    ill_ilu_d1 = spilu(illumination_map, drop_tol=LU_TOL, fill_factor=FILL) 
    times.append(datetime.now())
    M_d1 = LinearOperator(shape=(N, N), matvec=ill_ilu_d1.solve)
    times.append(datetime.now())
    c_d1, info_d1 = cg(illumination_map, b, x0=x0, tol=CG_TOL, maxiter=MAX_ITER, M=M_d1)
    times.append(datetime.now())
    
    return times, info_d1, c_d1

np.set_printoptions(precision=5)
tols = np.r_[0.01:0.3:20j]


timesets=[]
infos=[]
latent_images=[]

for tol in tols:
    print(f'{tol}')
    timeset, info, latent_image = test_cg(illumination_map, b, LU_TOL=tol)
    print(f'{(timeset[-1]-timeset[0]).microseconds}')
    timesets.append(timeset)
    infos.append(info)
    latent_images.append(latent_image.reshape(591,603).T)


show_image_array(latent_images, tols_str, 5,4)
show_image_array(latent_images, tols_str, 5,4, autoscale=False)
show_image_histograms(latent_images, tols_str, 5,4)

import numpy as np

def TwoDPCA(imgs,p):
    a,b,c = imgs.shape
    average = np.zeros((b,c))
    for i in range(a):
        average += imgs[i,:,:]/(a*1.0)
    G_t = np.zeros((c,c))
    for j in range(a):
        img = imgs[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    w,v = np.linalg.eigh(G_t)
    w = w[::-1]
    v = v[::-1]
    for k in range(c):
        alpha = sum(w[:k])*1.0/sum(w)
        if alpha >= p:
            u = v[:,:k]
            break
    return u

def TTwoDPCA(imgs,p):
    u = TwoDPCA(imgs,p)
    a1,b1,c1 = imgs.shape
    img = []
    for i in range(a1):
        temp1 = np.dot(imgs[i,:,:],u)
        img.append(temp1.T)
    img = np.array(img)
    uu = TwoDPCA(img,p)
    return u,uu

def PCA2D_2D(samples, row_top, col_top):
    '''samples are 2d matrices'''
    size = samples[0].shape
    # m*n matrix
    mean = np.zeros(size)

    for s in samples:
        mean = mean + s

    # get the mean of all samples
    mean /= float(len(samples))

    # n*n matrix
    cov_row = np.zeros((size[1],size[1]))
    for s in samples:
        diff = s - mean;
        cov_row = cov_row + np.dot(diff.T, diff)
    cov_row /= float(len(samples))
    row_eval, row_evec = np.linalg.eig(cov_row)
    # select the top t evals
    sorted_index = np.argsort(row_eval)
    # using slice operation to reverse
    X = row_evec[:,sorted_index[:-row_top-1 : -1]]

    # m*m matrix
    cov_col = np.zeros((size[0], size[0]))
    for s in samples:
        diff = s - mean;
        cov_col += np.dot(diff,diff.T)
    cov_col /= float(len(samples))
    col_eval, col_evec = np.linalg.eig(cov_col)
    sorted_index = np.argsort(col_eval)
    Z = col_evec[:,sorted_index[:-col_top-1 : -1]]

    return X, Z


samples = []
for i in range(1,6):
    im = Image.open('image/'+str(i)+'.png')
    im_data  = np.empty((im.size[1], im.size[0]))
    for j in range(im.size[1]):
        for k in range(im.size[0]):
            R = im.getpixel((k, j))
            im_data[j,k] = R/255.0
    samples.append(im_data)

X, Z = PCA2D_2D(samples, 90, 90)

res = np.dot(Z.T, np.dot(samples[0], X))
res = np.dot(Z, np.dot(res, X.T))

row_im = Image.new('L', (res.shape[1], res.shape[0]))
y=res.reshape(1, res.shape[0]*res.shape[1])

row_im.putdata([int(t*255) for t in y[0].tolist()])
row_im.save('X.png')
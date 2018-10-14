# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm

#load data
def loadData(dirName, count=""):
    originData = pd.read_csv(dirName)
    m,n = originData.shape
    if not m:
        print "data empty!"
        return False
    if not count:
        count = m
    #iloc 可以让pd实现基于索引进行数据选取,注意前开，后毕的集合原则
    labels = originData.iloc[0:count, :1].values
    images = originData.iloc[0:count, 1:].values
    return images, labels

def imageShow(img, imgTitle=""):
    plt.imshow(img, cmap="gray")
    plt.title(imgTitle)
    plt.colorbar()
    plt.show()

def imgHistShow(img):
    plt.hist(img)
    plt.show()

def preData(train_images):
    #归一
    train_images = train_images/255.0
    return train_images

def calcW(train_images, p, t1, t2):
    m, n = train_images.shape
    if  t1*t2 != n:
        print "error t1 and t2!"
    train_images = train_images.reshape(m, t1, t2)
    W = twoDPCA(train_images, p)
    return W

def transferData(data,W, t1, t2):
    nums = data.shape[0]
    data = data.reshape(nums, t1, t2)
    data = np.dot(data,W)
    m,t1, t2 = data.shape
    data = data.reshape(nums, -1)
    return data, t1, t2

def writeResult(results):
    df = pd.DataFrame(results)
    df.index.name = 'ImageId'
    df.index += 1
    df.columns = ['Label']
    df.to_csv('submission.csv', header=True)
    return df

def twoDPCA(samples,p):
    a,b,c = samples.shape
    #均值
    average = np.zeros((b,c))
    for i in range(a):
        average += samples[i,:,:]/(a*1.0)
    #协方差矩阵
    G_t = np.zeros((c,c))
    for j in range(a):
        img = samples[j,:,:]
        temp = img-average
        G_t = G_t + np.dot(temp.T,temp)/(a*1.0)
    #v特征值和w特征向量
    v,w = np.linalg.eigh(G_t)
    #w = w[::-1]
    #w.sort(axis=-1)
    order = np.argsort(-v)
    normal = w[:, order]
    total = 0
    for i in range(c):
        k = order[i]
        total += v[k]
        alpha = total*1.0/sum(v)
        if alpha >= p:
            u = normal[:,:i]
            break
    return u

def calcW2(train_images, p, size1, size2):
    #2D-2D-PCA
    W = calcW(train_images, p, size1, size2)
    m, n = train_images.shape
    train_images, t1, t2 = transferData(train_images, W, size1, size2)
    newData = []
    for i in range(m):
        temp = train_images[i,:].T
        newData.append(temp)
    newData = np.array(newData)
    W1 = calcW(newData, p, t1, t2)
    return W1

def digitRecogtion():
    #image size
    size1= 28
    size2= 28
    trainDataDir = './data/train.csv'
    images, labels = loadData(trainDataDir)
    #images = labeled_images.iloc[0:5000,1:]
    #labels = labeled_images.iloc[0:5000,:1]
    train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)
    #show img
    #img = trainData.iloc[8].as_matrix()
    #img = img.reshape((28, 28))
    #imageShow(img)
    #imgHistShow(img)97.94;98.03
    #data pre
    train_images = preData(train_images)
    test_images = preData(test_images)
    W = calcW(train_images, 0.85, size1, size2)
    #W2 = calcW2(train_images, 0.85, size1, size2)

    train_images, t1, t2 = transferData(train_images, W, size1, size2)
    #train_images, t1, t2 = transferData(train_images, W2, t1, t2)

    test_images, t1, t2 = transferData(test_images, W, size1, size2)
    #test_images, t1, t2 = transferData(test_images, W2, t1, t2)
    #train mode    C=7.0, gamma=0.009
    clf = svm.SVC(C=7.0, gamma=0.009)
    clf.fit(train_images, train_labels.ravel())
    precise = clf.score(test_images,test_labels)
    print precise
    #predict
    testDataDir = "./data/test.csv"
    test_data = pd.read_csv('./data/test.csv').values
    test_data = preData(test_data)
    test_data, t1, t2 =transferData(test_data, W, size1, size2)
    #test_data, t1, t2 = transferData(test_data, W2, t1, t2)
    result = clf.predict(test_data)
    print "predict result :", result[1:10]

    df = writeResult(result)
    #print df
if __name__ == "__main__":
    digitRecogtion()
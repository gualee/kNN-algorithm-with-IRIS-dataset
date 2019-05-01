# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 17:58:41 2018
@author: 段昊
"""
import numpy as np
import operator
import time

def loadDataset(filename):
        fr = open(filename)                       #打開檔案
        numberOfLines = len(fr.readlines())       #取得training instance
        #print(numberOfLines)                     #一共有178個instance
        Mat = np.zeros((numberOfLines, 13))       #建立欄位13個的零矩陣
        classLabel = []                           #建立分類標籤串列
        fr = open(filename)  
        index = 0
        
        for line in fr.readlines():               #讀取每一個instance 
                line = line.strip()               #去掉頭尾字串，預設為空白
                listFromLine = line.split('\t');  #以tab的長度來分開
                Mat[index,:] = listFromLine[1:]   #從第一個特徵(不含class標籤)開始放進Mat矩陣
                classLabel.append(int(listFromLine[0]))#把第一欄class放進label list內
                index += 1  
        fr.close()
        return Mat, classLabel

def dataNorm(dataSet):
        minVals = dataSet.min(0)                     #在numpy中，選0為對每個instance取每個column最小值
        maxVals = dataSet.max(0)                     #取最大值
        ranges = maxVals - minVals                   #取出範圍
        normData = np.zeros(np.shape(dataSet))       #建立維度和dataSet一樣大小的矩陣
        m = dataSet.shape[0]                         #取出dataset有幾列
        normData = dataSet - np.tile(minVals, (m,1)) #將每個 element的每個column減掉該column最小值
        normData = normData / np.tile(ranges, (m,1)) #正規化
        return normData

def knn(index, dataSet, labels, k):  
        dataSetSize = dataSet.shape[0]
                                                            #利用歐式距離計算距離  
        diffMat = np.tile(index, (dataSetSize, 1)) - dataSet   
        DiffMat = diffMat ** 2                              #將每個元素平方  
        Distances = DiffMat.sum(axis=1)                     #加總將matrix裡的元素  
        distances = Distances ** 0.5                        #取平方根
        #print("distances::::", sorted(distances))
        sortedDistIndices = distances.argsort()             #由小到大排序distance的索引值
        #sortedDistIndices = sorted(distances)
        #print(sortedDistIndices)
                                                            #選擇K個最短的距離，並決定大部分屬於哪一類  
        classCount = {}  
        for i in range(k):  
                voteIlabel = labels[sortedDistIndices[i]]  
                classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1  
        sortedClassCount = sorted(classCount.items(), key = operator.itemgetter(1), reverse=True)  
        #print(sortedClassCount[0][0])
        return sortedClassCount[0][0]

def main():
        DataMat, labels = loadDataset('C:\\Users\\User\\wine.txt')   #讀取資料集
        normMat = dataNorm(DataMat)                                  #執行正規化
        #np.random.shuffle(normMat)                                  #攪亂資料集
        m = normMat.shape[0]                                         #資料集筆數
        k = 5                                                        #選擇K為5
        numTest = int(m * 0.5)                                       #將data分別50%當作訓練集和測試集
        errorCount = 0
        tStart = time.time()                                         #計算執行時間
        for i in range(numTest):                                     #Training資料集
                classifierResult = knn(normMat[i,:], normMat[0:numTest,:], labels[0:numTest], k)  
                #print("The classifier is: {0}, the real label is: {1}".format(classifierResult, labels[i]))  
                if classifierResult != labels[i]: errorCount += 1.0
        tEnd = time.time()
        print("Training time is %f ms" % (1000 * (tEnd - tStart)))
        print("Training accuracy is {:.2%}".format(1 - (errorCount/float(numTest)))) 

        print("-----------------------------")
        
        tStart_test = time.time()                                    #計算執行時間
        for i in range(numTest):                                     #Training資料集
                classifierResult = knn(normMat[i+89,:], normMat[numTest:m,:], labels[numTest:m], k)  
                #print("The classifier is: {0}, the real label is: {1}".format(classifierResult, labels[i+89]))  
                if classifierResult != labels[i+89]: errorCount += 1.0
        tEnd_test = time.time()
        print("Test time is %f ms" % (1000 * (tEnd_test - tStart_test)))
        print("Test accuracy is {:.2%}".format(1 - (errorCount/float(numTest))))
main()

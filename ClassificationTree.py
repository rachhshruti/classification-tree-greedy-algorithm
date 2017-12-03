from array import *
import math
import csv
import numpy as np
from scipy.spatial.distance import * 
import operator
import sys

'''
Implemented a greedy algorithm for building classification tree given a data set. 
It uses gini or information gain as a spliting criteria to decide the best attribute 
based on the user input. The algorithm gives an average accuracy of 0.95 which was 
evaluated by performing 10 fold cross validation on 10 different data sets.
@author Shruti Rachh
'''
ipFl=sys.argv[1]
delim=sys.argv[2]
splitCriteria=sys.argv[3]
np.set_printoptions(threshold=np.nan)

class Node:
    def __init__(self):
        attribute=''
        class_lbl=''
        split_point=0       
        left=None
        right=None

def findInforGainForAttribute(data, targetAttribute, classData):
    uniqueClassDataValues= np.unique(classData)
    splitList=[]
    splitList.append(float(targetAttribute[0])-0.005)
    sameInd=[]
    for i in range(0, len(targetAttribute)-1):
        weightedAverage=[]
        avg=(float(targetAttribute[i])+float(targetAttribute[i+1]))/2
        if(avg == float(targetAttribute[i])):
            sameInd.append(i+1)
        splitList.append((float(targetAttribute[i])+float(targetAttribute[i+1]))/2)
    splitList.append(float(targetAttribute[-1])+0.005)
    for i in range(0, len(splitList)):
        if(i in sameInd):
            weightedAverage.append(1)
        else:
            entropyArray=  np.zeros((len(uniqueClassDataValues),2), dtype=float)
            entropySigmaLesser=0.0
            entropyGreaterSum=0.0
            entropyLesserSum=0.0
            entropySigmaGreater=0.0
            for k in range(0, len(uniqueClassDataValues)):
                countGreater=0
                countLesser=0       
                for j2 in range(i, len(classData)):
                    if classData[j2]==uniqueClassDataValues[k]:
                        countGreater+=1  
                for j in range(0, i):
                    if classData[j]==uniqueClassDataValues[k]:
                        countLesser+=1          
                entropyArray[k,0]=countLesser
                entropyArray[k,1]=countGreater
                entropyGreaterSum+=entropyArray[k,1]
                entropyLesserSum+=entropyArray[k,0]
        
            for k in range(0, len(uniqueClassDataValues)):
                if entropyLesserSum!=0:
                    logVal=entropyArray[k,0]/entropyLesserSum
                    if logVal==0:
                        logVal=1.0
                    entropySigmaLesser+=(-logVal)*math.log(logVal , 2)
                else: 
                    entropySigmaLesser=0.0
            
                if entropyGreaterSum!=0:
                    logValGreater=entropyArray[k,1]/entropyGreaterSum
                    if logValGreater==0:
                        logValGreater=1.0
                    entropySigmaGreater+=(-entropyArray[k,1]/entropyGreaterSum)*math.log(logValGreater, 2)
                else: 
                    entropySigmaGreater=0.0
            
            weightedAverage.append((entropyLesserSum*entropySigmaLesser/len(classData))+(entropyGreaterSum*entropySigmaGreater/len(classData)))
    entropyIndex=weightedAverage.index(min(weightedAverage))
    entropySplitDetailslist=[]
    entropySplitDetailslist.append(min(weightedAverage))
    entropySplitDetailslist.append(splitList[entropyIndex])
    return  entropySplitDetailslist

def findGiniForAttribute(data, targetAttribute, classData):
    uniqueClassDataValues= np.unique(classData)
    splitList=[]
    splitList.append(float(targetAttribute[0])-0.005)
    sameInd=[]
    for i in range(0, len(targetAttribute)-1):
        weightedAverage=[]
        avg=(float(targetAttribute[i])+float(targetAttribute[i+1]))/2
        if(avg == float(targetAttribute[i])):
            sameInd.append(i+1)    
        splitList.append(avg)
    splitList.append(float(targetAttribute[-1])+0.005)
    for i in range(0, len(splitList)):
        if(i in sameInd):
            weightedAverage.append(1)
        else:
            giniArray=  np.zeros((len(uniqueClassDataValues),2), dtype=float)
            giniSigmaLesser=0.0
            giniGreaterSum=0.0
            giniLesserSum=0.0
            giniSigmaGreater=0.0
            for k in range(0, len(uniqueClassDataValues)):
                countGreater=0
                countLesser=0       
                for j2 in range(i, len(classData)):
                    if classData[j2]==uniqueClassDataValues[k]:
                        countGreater+=1  
                for j in range(0, i):
                    if classData[j]==uniqueClassDataValues[k]:
                        countLesser+=1          
                giniArray[k,0]=countLesser
                giniArray[k,1]=countGreater
                giniGreaterSum+=giniArray[k,1]
                giniLesserSum+=giniArray[k,0]
        
            for k in range(0, len(uniqueClassDataValues)):
                if giniLesserSum!=0:
                    giniSigmaLesser+=((giniArray[k,0])/giniLesserSum)**2
                else: 
                    giniSigmaLesser=1.0
                giniLesser=1-giniSigmaLesser
                if giniGreaterSum!=0:
                    giniSigmaGreater+=((giniArray[k,1])/giniGreaterSum)**2
                else: 
                    giniSigmaGreater=1.0
                giniGreater=1-giniSigmaGreater
            weightedAverage.append((giniLesserSum*giniLesser/len(classData))+(giniGreaterSum*giniGreater/len(classData)))
    
    giniIndex=weightedAverage.index(min(weightedAverage))
                
    giniSplitDetailslist=[]
    giniSplitDetailslist.append(min(weightedAverage))
    giniSplitDetailslist.append(splitList[giniIndex])
    return  giniSplitDetailslist

def findBestAttributeSplit(data):
    
    noOfCols = len(data[0])
    min=1
    for i in range(0,noOfCols-1):        
        sortedData= sorted(data,key=lambda x: x[i])
        sortedData=np.array(sortedData)
        if splitCriteria=="gini":
            giniSplitDetailslist=findGiniForAttribute(data, sortedData[:,i], sortedData[:,-1])
        else:
            giniSplitDetailslist=findInforGainForAttribute(data, sortedData[:,i], sortedData[:,-1])
        if min>= giniSplitDetailslist[0]:
            min=giniSplitDetailslist[0]
            splitPoint=giniSplitDetailslist[1]
            minIndex=i
            if min==0:
                break
    return (minIndex,splitPoint)

def stop_cond(ds,cols):
    unique_val=[]
    for i in range(0,cols-1):
        col=[row[i] for row in ds]
        print 'col',col
        unique_val.append(len(np.unique(col)))
    max_val = max(unique_val)
    class_col=[row[cols-1] for row in ds]
    if(len(np.unique(class_col))==1 or max_val == 1 ):
        return True
    return False

def classify(ds,cols):
    class_col=[row[cols-1] for row in ds]
    max=0
    for i in np.unique(class_col):
        cnt=class_col.count(i)
        if(max<cnt):
            max=cnt
            ind=i
    return ind

def splitData(ds,test_cond_val,col_num):
    ds=np.array([np.array(x) for x in ds])
    subset1=[]
    subset2=[]
    for i in range(0,len(ds)):
        if(float(ds[i,col_num]) <= test_cond_val):
            subset1.append(ds[i])
        else:
            subset2.append(ds[i])
    return (subset1,subset2)

def createNode():
    return Node()

def predictClass(root,attr_set):
    if root.left==None and root.right==None:
        return root.class_lbl
    if float(attr_set[root.attribute])<=root.split_point:
        class_label=predictClass(root.left,attr_set)
    else:
        class_label=predictClass(root.right,attr_set)
    return class_label

def predict(root,ds):
    rows=len(ds)
    ds=np.array(ds)
    mat=[]
    for i in range(0,rows):
        mat.append(predictClass(root,ds[i]))
    return mat

def TreeGrowth(ds,cols):
    if(stop_cond(ds,cols) == True or len(ds) == 1):
        leaf=createNode()
        leaf.class_lbl=classify(ds,cols)
        leaf.left=None
        leaf.right=None
        return leaf
    else:
        root=createNode()
        best_attr=findBestAttributeSplit(ds)
        root.attribute=best_attr[0]
        root.split_point=best_attr[1]
        sets=splitData(ds,best_attr[1],int(root.attribute))
        root.left=TreeGrowth(sets[0],cols)
        root.right=TreeGrowth(sets[1],cols)   
    return root

def calcAcc(sub,mat):
    colNum=len(sub[0])
    column=[rows[colNum-1] for rows in sub]
    cnt=0
    for i in range(0,len(sub)):
        if str(mat[i])==str(column[i]):
            cnt+=1
    accuracy=float(cnt)/len(sub)
    return accuracy

def partitionData(ds):
    np.random.shuffle(ds)
    lengthData = np.size(ds, axis=0)
    folds=10
    splitSize= len(ds)/folds
    partitions=[]
    train=[]
    test=[]
    for i in range(0,folds):
        partitions.append(ds[i*splitSize:(i+1)*(splitSize)])
    
    for i in range(0, folds):
        test.append(partitions[i])

    for i in range(0, folds):
        train.append(np.empty((len(ds)-splitSize, len(ds[0])), dtype=object))

    for i in range(0,folds):
        l=0
        for j in range(0, folds):
            for k in range(0,splitSize):
                if j!=i:            
                    train[i][l]=partitions[j][k]
                    l+=1
    return (train,test)

def display(node):
    if(node.left == None and node.right == None):
        print "class label ",node.class_lbl
    else:
        display(node.left)
        print "attribute ",node.attribute,"\tsplit_point ",node.split_point
        display(node.right)

with open(ipFl) as file:
    reader = csv.reader(file,delimiter=sys.argv[2])
    fl=list(reader)

fl=np.array([row for row in fl])
train_test=partitionData(fl)
num_cols=len(fl[0])
accuracy=0.0
for i in range(0,10):
    print "Training: ",i
    root=TreeGrowth(train_test[0][i],num_cols)
    display(root)
    print "\n"
    mat=predict(root,train_test[1][i])
    accuracy+=calcAcc(train_test[1][i],mat)
accuracy_avg=accuracy/10
print accuracy_avg

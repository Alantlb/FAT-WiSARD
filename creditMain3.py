print('Importing Libraries')

import pandas as pd
#from creditBrain3 import *
from FATWiSARD import *
brain = FATWiSARD(20,2,5,100)
def to_one_hot(labels,n_labels):
    new_labels = []
    for label in labels:
        new = [0]*n_labels
        new[label] = 1
        new_labels.append(new)
    return new_labels

def getStats(inp,out):
    global brain

    acc = 0.
    a,b,c,d = [0.]*4

    for i in range(len(inp)):
        points = list(brain.predict([inp[i]])[0])
        if sum(points) == 0:
            continue
        if points.index(max(points)) == out[i]:
            acc += 1.0
            if out[i] == 0:
                a += 1
            else:
                d += 1
        else:
            if out[i] == 0:
                b += 1
            else:
                c += 1
            pass
        #print("Prediction Net: {}".format(points),out[i])
    print("Accuracy: " + str(acc/len(out)))
    print("F1(Good): " + str((2*a)/(2*a + b + c)))
    print("F1(Bad): " +  str((2*d)/(2*d + b + c)))

print('Reading File')

df = pd.read_csv("germanData.csv",sep='\s+',header=None)

forMem = 0
forTrain = 0

trainF = []
trainL = []

predF = []
predL = []

print('Spliting Data')

for i in range(len(df)):

    feature = list(df.iloc[i,:-5])
    label = int(df.iloc[i,-1]) - 1 #Label -1 pois no dataset labels sao numeradas por 1 e 2

    if(forMem < 400):
        brain.train([feature],label)
        forMem += 1
    elif(forTrain < 400):
        trainF.append(feature)
        trainL.append(label)
        forTrain += 1
    else:
        predF.append(feature)
        predL.append(label)

print("Training")

brain.optimize(trainF,to_one_hot(trainL,2),0.01,1000,one_hot=True)

getStats(predF,predL)

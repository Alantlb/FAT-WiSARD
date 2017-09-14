from myNet import *

def to_one_hot(labels,n_labels):
    new_labels = []
    for label in labels:
        new = [0]*n_labels
        new[label] = 1
        new_labels.append(new)
    return new_labels

class FATWiSARD:

    def __init__(self,nIn,nClasses,nRams,rInit=1):

        self.brain = Brain(nIn,nClasses)
        self.brain.newMemory(nClasses-1)
        b = [self.brain.newNetFromInput() for i in range(nClasses)]
        m = [self.brain.newNetFromMemory(i) for i in range(nClasses)]
        orders = [b[i].shuffle() for i in range(nClasses)]
        for i in range(nClasses):
            m[i].shuffle(orders[i])
        o = []
        for i in range(nClasses):
            ramsM = m[i].Split(nRams)
            ramsB = b[i].Split(nRams)
            w = [ElementWiseNeuron(int(nIn/nRams)) for n in range(nRams)]
            r = [CompareNeuron(rInit) for n in range(nRams)]
            e = [ElementWiseNeuron(1) for n in range(nRams)]
            for j in range(nRams):
                ramsM[j].add(w[j])
                ramsB[j].add(w[j])
                ramsB[j].add(r[j],ramsM[j])
                ramsB[j].add(e[j])
            o.append(Sum(*ramsB))
        self.brain.setFinal(ramsB[0])
        o[0].Join(*o[1:])
        o[0].toProb()
        self.brain.setFinal(o[0])
        self.brain.start()

    def train(self,inp,label):
        self.brain.Memorys[label].add(inp)

    def predict(self,inp):
        return self.brain.predict(inp)

    def optimize(self,inp,out,learning_rate,epochs=1,**kwargs):
        one_hot = kwargs.get("one_hot",False)
        if not one_hot:
            out = to_one_hot(out,self.brain.nOut)
        self.brain.optimize(inp,out,learning_rate,epochs,log=kwargs.get("log",None))

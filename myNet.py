import tensorflow as tf
import numpy as np
import random
from collections import deque

class Net:
    '''Net class used to create a network of neurons for disered brain object'''
    def __init__(self,tensor):
        '''Creates a new Net Object starting from disered tensor'''
        self.tensor = tensor

    def add(self,neuron,*args):
        '''Add a new Neuron to the Net'''
        if repr(neuron) == "WeightNeuron":
            self.tensor = neuron.op(self.tensor)
        elif repr(neuron) == "CompareNeuron":
            self.tensor = neuron.op(self.tensor,args[0].tensor)
        elif repr(neuron) == "ElemtWiseNeuron":
            self.tensor = neuron.op(self.tensor)

    def Join(self,*args,**kwargs):
        '''Join Net Objects togheter'''
        with tf.name_scope("Join"):
            axis = kwargs.get("axis",1)
            tensors = [self] + list(args)
            self.tensor = tf.stack([i.tensor for i in tensors],axis=axis)

    def Concat(self,*args,**kwargs):
        '''Join Net Objects togheter'''
        with tf.name_scope("Concat"):
            axis = kwargs.get("axis",0)
            tensors = [self] + list(args)
            self.tensor = tf.concat([i.tensor for i in tensors],axis)

    def Split(self,times):
        with tf.name_scope("Split"):
            return [Net(t) for t in tf.split(self.tensor,times,axis=1)]

    def shuffle(self,order=None):
        '''Shuffles Tensor Inputs'''
        with tf.name_scope("Shuffle"):
            dim = int(self.tensor.shape[1])
            if not order:
                order = randomR(dim)
            tensors = tf.split(self.tensor,dim,axis=1)
            new = []
            for i in order:
                new.append(tensors[i])
            self.tensor = tf.concat(new,1)
            return order

    def toProb(self):
        '''Method to convert neuron outputs to probablities'''
        with tf.name_scope("toProb"):
            self.tensor = tf.transpose(tf.multiply(tf.transpose(self.tensor),tf.reduce_sum(self.tensor,axis=1)**-1))

    def Softmax(self):
        with tf.name_scope("Softmax"):
            self.tensor = tf.nn.softmax(self.tensor)

class Memory:
    '''Memory Object used to store n dimensional data'''
    def __init__(self,n):
        '''Creates a new Memory Object'''
        self.Mem = deque()
        self.x = tf.placeholder(shape=[None,n],dtype=tf.float32)
        self.numMem = 0

    def add(self,x):
        '''Add a new or more inputs and outputs pairs to the Memory'''
        self.Mem.extend(x)
        self.numMem += len(x)

    def getFeedDict(self):
        '''Return feed_dict with the memorys in the MemoryObject to be used by the Brain'''
        return {self.x:self.Mem}

    def getRoot(self):
        '''Return placeholder of memory input'''
        return self.x

class Brain:
    '''Brain class used to create a neural network'''
    def __init__(self,nIn,nOut):
        '''Creates a new Brain Object with nIn inputs and nOut outputs'''
        self.nIn = nIn
        self.nOut = nOut

        self.sess = tf.Session()

        self.x = tf.placeholder(shape=[None,nIn],dtype=tf.float32)
        self.y = tf.placeholder(shape=[None,nOut],dtype=tf.float32)

        self.Memorys = [Memory(self.nIn)]
        self.numMem = 1

        self.tensor = None

    def start(self):
        '''Method to be called after brain construction is done in order to initializer tensorlfow backend variables'''
        self.sess.run(tf.global_variables_initializer())

    def getFeedDict(self,inp,out=None):
        feed_dict = {self.x:inp}
        if out != None:
            feed_dict[self.y] = out
        for i in range(len(self.Memorys)):
            feed_dict.update(self.Memorys[i].getFeedDict())
        return feed_dict

    def predict(self,inp):
        '''Predicts output given and inp Input'''
        feed_dict = self.getFeedDict(inp)
        return self.sess.run(self.tensor,feed_dict=feed_dict)

    def setFinal(self,tensor):
        '''Defines the final tensor of the Brain'''
        self.tensor = tensor.tensor

    def newMemory(self,n=1):
        '''Creates a N new Memory Objects in the Brain'''
        for i in range(n):
            self.Memorys.append(Memory(self.nIn))
            self.numMem += 1

    def addMemory(self,inp,addr=0):
        '''Add a new memory to desired Memory Object in the Brain'''
        self.Memorys[addr].add(inp)

    def newNetFromInput(self):
        '''Return a new Net Object with tensor coming from the Input of the Brain'''
        return Net(self.x)

    def newNetFromMemory(self,addr=0,**kwargs):
        '''Returns a new Net Object with tensor coming from selected Memory Object from the Brain'''
        getAll = kwargs.get("getAll",False)
        if getAll:
            return [Net(self.Memorys[i].getRoot()) for i in range(len(self.Memorys))]
        return Net(self.Memorys[addr].getRoot())

    def optimize(self,inp,out,learning_rate,epochs=1,**kwargs):
        '''Trains the Brain with given Inputs,Outputs, Learning Rate, Epochs, default=1, and optionally chosen Optimizer, default=GradientDescent'''
        with tf.name_scope("Loss"):
            loss = -tf.reduce_sum(out*tf.log(tf.clip_by_value(self.tensor,1e-10,1.0)))

        with tf.name_scope("Optimizer"):
            opt = kwargs.get("optimizer","Grad")
            log = kwargs.get("log",None)
            if opt == "Grad":
                opt = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            elif opt == "Adam":
                opt = tf.train.AdamOptimizer(learning_rate=learning_rate)

        with tf.name_scope("Train"):
            train_step = opt.minimize(loss)

        feed_dict = self.getFeedDict(inp,out)

        for i in range(epochs):
            if log != None and i % log == 0:
                print("Epoch: " + str(i))
            self.sess.run(train_step,feed_dict=feed_dict)

    def saveTensorBoard(self,name):
        writer = tf.summary.FileWriter(name,self.sess.graph)
        #writer.add_graph(self.sess.graph)

class WeightNeuron:
    '''Classical Weight used in Neural Networks'''
    def __init__(self,nIn,nOut,**kwargs):
        weight = np.array( kwargs.get("weight",np.ones([nIn,nOut])),dtype=np.float32 )
        bias = np.array( kwargs.get("bias",np.ones([nOut])),dtype=np.float32)
        with tf.name_scope("W"):
            self.weight = tf.Variable(weight,dtype=tf.float32)
        with tf.name_scope("b"):
            self.bias = tf.Variable(bias,dtype=tf.float32)

    def op(self,tensor):
        '''Return neuron operation in given tensor'''
        with tf.name_scope("WeightNeuron"):
            return tf.nn.relu(tf.add(tf.matmul(tensor,self.weight),self.bias))

    def __repr__(self):
        return "WeightNeuron"

class ElementWiseNeuron:
    '''Create a neuron that ops as an ElemtWise operation given the N dimension'''
    def __init__(self,n,**kwargs):
        weight = np.array( kwargs.get("array",np.ones([n])),dtype=np.float32 )
        with tf.name_scope("W"):
            self.var = tf.Variable(weight,dtype=tf.float32)

    def op(self,tensor):
        '''Return neuron operation in given tensor'''
        with tf.name_scope("ElemtWiseNeuron"):
            return tf.nn.relu(tf.multiply(tensor,self.var))

    def __repr__(self):
        return "ElemtWiseNeuron"



class CompareNeuron:
    '''Neuron with op made to compare t1 Tensor and t2 Tensor within a R radius'''
    def __init__(self,r=1):
        with tf.name_scope("R"):
            self.var = tf.Variable(r,dtype=tf.float32)

    def op(self,t1,t2):
        '''Return neuron operation in given tensors'''
        with tf.name_scope("CompareNeuron"):
            front  = tf.reduce_sum(tf.square(t1),axis=1)
            middle = tf.multiply(tf.matmul(t1,t2,transpose_b=True),-2)
            back = tf.reduce_sum(tf.square(t2),axis=1)
            join = tf.reduce_sum(tf.meshgrid(front,back),axis=0)
            body = tf.add(middle,tf.transpose(join))
            o = tf.reduce_sum(tf.nn.relu((self.var-body)/self.var),axis=1)
            return o

    def __repr__(self):
        return "CompareNeuron"

def Sum(*args):
    with tf.name_scope("Sum"):
        t = args[0].tensor
        for i in range(len(args)-1):
            t = tf.add(t,args[i+1].tensor)
        return Net(t)

def randomR(n):
    l = []
    while len(l) != n:
        r = random.randint(0,n-1)
        while r in l:
            r = random.randint(0,n-1)
        l.append(r)
    return l

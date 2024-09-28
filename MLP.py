import numpy as np

def convertLabel(label, num_classes):
    return np.reshape(np.eye(num_classes)[label], (num_classes, 1))

def ReLU(x):
    return np.maximum(x, 0)
# def Softmax(x):
#     return np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))

class MLP(object):

    def __init__(self, numLayer = 1, numIn = 0, numOut = 0, numHiddenUnits = []) -> None:
        # Initialize
        self.numLayer = numLayer
        self.numIn = numIn
        self.numOut = numOut
        self.numHiddenUnits = numHiddenUnits
        #Initialize list of weight and bias matrices
        #   Weight matrices --> random
        #   Bias matrices --> zeros
        self.W = [] 
        self.B = []
        np.random.seed(16)
        for i in range(0, numLayer):
            if i == 0:
                self.W.append(0.01*np.random.randn(numIn,numHiddenUnits[0]))
                self.B.append(np.zeros(([numHiddenUnits[0],1])))
            elif i == numLayer-1:
                self.W.append(0.01*np.random.randn(numHiddenUnits[i-1],numOut))
                self.B.append(np.zeros(([numOut,1])))
            else:
                self.W.append(0.01*np.random.randn(numHiddenUnits[i-1],numHiddenUnits[i]))
                self.B.append(np.zeros(([numHiddenUnits[i],1])))
    
    def loadData(self, train_x, train_y, test_x, test_y):
        self.train_x = train_x
        self.train_y = train_y
        self.test_x = test_x
        self.test_y = test_y

    def initData(self, input_train, output_train):
        X = np.reshape(input_train, (self.numIn, 1))
        Y = convertLabel(output_train, self.numOut)
        return (X, Y)
    
    def GD(self, dW, dB, pvW, pvB, learning_rate, momentum):
        cvW = []
        cvB = []
        for l in range(0, self.numLayer):
            cvW.append(momentum*pvW[l] + learning_rate*dW[self.numLayer - l - 1])
            cvB.append(momentum*pvB[l] + learning_rate*dB[self.numLayer - l - 1])
            self.W[l] += -cvW[l]
            self.B[l] += -cvB[l]
        return (cvW, cvB)

    def feedForward(self, X):
        Zarr = []
        A = []
        for l in range(0, self.numLayer):
            if l == 0:
                Z = np.dot(self.W[l].T, X) + self.B[l]
            else:
                Z = np.dot(self.W[l].T, A[l-1]) + self.B[l]
            Zarr.append(Z)
            if l == self.numLayer - 1:
                A.append(ReLU(Z))
            else:
                A.append(ReLU(Z))
        return (Zarr, A)
    
    def loss(self, Y, Yhat):
        return np.mean((Y-Yhat)**2)/2
    
    def backPropagation(self, X, Y, A, Zarr):
        dW = []
        dB = []
        preE = 0
        for l in range(0, self.numLayer):
            layerIndex = self.numLayer - 1 - l
            if l == 0:
                E = (A[-1] - Y)/self.numOut
            else:
                E = np.dot(self.W[layerIndex + 1], preE)
                E[Zarr[layerIndex] <= 0] = 0
            if layerIndex != 0:
                dW.append(np.dot(A[layerIndex - 1], E.T))
            else:
                dW.append(np.dot(X,E.T))
            dB.append(E)
            preE = E
        return (dW, dB)
    
    def train(self, numIter, learning_rate, batch_size = 1, momentum = 0):
        for i in range(numIter):
            # Initialize input and output layer
            (X, Y) = self.initData(self.train_x[i%self.train_x.shape[0]], 
                                        self.train_y[i%self.train_y.shape[0]])
            # Feedforward
            (Zarr, A) = self.feedForward(X)
            # Calculate the loss and print every 1000 iterations 
            cost = self.loss(Y, A[-1])
            if (i%1000 == 0):
                print(f"Iter {i} loss is {cost}")
            # Backpropagation, average the dW, dB in a batch
            if i%batch_size == 0:
                (dW, dB) = self.backPropagation(X, Y, A, Zarr)
            else:
                (tdW, tdB) = self.backPropagation(X, Y, A, Zarr)
                dW = [(x*(i%batch_size - 1)+y)/(i%batch_size) for x,y in zip(dW, tdW)]
                dB = [(x*(i%batch_size - 1)+y)/(i%batch_size) for x,y in zip(dB, tdB)]
            # Gradient Descent
            if i == 0:
                pvW = []
                pvB = []
                for l in range(self.numLayer):
                    pvW.append(np.zeros(dW[self.numLayer - l - 1].shape))
                    pvB.append(np.zeros(dB[self.numLayer - l - 1].shape))
            if i%batch_size == 0:
                (pvW, pvB) = self.GD(dW, dB, pvW, pvB, learning_rate, momentum)
            
        
    
    def test(self, numTest):
        print(f"-------------START {numTest} TESTS-------------")
        numRight = 0
        for test in range(numTest):
            (X, Y) = self.initData(self.test_x[test], self.test_y[test])
            (Zarr, A) = self.feedForward(X)
            Correct = np.argmax(Y)
            Predicted = np.argmax(A[-1])
            if Correct == Predicted:
                print ("RIGHT! Label ", Predicted)
                numRight+=1
            else:
                print("WRONG! Predicted class: ", Predicted, "| Correct class: ", Correct)
        print(f"TOTAL: {numRight}/{numTest} ({numRight*100/numTest}%) tests right ")





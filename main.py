import tensorflow as tf
from MLP import *
from tensorflow.keras.datasets import mnist
import os

def main():
    # Load data    
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    input_train = tf.reshape(x_train, (-1, 28*28)).numpy()/255
    input_test = tf.reshape(x_test, (-1, 28*28)).numpy()/255
    #Initialize 
    e = 0.1 #Learning rate
    momentum = 0.9
    hiddenLayersNum =  2
    unitPerHidden = [521, 125] #Number of units per hidden-layers
    inputNum = 28*28 #Number of input nodes
    outputNum = 10  #Number of output nodes
    iNum = 65000 #Number of training iterations
    batch_size = 1

    mnistMLP = MLP(hiddenLayersNum+1, inputNum, outputNum, unitPerHidden)
    mnistMLP.loadData(input_train, y_train, input_test, y_test)
    trainTimes = iNum
    while True:
        mnistMLP.train(iNum, e, batch_size, momentum)
        mnistMLP.test(500)
        more = int(input(f"Train more {iNum}? "))
        if more == 0:
            break
        else:
            trainTimes += more
            iNum = more
    print(f"TOTAL train {trainTimes} times")
    export = input("Do you want to export the model? Y/N")
    if export == "Y":
        print(f"Exporting model...")
        with open('output_W.txt', 'w') as file:
            print(mnistMLP.W, file=file)
        with open('output_B.txt', 'w') as file:
            print(mnistMLP.B, file=file)
        print(f"DONE!")
    


if __name__ == "__main__":
    os.system('cls')
    main()
from matplotlib import pyplot as plt
from multiLayerPerceptron import MLP
import numpy as np
from numpy import random

def char_to_int(char):
    return ord(char) - ord('A')+1

if __name__ == "__main__":
    letters = []
    numbers = []
    numbers2=[[]]
    results=[]
    counter=0

    try:
        with open('data.txt', 'r') as file:
            for line in file:
                elements = line.strip().split(',')
                letters.append(char_to_int(elements[0]))
                numbers.extend(map(int, elements[1:]))
                numbers2.append(numbers)
                numbers=[]
    except FileNotFoundError:
        print("File not found")

    numbers2=numbers2[1:]
    vectorInput=np.array(numbers2)
    vectorOutput=np.array(letters)

    mlp=MLP(16, [10,10], 1)

    for epoch in range(1000):
        outputs = mlp.forward(vectorInput.T)
        gradients = mlp.backward(vectorInput.T, vectorOutput.T)
        mlp.update_parameters(gradients, 0.01)
        loss = np.mean((outputs-vectorOutput.T)**2)
        if (epoch%100==0):
            results.append(loss)

    test_output = mlp.forward(vectorInput.T)
    test_loss = np.mean((test_output - vectorOutput.T) ** 2)
    plt.plot(results)
    plt.show()



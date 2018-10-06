import csv
import random
import math
import operator
from matplotlib.pylab import plt


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        #print(dataset)
        print(len(dataset))
        for x in range(len(dataset) - 1):
            for y in range(784):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                trainingSet.append(dataset[x])
            else:
                testSet.append(dataset[x])


def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance) - 1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def getAccuracy(testSet, predictions):
    correct = 0
    for x in range(len(testSet)):
        if testSet[x][-1] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def main():
    # prepare data
    trainingSet = []
    testSet = []
    split = 0.80
   # path = r'C:/Users/ASUS/PycharmProjects/knntest/iris'
    loadDataset("C:\\Users\\ASUS\\Desktop\\mnist_test_1.csv", split, trainingSet, testSet)
    print('Train set: ' + repr(len(trainingSet)))
    print('Test set: ' + repr(len(testSet)))
    # generate predictions
    predictions = []
    k = 5
    xvalues=[]
    pred=[]
    actual=[]
    for x in range(len(testSet)):
        neighbors = getNeighbors(trainingSet, testSet[x], k)
       # print(neighbors)
        result = getResponse(neighbors)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
        xvalues.append(x)
        pred.append(result)
        actual.append(testSet[x][-1])

        #plt.plot(x,repr(testSet[x][-1]),color="chocolate",label="Actual Values")
    accuracy = getAccuracy(testSet, predictions)
    plt.plot(xvalues,pred,".", color="r",label="Predicted Values")
    plt.plot(xvalues, actual,".", color="b", label="Actual Values")

    plt.show();
    print('Accuracy: ' + repr(accuracy) + '%')


main()
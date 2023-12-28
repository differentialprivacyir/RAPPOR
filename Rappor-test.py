import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from Client import Client
from Server import Server
from datetime import datetime
from time import time
import sys
import pandas as pd
import math

def fg(epsUlt):
    f = 2/(1 + math.exp(epsUlt/2))
    g = 1 -f
    return f, g

def pq(epsUlt, epsIns):
    p = (math.exp(epsUlt/2) - math.exp(epsIns/2)) / ((math.exp(epsUlt/2) - 1)*(math.exp(epsIns/2) + 1))
    q = 1 - p
    return p, q

def computeMean(estimation, N):
    result = 0
    for value in estimation:
        result += int(value) * estimation[value]
    return result / N

def convertCumulativeFrequencyToInstantaneous(previous, current):
    result = dict()
    for value in current:
        result[value] = current[value] - previous[value]
    return result

def convertDataFrequencyToBitFrequency(dataFrequency, DATA_SET_SIZE, N):
    bitEstimation = np.zeros(DATA_SET_SIZE)
    for value in dataFrequency:
        bitRepresentation = bin(int(value))[2:].zfill(DATA_SET_SIZE)
        numericalBitRepresentation = np.array([int(char) for char in bitRepresentation])
        bitEstimation = bitEstimation + numericalBitRepresentation * dataFrequency[value]
    bitEstimation = bitEstimation / N
    return bitEstimation

# f = 0.5  # privacy parameter for prr
# p = 0.5  # privacy parameter for irr
# q = 0.75  # privacy parameter for irr
k = 40  # number of bloom filter bits
m = 64  # number of cohorts
h = 5  # number of hash functions
alpha = 0.125  # estimation parameter
N = int(sys.argv[1]) * 1000  # number of clients
numbers = np.arange(0, 256)
Data = [f'{i}' for i in numbers]
#Maximum number of bits in data:
DATA_SET_SIZE = 8
#Overal Algorithm Execution Rounds:
OAER = 50
# Number of rounds to run the code:
ROUND_CHANGES = 20
levels = [0.1, 0.3, 0.5, 0.7, 0.9]
averageMSE = [[0] * ROUND_CHANGES for i in levels]
averageMAE = [[0] * ROUND_CHANGES for i in levels]
averageME = [[0] * ROUND_CHANGES for i in levels]
csvContent = pd.read_csv(f'./hpcDatasets/{sys.argv[2]}.csv')
dataSet = np.transpose(csvContent.to_numpy())
avgBudget = 0
maxBudget = 0
minBudget = 0

for oaer in range(OAER):
    print(f'Start of round {oaer} at:', datetime.now())
    clientSelectedLevel = [0] * int(N/len(levels)) + [1] * int(N/len(levels)) + [2] * int(N/len(levels)) + [3] * int(N/len(levels)) + [4] * int(N/len(levels))
      
    reports = dict()  # number of times each Word really reported
    estimated = dict()  # estimation of number of times each Word reported
    error = []
    real_freq = []
    est_freq = []

    servers = []
    clients = []
    for eps in levels:
        f, g = fg(eps)
        p, q = pq(eps, 1/2 * eps)
        
        servers.append(
            Server(f, p, q, k, m, h, alpha)
        )
        clients.extend([Client(f, p, q, k, m, h) for i in range(int(N/len(levels)))])
    # S = Server(f, p, q, k, m, h, alpha)  # creating server S
    # Prepare to keep results of estimations:
    estimations = []
    cumulativeEstimatedFrequency = {}
    for l in levels:
        cumulativeEstimatedFrequency[l] = {}
        for value in Data:
            cumulativeEstimatedFrequency[l][value] = 0
    startRoundTime = time()
    consumedBudgets = [0 for i in range(N)]
    for i in range(ROUND_CHANGES):
        print(f'round {i} started')
        startTimestamp = time()
        for j in range(N):
            if i != 0:
                if dataSet[i][j] != dataSet[i - 1][j]:
                    consumedBudgets[j] += (1/2 * levels[clientSelectedLevel[j]])
            report = clients[j].report(f'{dataSet[i][j]}')
            consumedBudgets[j] += levels[clientSelectedLevel[j]]
            servers[clientSelectedLevel[j]].collect(report)
        estimations.append([])
        for serverIndex in range(len(servers)):
            estimated = servers[serverIndex].estimation(Data)
            # sumOfEstimatedUsers = np.sum([estimated[i] for i in estimated])
            # for value in estimated:
            #     estimated[value] = estimated[value] * ((N/len(levels))/sumOfEstimatedUsers)
            for value in estimated:
                if estimated[value] < 0:
                    print('Negative estimation detected.')
                    raise Exception("Error! Negative frequency.")
            bitFrequency = convertDataFrequencyToBitFrequency(estimated, DATA_SET_SIZE, N/len(levels))
            max = np.max(bitFrequency)
            if max > 1:
                bitFrequency /= max
            estimations[i].append(bitFrequency)
            servers[serverIndex].clear()
            estimated.clear()
            # estimations.append[estimated]
        endTimestamp = time()
        print(f'Server estimated at {(endTimestamp-startTimestamp)/60} minutes')
        startTimestamp = time()
    
    endRoundTime = time()
    print(f'Round took {(endRoundTime - startRoundTime) / 60} minutes.')

    # Compute bit frequency of real values:
    frequencies = []
    for singleRound in dataSet:
        bitRepresentationOfDataSet = [bin(i)[2:].zfill(DATA_SET_SIZE) for i in singleRound]
        numericalBitRepresentationDataSet = [[int(char) for char in list(data)] \
                                            for data in bitRepresentationOfDataSet]
        frequencies.append(np.sum(numericalBitRepresentationDataSet, axis=0))
    normalized = np.array(frequencies) / N

    # Evaluate mean error:
    for r in range(ROUND_CHANGES):
        print(f'\n\n\n ========================================== \nResults of Round {r}:\n==========================================')
        error = []
        for index, estimation in enumerate(estimations[r]):
            print(f'Evaluation for level eps = {levels[index]}')
            for i, _ in enumerate(normalized[r]):  # calculating errors
                error.append(abs(estimation[i] - normalized[r][i]) * 100)
                print("index:", i, "-> Estimated:", estimation[i], " Real:", normalized[r][i], " Error: %", int(error[-1]))
            print("Global Mean Square Error:", mean_squared_error(normalized[r], estimation))
            print("Global Mean Absolute Error:", mean_absolute_error(normalized[r], estimation))
            averageMSE[index][r] = (averageMSE[index][r] * oaer + mean_squared_error(normalized[r], estimation))/(oaer+1)
            averageMAE[index][r] = (averageMAE[index][r] * oaer + mean_absolute_error(normalized[r], estimation))/(oaer+1)


    meanOfRounds = np.mean(dataSet, axis=1)
    print('Real Mean of rounds is:', meanOfRounds)
    outputMean = []
    for r in range(ROUND_CHANGES):
        outputMean.append([])
        for index, estimationAtL in enumerate(estimations[r]):
            ROUND_MEAN = 0
            REAL_ROUND_MEAN = 0
            for index2, number in enumerate(estimationAtL):
                ROUND_MEAN += (number*(N) * 2 ** (len(estimationAtL) - 1 - index2))
                REAL_ROUND_MEAN += (normalized[r][index2]*(N) * 2 ** (len(estimationAtL) - 1 - index2))
            ROUND_MEAN /= N
            REAL_ROUND_MEAN /= N
            print(f'Mean Difference at round {r} and level {levels[index]}:', abs(ROUND_MEAN - meanOfRounds[r]), abs(meanOfRounds[r] - REAL_ROUND_MEAN))
            averageME[index][r] = (averageME[index][r] * oaer + abs(ROUND_MEAN - meanOfRounds[r]))/(oaer+1)
            outputMean[r].append(ROUND_MEAN)
    print("Estimated Mean is:", outputMean)
    avgBudget = (avgBudget * oaer + np.mean(consumedBudgets))/(oaer + 1)
    maxBudget = (maxBudget * oaer + np.max(consumedBudgets))/(oaer + 1)
    minBudget = (minBudget * oaer + np.min(consumedBudgets))/(oaer + 1)

print("Results for Averaged MSE:", averageMSE)
print("Results for Averaged MAE:", averageMAE)
print("Results for Averaged ME:", averageME)

print("Averaged mean budget usage:", avgBudget)
print("Averaged max budget usage:", maxBudget)
print("Averaged min budget usage:", minBudget) 

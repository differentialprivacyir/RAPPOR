import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from Client import Client
from Server import Server


f = 0.5  # privacy parameter for prr
p = 0.5  # privacy parameter for irr
q = 0.75  # privacy parameter for irr
k = 40  # number of bloom filter bits
m = 64  # number of cohorts
h = 5  # number of hash functions
alpha = 0.125  # estimation parameter
clients = 100000  # number of clients
Data = ['contrary', 'popular', 'belief', 'lorem', 'ipsum', 'simply', 'random', 'text', 'latin', 'literature']

reports = dict()  # number of times each Word really reported
estimated = dict()  # estimation of number of times each Word reported
error = []
real_freq = []
est_freq = []

S = Server(f, p, q, k, m, h, alpha)  # creating server S

for i in range(clients):
    C = Client(f, p, q, k, m, h)  # creating clients
    for j in range(np.random.randint(0, 25)):  # select number of the client's reports randomly from 0 to 25
        np.random.shuffle(Data)   # select the report randomly from Data list
        datum = Data[0]

        # calculating number of times each Word reported
        if reports.get(datum) is None:  reports[datum] = 1
        else:  reports[datum] += 1

        report = C.report(datum)  # passing the data to the client and getting its report
        S.collect(report)  # passing the client's report to the server

estimated = S.estimation(Data)  # getting the estimation result from the server

for datum in Data:  # calculating errors
    real_freq.append(reports[datum])
    est_freq.append(estimated[datum])
    error.append((estimated[datum] - reports[datum]) / reports[datum] * 100)
    print(datum, "-> Estimated:", int(estimated[datum]), " Real:", reports[datum], " Error: %", int(error[-1]))

print("Avg Error: %", np.mean(error))
print("Min Squared Error:", mean_squared_error(real_freq, est_freq))

x_axis = np.arange(len(Data))
plt.bar(x_axis - 0.25, real_freq, label='Real Freq', width=0.5)
plt.bar(x_axis + 0.25, est_freq, label='Est Freq', width=0.5)
plt.ylabel('Frequency')
plt.xlabel('Domain Values')
plt.legend(loc='upper right', bbox_to_anchor=(1, 1.15))
plt.show()

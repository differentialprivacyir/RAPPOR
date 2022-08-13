- [RAPPOR](#rappor)
  - [Client](#client)
  - [Server](#server)
- [Installation](#installation)
  - [Virtual Environment](#virtual-environment)
  - [Dependencies](#dependencies)
- [Usage & Test](#usage--test)
- [Results](#results)


# RAPPOR
This module is developed based on *RAPPOR*.  
*RAPPOR* is a solution invented by google leveraging [*Local Differential Privacy* (*LDP*)][ldp] to aggregate data from clients.

In this solution, clients are assigned to some cohorts where each cohort uses a different set of hash functions for bloom filter.

## Client
Client apply 3 step to publish its data:
1. Encode its data to a bit array (bitmap) using bloom filter.
2. Applying [*Random Response*][RR] to resulted bitmap. The result of this step is stored permanently.
3. By getting result of last step, apply another [*Random Response*][RR] and publish the result finally.

## Server
Server query clients and aggregate the data of clients along their cohort.
Then server apply following steps to estimate frequency of each item:
1. Evaluate the frequency of each bit using [*Random Response Estimation*][RR] in received bitmaps from clients.
2. Train and apply a [*Lasso Regression*][LR] model to predicate the frequency of each item.


# Installation
Before installing this module, you may need to download some of libraries as dependency.  
We assume you have `python` version `3` installed already.

You can skip to [Dependencies](#dependencies) section to install required libraries but it's highly recommended to create a [virtual environment](#virtual-environment) to keep your workspace clean.
## Virtual Environment
Assuming you have `python3` installed on your system, To create a virtual environment you can run following command:

```
$ python3 -m venv rappor
```
After creating environment, you should activate it as follow:
```
$ source ./rappor/bin/activate
```

> For more information about virtual environments, you can refer to [venv][venv].

*NOTE:* When you are done with *RAPPOR*, you can exit from virutal environment using following command:
```
$ deactivate
```

## Dependencies
You need to install 3 module to run this code:

1. `mmh3`: A module which provides a set of *non-cryptography* hash functions which are well-known for their fast operations:
```
    $ pip3 install mmh3
```
2. `numpy`: A well-known module which provides a high quality data type and functions for array in python:
```
    $ pip3 install numpy
```
3. `sklearn` (`scikit-learn`): A free and popular module for implementing machine learning tasks in python. It has a lot of features such as classification, regression, clustering and so on:
```
    $ pip3 install sklearn
```
4. `matplotlib`: A module to draw digrams and charts easily. You can install it as follow:
```
    $ pip3 install matplotlib
```

# Usage & Test
A sample test is provided at [Rappor-test.py](./Rappor-test.py) which also shows how you can use this module.  

You can run this test by following command:
```
    $ ./python3 ./Rappor-test.py
```

After some minutes you will observe the great result of this module.

# Results
A sample result of execution is provided below to show the strength of this implementation:

![sampleResult][result]


[venv]:https://docs.python.org/3/library/venv.html
[ldp]: https://en.wikipedia.org/wiki/Local_differential_privacy
[result]: ./result.jpeg
[RR]: https://en.wikipedia.org/wiki/Randomized_response
[LR]: https://en.wikipedia.org/wiki/Lasso_(statistics)
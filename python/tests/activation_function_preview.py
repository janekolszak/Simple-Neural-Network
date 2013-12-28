#!/usr/bin/python
import sys
sys.path.append("../../lib")

import pysnn
import numpy as np
import matplotlib.pyplot as plt

def preview(af):
    xxx = np.linspace(-10, 10, 1000)
    yyy =[]
    zzz =[]
    for x in xxx:
        yyy.append(af.value(x))
        zzz.append(af.derivative(x))

    plt.plot(xxx, yyy, '-', linewidth=1, label="value")
    plt.plot(xxx, zzz, '--', linewidth=2, label="derivative")
    plt.legend()
    plt.grid()
    plt.show()


preview(pysnn.LogSigmoid(10,100))
preview(pysnn.Linear(0,5, 1))
preview(pysnn.LinearScalingFunction(1,5,-2, 1))
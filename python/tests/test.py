#!/usr/bin/python
import sys
sys.path.append("../../lib")

import pysnn

print pysnn.logSigmoid(0)
print pysnn.logSigmoidDerivative(0)
import numpy as np
import matplotlib.pyplot as plt


xxx = np.linspace(-10, 10)
yyy =[]
zzz =[]
for x in xxx:
    yyy.append(pysnn.logSigmoid(x))
    zzz.append(pysnn.logSigmoidDerivative(x))


plt.plot(xxx, yyy, '-', linewidth=1)
plt.plot(xxx, zzz, '--', linewidth=2)
# plt.show()


print np.fromiter(pysnn.qwerqwer(), np.float)

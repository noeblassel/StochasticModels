import bm
import matplotlib.pyplot as plt
import numpy as np

T=np.linspace(0,10,10,dtype=int)
X=T**2
W=bm.BrownianInterpolation(1,T,X)
Ts=np.linspace(0,10,10000)
Y=W.sample(0,10,10000)
plt.plot(Ts,Y)
plt.show()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math
a=1
count=[10,100,1000,10000,100000,1000000]
Pi_Cap=[]
for i in count:
    RanVal = [[np.random.uniform(-a/2,a/2),np.random.uniform(-a/2, a/2)] for j in range(i)]
    OnTarget=0
    for j in RanVal:
        if(((j[0]**2+j[1]**2)**(1/2))<=a/2):
            OnTarget+=1
    Pi_Cap.append((OnTarget/i)*4)
print(Pi_Cap)
print(count)
Mc=pd.DataFrame({"Number of Darts":count,"Estimated value of Pi":Pi_Cap})
Mc.to_csv('MonteCarloUniform.csv',index=False)
plt.plot(count, Pi_Cap, marker='D', linestyle='-',label='Estimated Value of Pi')
plt.axhline(y=math.pi,color="black",linestyle="--")
plt.title("Monte Carlo Simulations")
plt.yscale('linear')
plt.xscale('log')
plt.xlabel('Number of Darts')
plt.ylabel('Estimated Value of Pi')
plt.figtext(0.5, 0.01, 'As the number of random points increases, the estimated value of π converges towards the actual value of π (approximately 3.14159)\n. This convergence demonstrates the principle of the Monte Carlo method, \nwhere more iterations lead to more accurate estimates.', fontsize=12, color='black', ha='center')
plt.show()



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
Data={
    "X":[-3.0,-2.0,-1.0,0.0,1.0,2.0,3.0],
    "Y":[7.0,2.0,0.0,0.0,0.0,2.0,7.0]
}
DataSet=pd.DataFrame(Data)
fn={
    "b1":[],
    "b2":[],
    "Epsilon":[]
}
MinEpsilon=2**30
B1=-1
B2=-1
for i in np.arange(-1,1,0.01):
    for j in np.arange(-1,1,0.01):
        b1=i
        b2=j
        count=0
        fn["b1"].append(i)
        fn["b2"].append(j)
        for k in range(len(DataSet)):
            HypVal=(b1*DataSet.iloc[k].X)+(b2*(DataSet.iloc[k].X)**2)
            count+=abs(DataSet.iloc[k].Y-HypVal)
        fn["Epsilon"].append(count)    
        if(count<MinEpsilon):
            MinEpsilon=count
            B1=b1
            B2=b2 
print(f"MinEpsilon is {MinEpsilon} and it occurs when B1 is {B1} and B2 is {B2}")           
EpsilonVals=pd.DataFrame(fn)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(EpsilonVals.b1,EpsilonVals.b2,EpsilonVals.Epsilon, cmap='viridis', edgecolor='none')
ax.set_xlabel('b1')
ax.set_ylabel('b2')
ax.set_zlabel('Epsilon')
ax.set_title('Surface Plot')
plt.show() 

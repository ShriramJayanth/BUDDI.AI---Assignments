import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def calcNorm(sd,mean,x):
    part1=(1/((sd)*(2*math.pi)))
    part2=math.exp((-1/2)*(((x-mean)/sd))**2)
    return(part1*part2)
sd=[3,2,1]
mean=0
dt={
    "X1":[],
    "Y1":[],
    "X2":[],
    "Y2":[],
    "X3":[],
    "Y3":[],
}
for i in range(len(sd)):
    for x in np.arange(mean-(5*sd[i]),mean+(5*sd[i])+1,0.1):
        dt[f"X{i+1}"].append(x)
        dt[f"Y{i+1}"].append(calcNorm(sd[i],mean,x))
# for x in np.arange(mean-(5*sd[0]),mean+(5*sd[0])+1,0.1):
#     dt["X1"].append(x)
#     dt["Y1"].append(calcNorm(sd[0],mean,x))
#     dt["X2"].append(x)
#     dt["Y2"].append(calcNorm(sd[1],mean,x))
#     dt["X3"].append(x)
#     dt["Y3"].append(calcNorm(sd[2],mean,x))
plt.plot(dt["X1"],dt["Y1"],color="blue",label="sd=3")
plt.plot(dt["X2"],dt["Y2"],color="red",label="sd=2")
plt.plot(dt["X3"],dt["Y3"],color="black",label="sd=1")
plt.title("Normal Distribution")
plt.xlabel("X(linear scale)")
plt.ylabel("Y(linear scale)")
plt.figtext(0.5, 0.01, 'This graph is obtained by keeping mean constant and varying Standard deviation. From this graph we can infer that,\n when we reduce the standard deviation the height of the graph becomes lower and vice versa. The height of the peak is inversely proprtional to Standard deviation.', fontsize=12, color='black', ha='center')
plt.legend()
plt.show()    


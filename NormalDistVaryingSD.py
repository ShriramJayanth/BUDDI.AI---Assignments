import math
import numpy as np
import matplotlib.pyplot as plt

# CalcNorm is a function to calculate the y for the given x using normal distribution formula
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

#A loop to compute the values for given x by keeping mean constant and varying sd
for i in range(len(sd)):
    for x in np.arange(mean-(5*sd[i]),mean+(5*sd[i])+1,0.01):
        dt[f"X{i+1}"].append(x)
        dt[f"Y{i+1}"].append(calcNorm(sd[i],mean,x))

#plotting all 3 graphs        
plt.plot(dt["X1"],dt["Y1"],color="blue",label="sd=3 mean=0")
plt.plot(dt["X2"],dt["Y2"],color="red",label="sd=2 mean=0")
plt.plot(dt["X3"],dt["Y3"],color="black",label="sd=1 mean=0")
plt.title("Normal Distribution")
plt.xlabel("X(linear scale)")
plt.ylabel("Y(linear scale)")
plt.figtext(0.5, 0.01, 'This graph is obtained by keeping mean constant and varying Standard deviation. From this graph we can infer that,\n when we reduce the standard deviation the height of the graph becomes lower and vice versa. The height of the peak is inversely proprtional to Standard deviation.', fontsize=12, color='black', ha='center')
plt.legend()
plt.show()    


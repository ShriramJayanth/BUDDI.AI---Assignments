import math
import numpy as np
import matplotlib.pyplot as plt
# CalcNorm is a function to calculate the y for the given x using normal distribution formula
def calcNorm(sd,mean,x):
    part1=(1/((sd)*(2*math.pi)))
    part2=math.exp((-1/2)*(((x-mean)/sd))**2)
    return(part1*part2)

sd=1
mean=[0,1,2]
dt={
    "X1":[],
    "Y1":[],
    "X2":[],
    "Y2":[],
    "X3":[],
    "Y3":[],
}

#A loop to compute the values for  given x by keeping Sd constant and varying mean
for i in range(len(mean)):
    for x in np.arange(mean[i]-(5*sd),mean[i]+(5*sd)+1,0.01):
        dt[f"X{i+1}"].append(x)
        dt[f"Y{i+1}"].append(calcNorm(sd,mean[i],x))

#plotting all 3 graphs        
plt.plot(dt["X1"],dt["Y1"],color="blue",label="mean=0 sd=1")
plt.plot(dt["X2"],dt["Y2"],color="red",label="mean=1 sd=1")
plt.plot(dt["X3"],dt["Y3"],color="black",label="mean=2 sd=1s")
plt.title("Normal Distribution")
plt.xlabel("X(linear scale)")
plt.ylabel("Y(linear scale)")
plt.figtext(0.5, 0.01, 'This graph is obtained by keeping Standard deviation constant and varying mean .From this graph we can infer that,\n mean value determines the position of the peak', fontsize=12, color='black', ha='center')
plt.legend()
plt.show()    

#μσ
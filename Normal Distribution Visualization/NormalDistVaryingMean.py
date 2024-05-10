import math
import numpy as np
import matplotlib.pyplot as plt

# Function to calculate the y for the given x using the normal distribution formula
def calcNorm(sd, mean, x):
    part1=(1/((sd)*(2*math.pi)))
    part2=math.exp((-1 / 2)*(((x-mean)/sd))**2)
    return part1*part2

# Parameters
standardDeviation=1
meanValues=[0,1,2]

# Dictionary to store data for plotting
data={"X1":[],"Y1":[], "X2":[], "Y2":[], "X3":[], "Y3":[]}

# Loop to compute the values for given x by keeping Sd constant and varying mean
for i in range(len(meanValues)):
    for x in np.arange(meanValues[i]-(5*standardDeviation),meanValues[i]+(5*standardDeviation)+1,0.01):
        data[f"X{i+1}"].append(x)
        data[f"Y{i+1}"].append(calcNorm(standardDeviation,meanValues[i],x))

# Plotting all three graphs
plt.plot(data["X1"],data["Y1"],color="blue",label="mean=0 sd=1")
plt.plot(data["X2"],data["Y2"],color="red",label="mean=1 sd=1")
plt.plot(data["X3"],data["Y3"],color="black",label="mean=2 sd=1s")
plt.title("Normal Distribution")
plt.xlabel("X (linear scale)")
plt.ylabel("Y (linear scale)")
plt.figtext(0.5, 0.01, 'This graph is obtained by keeping standard deviation constant and varying mean. From this graph, we can infer that the mean value determines the position of the peak.', fontsize=12, color='black', ha='center')
plt.legend()
plt.show()  

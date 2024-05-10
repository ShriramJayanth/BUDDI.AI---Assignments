import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

def monteCarloSimulation(counts, a):
    # List to store estimated values of Pi
    piCap=[]
    
    # Iterate over different number of darts
    for i in counts:
        # Generate random points within the square [-a/2, a/2] x [-a/2, a/2]
        randomValues=[[np.random.uniform(-a/2,a/2),np.random.uniform(-a/2, a/2)] for j in range(i)]
        
        # Count the number of points falling within the quarter circle
        onTarget=sum(1 for j in randomValues if (j[0]**2 + j[1]**2)**(1/2) <= a/2)
        
        # Estimate Pi using the Monte Carlo method
        piCap.append((onTarget/i)*4)
    
    return piCap

def plotMonteCarloSimulation(counts, piCap):
    # Create a DataFrame to store the results
    mc=pd.DataFrame({"Number of Darts":counts,"Estimated value of Pi":piCap})
    mc.to_csv('MonteCarloUniform.csv',index=False)
    
    # Plot the results
    plt.plot(counts,piCap,marker='D',linestyle='-',label='Estimated Value of Pi')
    plt.axhline(y=math.pi,color="black",linestyle="--")
    plt.title("Monte Carlo Simulations")
    plt.yscale('linear')
    plt.xscale('log')
    plt.xlabel('Number of Darts')
    plt.ylabel('Estimated Value of Pi')
    plt.figtext(0.5, 0.01, 'As the number of random points increases, the estimated value of π converges towards the actual value of π (approximately 3.14159). This convergence demonstrates the principle of the Monte Carlo method, where more iterations lead to more accurate estimates.', fontsize=12, color='black', ha='center')
    plt.show()

# Set parameters
a=1
counts=[10,100,1000,10000,100000,1000000]

# Perform the Monte Carlo simulation
piCap=monteCarloSimulation(counts,a)

# Print results for inspection
print(piCap)
print(counts)

# Plot the results of the Monte Carlo simulation
plotMonteCarloSimulation(counts, piCap)

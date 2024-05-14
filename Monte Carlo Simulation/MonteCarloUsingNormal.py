import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import math

# Function to perform the Monte Carlo simulation
def monteCarloSimulation(counts):
    piCap=[]
    for n in counts:
        X=[]
        Y=[]
        # Generate random points within the square [-0.5, 0.5] x [-0.5, 0.5]
        for j in range(n):
            x=np.random.normal(0,1)
            y=np.random.normal(0.1)
            while(x< -0.5 or x>0.5):
                x=np.random.normal(0, 1)
            while(y< -0.5 or y>0.5):
                y=np.random.normal(0.1)
            X.append(x)
            Y.append(y)              
        # Count the number of points falling within the quarter circle
        onTarget=0    
        for j,k in zip(X, Y):
            if(((j**2 + k**2)**(1/2))<=0.5):
                onTarget+=1
        # Estimate Pi using the Monte Carlo method
        piCap.append((onTarget/n)*4)        
    return piCap

# Function to plot the Monte Carlo simulation results
def plotMonteCarloSimulation(counts,piCap):
    # Create a DataFrame to store the results
    mc = pd.DataFrame({"number_of_darts": counts, "estimated_value_of_pi": piCap})
    mc.to_csv('MonteCarloNormal.csv', index=False)
    # Plot the results
    plt.plot(counts, piCap, marker='D', linestyle='-', label='Estimated Value of Pi')
    plt.axhline(y=math.pi, color="black", linestyle="--")
    plt.title("Monte Carlo Simulations")
    plt.xscale('log')
    plt.xlabel('Number of Darts')
    plt.ylabel('Estimated Value of Pi')
    plt.legend()
    # Add a description below the plot
    plt.figtext(0.5, 0.01, 'As the number of random points increases, the estimated value of π converges towards the actual value of π (approximately 3.14159).\n This convergence demonstrates the principle of the Monte Carlo method, where more iterations lead to more accurate estimates.', fontsize=12, color='red', ha='center')
    plt.show()

# Main part of the code
a = 1
counts = [10, 100, 1000, 10000, 100000, 1000000]
# Perform the Monte Carlo simulation
piCap = monteCarloSimulation(counts)
print(piCap)
print(counts)
# Plot the results of the Monte Carlo simulation
plotMonteCarloSimulation(counts, piCap)

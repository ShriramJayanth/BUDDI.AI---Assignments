import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Function to minimize epsilon by iterating over possible values of b1 and b2
def minimize_epsilon(data_set):
    min_epsilon=2**30
    b1_optimal=-1
    b2_optimal=-1
    fn = {"b1":[], "b2":[], "epsilon":[]}
    
    # Iterate over possible values of b1 and b2
    for i in np.arange(-1,1,0.01):
        for j in np.arange(-1,1,0.01):
            b1_val=i
            b2_val=j
            count=0
            fn["b1"].append(i)
            fn["b2"].append(j)
            
            # Calculate epsilon for current values of b1 and b2
            for k in range(len(data_set)):
                hyp_val=(b1_val*data_set.iloc[k].X)+(b2_val*(data_set.iloc[k].X) ** 2)
                count+=abs(data_set.iloc[k].Y - hyp_val)
                
            fn["epsilon"].append(count)    
            
            # Update minimum epsilon and optimal b1 and b2 if current epsilon is smaller
            if count<min_epsilon:
                min_epsilon=count
                b1_optimal=b1_val
                b2_optimal=b2_val
                
    print(f"MinEpsilon is {min_epsilon} and it occurs when b1 is {b1_optimal} and b2 is {b2_optimal}")           
    epsilon_vals=pd.DataFrame(fn)
    
    return epsilon_vals, b1_optimal, b2_optimal

# Function to plot the surface plot of epsilon values against b1 and b2
def plot_surface(epsilon_vals):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(epsilon_vals.b1,epsilon_vals.b2,epsilon_vals.epsilon,cmap='viridis',edgecolor='none')
    ax.set_xlabel('b1')
    ax.set_ylabel('b2')
    ax.set_zlabel('Epsilon')
    ax.set_title('Surface Plot')
    plt.show() 

# Sample data
data = {
    "X": [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0],
    "Y": [7.0, 2.0, 0.0, 0.0, 0.0, 2.0, 7.0]
}

data_set = pd.DataFrame(data)

# Find optimal values of b1 and b2 that minimize epsilon
epsilon_vals,b1_optimal,b2_optimal=minimize_epsilon(data_set)

# Plot the surface plot of epsilon values
plot_surface(epsilon_vals)

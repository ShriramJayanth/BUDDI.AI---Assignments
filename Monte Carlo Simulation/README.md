# Monte Carlo Simulations


Monte Carlo methods are computational techniques that rely on repeated random sampling to obtain numerical results. These methods are particularly useful for problems that are deterministic in nature but difficult to solve using traditional mathematical approaches. By simulating a large number of random scenarios or events, Monte Carlo methods can provide estimates or solutions to complex problems.

Here's how Monte Carlo methods work in general:

Problem Formulation: Identify a problem that can be represented probabilistically or involves uncertainty.

Random Sampling: Generate a large number of random samples or scenarios according to a specified probability distribution or range.

Simulation: For each sample or scenario, perform calculations or simulations to evaluate the problem.

Aggregation: Aggregate the results obtained from all the samples to obtain an overall estimate or solution.

Analysis: Analyze the aggregated results, such as calculating statistical measures or drawing conclusions.

## In context of my code:

Monte Carlo Simulation Function (monteCarloSimulation):

This function implements a Monte Carlo method to estimate the value of π.
It generates random points within a square region and counts the proportion of points falling within a quarter circle inscribed within the square.
By comparing the area of the quarter circle to the area of the square, it estimates the value of π using the formula: π ≈ (points within quarter circle / total points) * 4.
This process is repeated for different numbers of random points (darts), and the estimated values of π are stored in the piCap list.
Plotting Function (plotMonteCarloSimulation):

This function visualizes the results of the Monte Carlo simulation.
It plots the estimated values of π against the number of darts used in the simulation.
Additionally, it adds a reference line representing the actual value of π for comparison.
Main Code:

The main code sets the parameters for the Monte Carlo simulation, such as the side length of the square (a) and the number of darts for each simulation (counts).
It calls the monteCarloSimulation function to perform the simulation and obtain the estimated values of π.
It then calls the plotMonteCarloSimulation function to visualize the results of the simulation.
In summary, the provided code demonstrates how Monte Carlo methods can be applied to estimate the value of π by simulating random points within a geometric shape and leveraging the concept of geometric probability to obtain an approximation of the mathematical constant.
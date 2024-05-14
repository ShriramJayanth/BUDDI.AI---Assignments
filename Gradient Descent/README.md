
# Gradient Descent

Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent (negative gradient) of the function. It's commonly used in machine learning for optimizing the parameters of a model to minimize a loss function.

Here's a high-level explanation of gradient descent:

Initialization: Start with initial values for the parameters (weights or coefficients) of the model.

Compute Gradient: Calculate the gradient of the loss function with respect to each parameter. The gradient points in the direction of the steepest increase of the function.

Update Parameters: Adjust the parameters in the direction opposite to the gradient to minimize the loss function. This adjustment is made by multiplying the gradient by a small scalar value called the learning rate and subtracting it from the current parameter values.

Repeat: Repeat steps 2 and 3 until convergence criteria are met, such as reaching a maximum number of iterations or the change in the loss function becoming small.

## In context of my code:

Initialization:initialize random values for the parameters beta0 and beta1.

Compute Gradient:compute the gradient of the loss function with respect to beta0 and beta1 using the findNewBeta0 and findNewBeta1 functions.

Update Parameters: update beta0 and beta1 by subtracting the gradient multiplied by the learning rate eta from the current parameter values.

Repeat: repeat the process until convergence, where convergence is determined by a small change in the error or reaching a maximum number of epochs.

This process iteratively adjusts the parameters of the model to minimize the error between the predicted and actual values, ultimately finding the optimal parameters for the linear regression model.

![image](https://github.com/ShriramJayanth/BUDDI.AI---Assignments/assets/131799455/92af9c56-926e-4dec-b799-0cad5ca3b8d3)
![image](https://github.com/ShriramJayanth/BUDDI.AI---Assignments/assets/131799455/9f9c0ddb-a6c5-4edf-a200-2c21eeb4b2eb)



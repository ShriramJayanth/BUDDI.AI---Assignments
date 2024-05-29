import numpy as np
import matplotlib.pyplot as plt

# Calculating Beta values using closed form solution
def findBetaClosedForm(xMatrix:np.array,yActual:np.array)->np.array:
    # Transpose of the xMatrix
    xTranspose=xMatrix.transpose()
    # Calculate the first part of the closed form solution
    firstPart=np.linalg.inv(np.matmul(xTranspose,xMatrix))
    # Calculate the second part of the closed form solution
    secondPart=np.matmul(xTranspose,yActual)
    # Return the product of the two parts
    return np.matmul(firstPart,secondPart)

# Function to calculate the linear function
def linearFunction(x:np.array,beta0:float,beta1:float)->np.array:
    return (beta0+(beta1*x))

# Function to calculate the perpendicular (decision boundary)
def perpendicular(x:np.array,beta0:float,beta1:float)->np.array:
    return(beta0+((-1/beta1)*x))

#Indicator function to change class labels into numbers
def indicator(yActual:np.array)->np.array:
    yNew=[]
    for i in yActual:
        if(i=="b"):
            yNew.append(1)
        else:
            yNew.append(0)
    return np.array(yNew)

# sigmoid function
def sigmoid(x:np.array)->np.array:
    return 1 / (1 + np.exp(-x))

#main function
def main():
    xActual = np.linspace(-5,5,100)
    yActual = np.where(xActual<0,'a','b')
    print(xActual.shape,yActual.shape)
    xMatrix=np.array([[1,i] for i in xActual])
    #using indicator function to transform y to numericals
    yNew=indicator(yActual)
    #calculating beta values using closed form solutions
    betaVals=findBetaClosedForm(xMatrix,yNew)
    yPred=linearFunction(xActual,betaVals[0],betaVals[1])
    yPerp=perpendicular(xActual,betaVals[0],betaVals[1])
    # calculating y values using sigmoid function
    xPlot=xActual
    yPlot=linearFunction(xPlot,betaVals[0],betaVals[1])
    yPlot=sigmoid(yPlot)
    # plotting blue dots and green dots
    plt.scatter(xActual[yActual=='a'], yNew[yActual=='a'], marker="o", color="b", label='Class a')
    plt.scatter(xActual[yActual=='b'], yNew[yActual=='b'], marker="o", color="g", label='Class b')
    # plotting sigmoid function
    plt.plot(xPlot,yPlot,"r-",label="Sigmoid function")
    # plotting regressor
    plt.plot(xActual,yPred,"k-",label="Regressor")
    # plotting decision boundary
    plt.plot(xActual,yPerp,"y--",label="Decision Boundary")
    # adding labels
    plt.xlabel("X (Linear scale)")
    plt.ylabel("Y (Linear scale)")
    plt.title("Logistic Regression")
    # adding desc
    plt.figtext(0.5, 0.01, "This plot visualizes logistic regression's transformation of a linear predictor into probabilities using a sigmoid function, showcasing a decision boundary separating two classes based on input features.", fontsize=12, color='black', ha='center')
    plt.ylim(-0.5,1.5)
    plt.xlim(-5,5)
    plt.legend()
    plt.show()
    
if(__name__=="__main__"):
    main()
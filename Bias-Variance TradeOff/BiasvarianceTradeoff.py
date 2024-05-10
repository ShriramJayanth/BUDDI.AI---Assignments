import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# A function to find Error
def findError(betaVals, x, y):
    bias = []
    # calculating the predicted values
    for j in range(len(x[:,1])):
        pred=0
        for k in range(len(betaVals)):
            pred+=betaVals[k]*(x[:,1][j]**k)
        bias.append(pred) 
    bias=np.array(bias)   
    # returning the mean square error
    return np.mean((bias-y)**2)

# A function to find the predicted values using Beta values
def findPred(betaVals,xActual):
    x=xActual[:,1]
    predictions = np.zeros_like(x)
    for k in range(len(betaVals)):
        predictions+=betaVals[k]*(x ** k)
    return predictions

# A function to find beta values for a given degree
def findBeta(degree,xData,yData):
    #slicing actual x values
    x = xData[:, 0:degree+1]
    xTranspose=x.transpose()
    #closed form solution
    firstPart = np.linalg.inv(np.matmul(xTranspose, x))
    secondPart = np.matmul(xTranspose, yData)
    return np.matmul(firstPart, secondPart)

# A function to plot the ploynomial regression with multiple degrees
def plotPolyNomialRegression(xTrain,yTrain,xActual,yActual,degree):
    col=["red","blue","black","orange","purple"]
    # calculating the BetaVals using ((X*Xt)inv)*(Xt*Y)
    for i in range(1, degree+1):
        betaVals = findBeta(i,xTrain,yTrain)
        pred = findPred(betaVals,xActual)
        plt.plot(xActual[:,1], pred, label=f'Degree {i}',color=col[i])
    # calculating expected values using lagarange's polynomial 
    lag = [lagrange(xActual[:, 1], yActual, xi, len(xActual)) for xi in xActual[:, 1]]
    #plotting lagrange polynomial values
    plt.plot(xActual[:, 1], lag,label="Lagrange's values")
    plt.scatter(xTrain[:, 1], yTrain, color='black', label='Actual Data')
    plt.xlabel('X(Linear Scale)')
    plt.ylabel('Y(Linear Scale)')
    plt.title('Polynomial Regression Predictions')
    plt.figtext(0.5, 0.01, 'The graph shows the Predicted values of polynomials(for given x) of various degree by finding their parameters using closed form solutions when dots represent the original data.', fontsize=12, color='black', ha='center')
    plt.legend()
    plt.show()


# A function to plot Bias-Variance TradeOff graph
def plotBiasVarianceGraph(xTrain,yTrain,xTest,yTest,degree):
    bias=[]
    variance=[]
    # calculating the BetaVals using ((X*Xt)inv)*(Xt*Y)
    for i in range(1, degree+1):
        betaVals=findBeta(i,xTrain,yTrain)
        #calculating bias
        ithbias=findError(betaVals,xTrain,yTrain)
        #calculating variance
        ithvariance=findError(betaVals,xTest,yTest)
        bias.append(ithbias)
        variance.append(ithvariance)
    deg=[i+1 for i in range(degree)] 
    # plotting Bias and Variance 
    plt.plot(deg,bias,marker="o",color="blue",label="bias")
    plt.plot(deg,variance,marker="o",color="red",label="variance")
    plt.xlabel('Degree of polynomial(Linear Scale)')
    plt.ylabel('MSE(Linear Scale)')
    plt.title('Bias Variance TradeOff')
    plt.figtext(0.5, 0.01, 'This graph illustrates the bias-variance tradeoff in polynomial regression models of varying complexity. the point at which the difference between the bias and variance is minimum is the sweet spot.', fontsize=12, color='black', ha='center')
    plt.legend()
    plt.show()

# Generating x and y values
def xyGenerator(start,stop,num,degree):
    xActual=[]
    yActual=[]
    for i in np.linspace(start,stop,num):
        xSamples = [i**j for j in range(0,degree+1)]
        xActual.append(xSamples)
        yGenerator = (2 * (i**4)) - (3 * (i**3)) + (7 * (i**2)) - (23 * i) + 8 + np.random.normal(0, 3)
        yActual.append(yGenerator)
    xActual=np.array(xActual)
    yActual=np.array(yActual)
    return xActual,yActual        

# A function to calculate y for a given x using Lagrange polynomial
def lagrange(xActual,yActual,xi,n):
    res=0.0
    # calculating expected y using lagarange's polynomial
    for i in range(n):
        t=yActual[i]
        for j in range(n):
            if(i!=j):
                t=t*(xi-xActual[j])/(xActual[i]-xActual[j])
        res+=t
    return res            

degree=4

xActual,yActual=xyGenerator(-5,5,100,degree)
   
#splitting the data
xTrain, xTest, yTrain, yTest = train_test_split(xActual, yActual, test_size=0.2)

# plotPolyNomialRegression(xTrain,yTrain,xActual,yActual,degree)
plotBiasVarianceGraph(xTrain,yTrain,xTest,yTest,degree)


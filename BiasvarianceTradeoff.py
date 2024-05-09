import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def findVariance(degree, betaVals, xTest, yTest):
    predictions = []
    for j in range(len(xTest[:,1])):
        pred=0
        for k in range(degree+1):
            pred+=betaVals[k]*(xTest[:,1][j]**k)
        predictions.append(pred) 
    predictions=np.array(predictions)    
    return np.mean((predictions-yTest)**2)

def findBias(degree, betaVals, xTrain, yTrain):
    predictions = []
    for j in range(len(xTrain[:,1])):
        pred=0
        for k in range(degree+1):
            pred+=betaVals[k]*(xTrain[:,1][j]**k)
        predictions.append(pred) 
    predictions=np.array(predictions)    
    return np.mean((predictions-yTrain)**2)

def findPred(degree, betaVals):
    x = np.linspace(-5, 5, 100)
    predictions = np.zeros_like(x)
    for k in range(degree + 1):
        predictions += betaVals[k] * (x ** k)
    return x, predictions

def plotPolyNomialRegression(xTrain,yTrain,xActual,yActual):
    col=["red","blue","black","orange","purple"]
    for i in range(1, 5):
     SlicedX = xTrain[:, 0:i+1]
     FirstPart = np.linalg.inv(np.matmul(SlicedX.transpose(), SlicedX))
     SecondPart = np.matmul(SlicedX.transpose(), yTrain)
     betaVals = np.matmul(FirstPart, SecondPart)
     x, pred = findPred(i, betaVals)
     plt.plot(x, pred, label=f'Degree {i}',color=col[i])
    lag = [lagrange(xActual[:, 1], yActual, xi, len(xActual)) for xi in xActual[:, 1]]
    plt.plot(xActual[:, 1], lag,label="Lagrange's values")
    plt.scatter(xTrain[:, 1], yTrain, color='black', label='Actual Data')
    plt.xlabel('X(Linear Scale)')
    plt.ylabel('Y(Linear Scale)')
    plt.title('Polynomial Regression Predictions')
    plt.figtext(0.5, 0.01, 'The graph shows the Predicted values of polynomials(for given x) of various degree by finding their parameters using closed form solutions when dots represent the original data.', fontsize=12, color='black', ha='center')
    plt.legend()
    plt.show()



def plotBiasVarianceGraph(xTrain,yTrain,xTest,yTest):
    bias=[]
    variance=[]
    for i in range(1, 5):
     SlicedX = xTrain[:, 0:i+1]
     FirstPart = np.linalg.inv(np.matmul(SlicedX.transpose(), SlicedX))
     SecondPart = np.matmul(SlicedX.transpose(), yTrain)
     betaVals = np.matmul(FirstPart, SecondPart)
     ithbias=findBias(i,betaVals,xTrain,yTrain)
     ithvariance=findVariance(i,betaVals,xTest,yTest)
     bias.append(ithbias)
     variance.append(ithvariance)
    plt.plot([1,2,3,4],bias,color="blue",label="bias")
    plt.plot([1,2,3,4],variance,color="red",label="variance")
    plt.xlabel('Degree of polynomial(Linear Scale)')
    plt.ylabel('MSE(Linear Scale)')
    plt.title('Bias Variance TradeOff')
    plt.figtext(0.5, 0.01, 'This graph illustrates the bias-variance tradeoff in polynomial regression models of varying complexity.', fontsize=12, color='black', ha='center')
    plt.legend()
    plt.show()

def lagrange(X,Y,xi,n):
    res=0.0
    for i in range(n):
        t=Y[i]
        for j in range(n):
            if(i!=j):
                t=t*(xi-X[j])/(X[i]-X[j])
        res+=t
    return res            


xActual=[]
yActual=[]

for i in np.linspace(-5, 5, 101):
    xSamples = [i**j for j in range(0,5)]
    xActual.append(xSamples)
    yGenerator = (2 * (i**4)) - (3 * (i**3)) + (7 * (i**2)) - (23 * i) + 8 + np.random.normal(0, 3)
    yActual.append(yGenerator)

xActual=np.array(xActual)
yActual=np.array(yActual)    

xTrain, xTest, yTrain, yTest = train_test_split(xActual, yActual, test_size=0.2)

# plotPolyNomialRegression(xTrain,yTrain,xActual,yActual)
plotBiasVarianceGraph(xTrain,yTrain,xTest,yTest)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# generating x and y values
def xyGenerator(start:int,stop:int,num:int)->list[np.array,np.array]:
    xActual=[]
    yActual=[]
    for i in np.linspace(start,stop,num):
        xActual.append(i)
        yActual.append((2*i)-3+np.random.normal(0,5))
    return np.array(xActual),np.array(yActual)

# Calculating Beta values using closed form solution
def findBetaClosedForm(xMatrix:np.array,yActual:np.array)->np.array:
    xTranspose=xMatrix.transpose()
    firstPart = np.linalg.inv(np.matmul(xTranspose, xMatrix))
    secondPart = np.matmul(xTranspose, yActual)
    return np.matmul(firstPart, secondPart)

# Calculating mean sum of error square
def findError(xActual:np.array,yActual:np.array,beta0:float,beta1:float)->int:
    predvals=(xActual*beta1)+beta0
    return np.mean((yActual-predvals)**2)

# Function to find new beta1 after performing partial differentiation
def findNewBeta1(xActual:np.array, yActual:np.array, beta0:float, beta1:float)->int:
    # substituting x on equation obtained by partial differentiation on beta1
    predVals=(beta1*xActual)+beta0
    grad_beta1=-2*np.mean(xActual*(yActual-predVals))
    return grad_beta1

# Function to find new beta0 after performing partial differentiation
def findNewBeta0(xActual:np.array,yActual:np.array,beta0:float,beta1:float)->int:
    # substituting x on equation obtained by partial differentiation on beta0
    predVals=(beta1*xActual)+beta0
    grad_beta0=-2*np.mean(yActual-predVals)
    return grad_beta0

def printDetails(title:str,beta0:float,beta1:float,bias:float,variance:float,epoch:int,eta:float):
    print(f"\nBeta values of {title} Gradient Descent\n")
    print(f"Bo: {beta0}")
    print(f"B1: {beta1}")     
    print(f"Bias: {bias}")
    print(f"Variance: {variance}")
    print(f"epochs: {epoch}")
    print(f"eta: {eta}")

# finding beta values using gradient descent
def gradientDescent(xTrain:np.array,yTrain:np.array,xTest:np.array,yTest:np.array,eta:float)->None:
    bias=[]
    variance=[]
    epo=[]
    #generate random b1 and b2
    beta0=np.random.normal(0,1)
    beta1=np.random.normal(0,1)
    error=findError(xTrain,yTrain,beta0,beta1)
    flag=True
    epoch=0
    #gradient descent loop
    while(flag):
        #calculating new b1 and b2
        beta0-=(eta*findNewBeta0(xTrain,yTrain,beta0,beta1))
        beta1-=(eta*findNewBeta1(xTrain,yTrain,beta0,beta1))
        # storing epochs,bias,variances for all iterations
        bias.append(findError(xTrain,yTrain,beta0,beta1))
        variance.append(findError(xTest,yTest,beta0,beta1))
        epoch+=1
        epo.append(epoch)
        # print(f"b0:{beta0} b1:{beta1}")
        #loop runs till difference between new error and old error is less than 0.001 
        if(abs(error-bias[-1])<10e-6 or bias[-1]<0.0001):
            flag=False
        else:
            error=bias[-1]     
    #printing details
    printDetails("batch method",beta0,beta1,bias[-1],variance[-1],epoch,eta)           
    # plotting Error Epochs graph        
    plt.plot(epo[5:],bias[5:],label=f"bias at eta : {eta}")
    plt.plot(epo[5:],variance[5:],label=f"variance at eta : {eta}")
    #adding labels
    addLabels("Epochs (linear scale)","Epsilon (linear scale)","Error Rate During Training using batch method gradient descent",'This graph illustrates the training loss of a machine learning model over successive training epochs. As training progresses, the model learns to minimize its loss function,\n resulting in a decrease in error over time. Tshis visualization provides insights into the training dynamics and convergence behavior of the model')

def miniBatchGradientDescent(xTrain: np.array, yTrain: np.array, xTest: np.array, yTest: np.array, eta: float, batch_size: int) -> None:
    steps = []
    bias = []
    variance = []
    step = 0
    beta0=np.random.normal(0,1)
    beta1=np.random.normal(0,1)
    error=error=findError(xTrain,yTrain,beta0,beta1)
    #finding number of batches
    numBatches=len(xTrain)//batch_size
    flag=True
    step=0
    epoch=0
    #gradient loop
    while(flag):
        for batch_index in range(numBatches):
            # Create batch
            start_index = batch_index * batch_size
            end_index = min((batch_index + 1) * batch_size, len(xTrain))
            xTemp = xTrain[start_index:end_index]
            yTemp = yTrain[start_index:end_index]
            # Update parameters
            beta0 -= (eta * findNewBeta0(xTemp, yTemp, beta0, beta1))
            beta1 -= (eta * findNewBeta1(xTemp, yTemp, beta0, beta1))
        # Calculate error for training and testing data after each epoch
            bias.append(findError(xTrain, yTrain, beta0, beta1))
            variance.append(findError(xTest, yTest, beta0, beta1))
            step+=1
            steps.append(step) 
        epoch+=1      #increament epochs
        if(abs(error-bias[-1])<10e-6 or bias[-1]<0.0001):
            flag=False
        else:
            error=bias[-1]
    #printing details        
    printDetails("Mini batch method",beta0,beta1,bias[-1],variance[-1],epoch,eta)
    # plotting Error Epochs graph      
    plt.plot(steps[5:], bias[5:], label=f"bias at eta : {eta}")
    plt.plot(steps[5:], variance[5:], label=f"variance at eta : {eta}")
    #adding labels
    addLabels("Steps (linear scale)","Epsilon (linear scale)","Error Rate During Training using mini Batch gradient descent",'This graph illustrates the training loss of a machine learning model over successive training epochs. As training progresses, the model learns to minimize its loss function,\n resulting in a decrease in error over time. This visualization provides insights into the training dynamics and convergence behavior of the model.')

def stochiasticGradientDescent(xTrain:np.array,yTrain:np.array,xTest:np.array,yTest:np.array,eta:float)->None:
    steps=[]
    bias=[]
    variance=[]
    #generate random b1 and b2
    beta0=np.random.normal(0,1)
    beta1=np.random.normal(0,1)
    error=findError(xTrain,yTrain,beta0,beta1)
    step=0
    random_indices = np.random.choice(len(xTrain), size=len(xTrain), replace=True)
    epoch=0
    flag=True
    #gradient descent loop
    while(flag):
        for i in random_indices:
            beta0-=(eta*findNewBeta0(xTrain[i],yTrain[i],beta0,beta1))
            beta1-=(eta*findNewBeta1(xTrain[i],yTrain[i],beta0,beta1))
            step+=1
            steps.append(step)
            bias.append(findError(xTrain,yTrain,beta0,beta1))   
            variance.append(findError(xTest,yTest,beta0,beta1)) 
        epoch+=1
        if(abs(error-bias[-1])<10e-6 or bias[-1]<0.0001):
            flag=False
        else:
            error=bias[-1]
    #printing details
    printDetails("Stochiastic",beta0,beta1,bias[-1],variance[-1],epoch,eta)  
    # plotting Error Epochs graph         
    plt.plot(steps[5:],bias[5:],label=f"bias at eta : {eta}")
    plt.plot(steps[5:],variance[5:],label=f"variance at eta : {eta}")
    #adding labels
    addLabels("Steps (linear scale)","Epsilon (linear scale)","Error Rate During Training using stochiastic gradient descent",'This graph illustrates the training loss of a machine learning model over successive training epochs. As training progresses, the model learns to minimize its loss function,\n resulting in a decrease in error over time. This visualization provides insights into the training dynamics and convergence behavior of the model.')

# function to plot Error vs Epochs graph
def addLabels(xlabel:str,ylabel:str,title:str,desc:str)->None:
    # adding labels
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    # adding description
    plt.figtext(0.5, 0.01, desc, fontsize=12, color='black', ha='center')
    plt.show()

def main():
    xActual,yActual=xyGenerator(-5,5,1000)
    xTrain, xTest, yTrain, yTest = train_test_split(xActual, yActual, test_size=0.2)
    #finding beta values using gradient descent
    xMatrix=[]
    for i in xTrain:
        xMatrix.append([1,i])
    xMatrix=np.array(xMatrix)    
    #finding beta values using closed form solution
    cf=findBetaClosedForm(xMatrix,yTrain)
    bias=findError(xTrain,yTrain,cf[0],cf[1])
    variance=findError(xTest,yTest,cf[0],cf[1])
    print("Beta values of Closed form solution\n")
    print(f"Bo: {cf[0]}")
    print(f"B1: {cf[1]}")
    print(f"Bias:{bias}")
    print(f"Variance:{variance}")
    gradientDescent(xTrain,yTrain,xTest,yTest,0.001)
    stochiasticGradientDescent(xTrain,yTrain,xTest,yTest,0.001)
    miniBatchGradientDescent(xTrain,yTrain,xTest,yTest,0.001,50)

if __name__=="__main__":
    main()
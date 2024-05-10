import numpy as np
import matplotlib.pyplot as plt

# generating x and y values
def xyGenerator(start,stop,num):
    xActual=[]
    yActual=[]
    for i in np.linspace(start,stop,num):
        xActual.append(i)
        yActual.append((2*i)-3+np.random.normal(0,5))   
    return np.array(xActual),np.array(yActual)

# Calculating Beta values using complete form solution
def findBetaCompleteForm(xActual,yActual,degree):
    xMatrix=[]
    for i in xActual:
        xMatrix.append([i**j for j in range(degree+1)])
    # closed form solution    
    xMatrix=np.array(xMatrix)    
    xTranspose=xMatrix.transpose()
    firstPart = np.linalg.inv(np.matmul(xTranspose, xMatrix))
    secondPart = np.matmul(xTranspose, yActual)
    return np.matmul(firstPart, secondPart)

# Calculating sum of error square
def findError(xActual,yActual,beta1,beta2):
    epsilon=0
    for i in range(len(xActual)):
        predVal=(xActual[i]*beta2)+beta1
        epsilon+=((predVal-yActual[i])**2)
    return epsilon
    
# Function to find new beta1 after performing partial differentiation
def findNewBeta1(xActual, yActual, beta0, beta1):
    # substituting x on equation obtained by partial differentiation on beta1
    predVals=(beta1*xActual)+beta0
    grad_beta1=-2*np.mean(xActual*(yActual-predVals))
    return grad_beta1

# Function to find new beta0 after performing partial differentiation
def findNewBeta0(xActual,yActual,beta0,beta1):
    # substituting x on equation obtained by partial differentiation on beta0
    predVals=(beta1*xActual)+beta0
    grad_beta0=-2*np.mean(yActual-predVals)
    return grad_beta0


# finding beta values using gradient descent
def findBetaGradientDescent(xActual,yActual):
    #generate random b1 and b2
    beta0=np.random.normal(0,1)
    beta1=np.random.normal(0,1)
    eta=0.01
    error=findError(xActual,yActual,beta0,beta1)
    flag=True
    #gradient descent loop
    while(flag):
        #calculating new b1 and b2
        newBeta0=beta0-(eta*findNewBeta0(xActual,yActual,beta0,beta1))
        newBeta1=beta1-(eta*findNewBeta1(xActual,yActual,beta0,beta1))
        newError=findError(xActual,yActual,newBeta0,newBeta1)
        beta0=newBeta0
        beta1=newBeta1
        #loop runs till difference between new error and old error is less than 0.001 
        if(abs(error-newError)<0.0001):
            flag=False
        else:
            error=newError    
        
    return beta0,beta1

def main():
    xActual,yActual=xyGenerator(-5,5,1000)
    gd=findBetaGradientDescent(xActual,yActual)
    cf=findBetaCompleteForm(xActual,yActual,1)
    print("Beta values of Closed form solution\n")
    print(f"Bo: {cf[0]}")
    print(f"B1: {cf[1]}")
    print("\nBeta values of Gradient Descent\n")
    print(f"Bo: {gd[0]}")
    print(f"B1: {gd[1]}")     

if __name__=="__main__":
    main()        







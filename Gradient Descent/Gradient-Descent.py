import numpy as np
import matplotlib.pyplot as plt

# generating x and y values
def xyGenerator(start:int,stop:int,num:int)->list[np.array,np.array]:
    xActual=[]
    yActual=[]
    for i in np.linspace(start,stop,num):
        xActual.append(i)
        yActual.append((2*i)-3+np.random.normal(0,5))   
    return np.array(xActual),np.array(yActual)

# Calculating Beta values using closed form solution
def findBetaClosedForm(xActual:np.array,yActual:np.array,degree:int)->np.array:
    xMatrix=[]
    for i in xActual:
        xMatrix.append([i**j for j in range(degree+1)])
    # closed form solution
    xMatrix=np.array(xMatrix)
    xTranspose=xMatrix.transpose()
    firstPart = np.linalg.inv(np.matmul(xTranspose, xMatrix))
    secondPart = np.matmul(xTranspose, yActual)
    return np.matmul(firstPart, secondPart)

# Calculating mean sum of error square
def findError(xActual:np.array,yActual:np.array,beta0:int,beta1:int)->int:
    predvals=(xActual*beta1)+beta0
    return np.mean((yActual-predvals)**2)

# Function to find new beta1 after performing partial differentiation
def findNewBeta1(xActual:np.array, yActual:np.array, beta0:int, beta1:int)->int:
    # substituting x on equation obtained by partial differentiation on beta1
    predVals=(beta1*xActual)+beta0
    grad_beta1=-2*np.mean(xActual*(yActual-predVals))
    return grad_beta1

# Function to find new beta0 after performing partial differentiation
def findNewBeta0(xActual:np.array,yActual:np.array,beta0:int,beta1:int)->int:
    # substituting x on equation obtained by partial differentiation on beta0
    predVals=(beta1*xActual)+beta0
    grad_beta0=-2*np.mean(yActual-predVals)
    return grad_beta0


# finding beta values using gradient descent
def findBetaGradientDescent(xActual:np.array,yActual:np.array,eta:int)->list[int,int,np.array,np.array,np.array]:
    err=[]
    epo=[]
    b1=[]
    b0=[]
    #generate random b1 and b2
    beta0=np.random.normal(0,1)
    beta1=np.random.normal(0,1)
    error=findError(xActual,yActual,beta0,beta1)
    flag=True
    epoch=0
    #gradient descent loop
    while(flag):
        #calculating new b1 and b2
        newBeta0=beta0-(eta*findNewBeta0(xActual,yActual,beta0,beta1))
        newBeta1=beta1-(eta*findNewBeta1(xActual,yActual,beta0,beta1))
        newError=findError(xActual,yActual,newBeta0,newBeta1)
        beta0=newBeta0
        beta1=newBeta1
        epoch+=1
        # storing epochs,b0,b1,error for all iterations
        epo.append(epoch)
        err.append(newError)
        b0.append(beta0)
        b1.append(beta1)
        # print(f"b0:{beta0} b1:{beta1}")
        #loop runs till difference between new error and old error is less than 0.001 
        if(abs(error-newError)<10e-6 or newError<0.0001):
            flag=False
        else:
            error=newError
    # plotting Error Epochs graph         
    plt.plot(epo[5:],err[5:],label=f"Error at eta : {eta}")
    return epoch,eta,np.array(b0),np.array(b1),np.array(err)

# function to plot Error vs Epochs graph
def plotEpochsErrorGraph(xActual:np.array,yActual:np.array)->None:
    findBetaGradientDescent(xActual,yActual,0.02)
    findBetaGradientDescent(xActual,yActual,0.03)
    findBetaGradientDescent(xActual,yActual,0.04)
    # adding labels
    plt.xlabel("Epochs (linear scale)")
    plt.ylabel("Epsilon (linear scale)")
    plt.title("Error Rate During Training")
    plt.legend()
    # adding description
    plt.figtext(0.5, 0.01, 'This graph illustrates the training loss of a machine learning model over successive training epochs. As training progresses, the model learns to minimize its loss function,\n resulting in a decrease in error over time. This visualization provides insights into the training dynamics and convergence behavior of the model. as learning rate increases number of epochs gets reduced', fontsize=12, color='black', ha='center')
    plt.show()

# function to plot error surface plot in after gradient descent
def plotErrorSurface(b0:np.array,b1:np.array,epsilon:np.array)->None:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(b0,b1,epsilon,cmap="viridis",edgecolor='none')
    # as there are no legends in 3d surface plot , adding a dummy legend
    ax.scatter([], [], [],label='Error Surface')
    # adding labels
    ax.set_xlabel("b0 (Linear scale)")
    ax.set_ylabel("b1 (Linear scale)")
    ax.set_zlabel("Epsilon (Linear scale)")
    ax.set_title("Error surface plot")
    ax.legend()
    # adding description
    plt.figtext(0.5, 0.01, 'This graph shows the Epsilon value for each and every beta values checked while performing gradient descent', fontsize=12, color='black', ha='center')
    plt.show()


def main():
    xActual,yActual=xyGenerator(-5,5,1000)
    #finding beta values using gradient descent
    gd=findBetaGradientDescent(xActual,yActual,0.01)
    #finding beta values using closed form solution
    cf=findBetaClosedForm(xActual,yActual,1)
    error=findError(xActual,yActual,cf[0],cf[1])
    plotEpochsErrorGraph(xActual,yActual)

    print("Beta values of Closed form solution\n")
    print(f"Bo: {cf[0]}")
    print(f"B1: {cf[1]}")
    print(f"Error:{error}")
    print("\nBeta values of Gradient Descent\n")
    print(f"Bo: {gd[-3][-1]}")
    print(f"B1: {gd[-2][-1]}")     
    print(f"Error: {gd[-1][-1]}")
    print(f"Epochs: {gd[0]}")
    print(f"eta: {gd[1]}")

    plotErrorSurface(gd[-3],gd[-2],gd[-1])

if __name__=="__main__":
    main()
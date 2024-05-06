import numpy as np
import matplotlib.pyplot as plt
import math
a=1
count=[10,100,1000,10000,100000,1000000]
Pi_Cap=[]
for i in count:
    X=[]
    Y=[]
    for j in range(i):
        x=np.random.normal(0,1)
        y=np.random.normal(0.1)
        while(x<(-0.5) or x>0.5):
            x=np.random.normal(0,1)
        while(y<(-0.5) or y>0.5):
            y=np.random.normal(0.1)
        X.append(x)
        Y.append(y)        
    # X=np.clip(X_sample,-0.5,0.5)
    # Y=np.clip(Y_sample,-0.5,0.5)
    # while(len(X)<i):
    #     X_sample=[np.random.normal(0,1) for _ in range(i)]
    #     X=np.clip(X_sample,-0.5,0.5)
    # while(len(Y)<i):
    #     Y_sample=[np.random.normal(0,1) for _ in range(i)]
    #     Y=np.clip(Y_sample,-0.5,0.5)
    # while(len(X)>i):
    #     X.pop()
    # while(len(Y)>i):
    #     Y.pop()     
    # print(X,Y)       
    OnTarget=0    
    for j,k in zip(X,Y):
        if(((j**2+k**2)**(1/2))<=0.5):
            OnTarget+=1
    Pi_Cap.append((OnTarget/i)*4)        

print(Pi_Cap)
print(count)
plt.plot(count, Pi_Cap, marker='D', linestyle='-',label='Estimated Value of Pi')
plt.axhline(y=math.pi,color="black",linestyle="--")
plt.title("Monte Carlo Simulations")
plt.xscale('log')
plt.xlabel('Number of Darts')
plt.ylabel('Estimated Value of Pi')
plt.legend()
plt.figtext(0.5, 0.01, 'As the number of random points increases, the estimated value of π converges towards the actual value of π (approximately 3.14159)\n. This convergence demonstrates the principle of the Monte Carlo method, \nwhere more iterations lead to more accurate estimates.', fontsize=12, color='red', ha='center')
plt.show()



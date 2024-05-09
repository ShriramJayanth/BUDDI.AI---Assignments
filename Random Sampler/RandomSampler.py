import random
def drawSample(pdf:dict[str,float],n:int)->list[str]:
    cdf=[]
    total=0
    for i in pdf:
        total+=pdf[i]
        cdf.append(total)
    SampleData=[]    
    for i in range(n):
        RandomNumber=random.uniform(0,1)
        index=-1
        flag=True
        for j in range(len(cdf)):
            if(flag and cdf[j]>RandomNumber):
                index=j
                flag=False
        count=0    
        for j in pdf:
            if(count==index):
                SampleData.append(j)
            count+=1    
    return SampleData            
                
num=int(input())
pdf={}
for i in range(num):
    name=input()
    count=int(input())
    pdf[name]=count
total=0
for i in pdf:
    total+=pdf[i]
for i in pdf:
    pdf[i]=pdf[i]/total
print(drawSample(pdf,10))    

    




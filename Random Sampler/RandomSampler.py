import random

# Function to draw a sample from a probability distribution function (pdf)
def drawSample(pdf: dict[str, float], n: int) -> list[str]:
    # Calculate the cumulative distribution function (cdf)
    cdf=[]
    total=0
    for i in pdf:
        total+=pdf[i]
        cdf.append(total)
    # Generate the sample data based on the cdf
    sampleData = []
    for _ in range(n):
        randomNumber=random.uniform(0, 1)
        index=-1
        flag=True
        # Find the index in cdf where the random number falls
        for j in range(len(cdf)):
            if flag and cdf[j] > randomNumber:
                index=j
                flag=False
        # Append the corresponding value to the sample data
        count=0
        for j in pdf:
            if count==index:
                sampleData.append(j)
            count+=1
    return sampleData

# Main function
def main():
    # Input number of items in the pdf
    num=int(input("Enter the number of items in the probability distribution function: "))
    # Input items and their counts to create the pdf
    pdf={}
    for i in range(num):
        name=input("Enter item name: ")
        count=int(input("Enter item count: "))
        pdf[name]=count
    
    # Normalize the pdf
    total=sum(pdf.values())
    for i in pdf:
        pdf[i]=pdf[i]/total
    
    # Draw a sample from the pdf and print it
    sample=drawSample(pdf,10)
    print("Sample drawn from the probability distribution function:", sample)

# Call the main function
if __name__ == "__main__":
    main()

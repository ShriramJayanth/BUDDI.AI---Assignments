from nltk.tokenize import word_tokenize

# Function to compare two lists of words and count how many words are the same at the same positions and substrings
def compareString(word1:list[str],word2:list[str]):
    count1=0
    count2=0
    for i in range(len(word1)):
        # Compare each word at the same position
        if(word1[i]==word2[i]):
            count1+=1
        if(word1[i] in word2):
            count2+=1    
    return count1,count2       

def stringCnn(word1:str,word2:str):
    # Lists to store similarity counts and word counts
    posWise,subStrWise,count=[],[],[]
    # tokenizing sentences
    word1=word_tokenize(word1)
    word2=word_tokenize(word2)

    # Compare words from the end of word1 and start of word2
    for i in range(len(word1)):
        # Get the first (i+1) words of word2
        tempWord2=word2[:i+1]
        # Get the last (i+1) words of word1, adjusted to match the length of tempWord2
        tempWord1=word1[-(i+1):][:len(tempWord2)]
        # convolving the string
        similaritites=compareString(tempWord1,tempWord2)
        # appending similarities and len of sentence compared
        posWise.append(similaritites[0])
        subStrWise.append(similaritites[1])
        count.append(len(tempWord2))

    # Compare words from the start of word1 and end of word2    
    for i in range(len(word2)):
        tempWord1=word1[:len(word2)][:-i or None]
        # Get the words of word2 starting from i to the end
        tempWord2=word2[i:]
        # convolving the string
        similaritites=compareString(tempWord1,tempWord2)
        # appending similarities and len of sentence compared
        posWise.append(similaritites[0])
        subStrWise.append(similaritites[1])
        count.append(len(tempWord2))
    
    return posWise,subStrWise,count


def main():
    posWise,subStrWise,count=stringCnn("I said this is a simple example but he says it's complex.","This is a simple example.")
    print(f"comparisonLength: {count}")
    print(f"positionWise Comparision: {posWise}")
    print(f"subStringWise Comparision: {subStrWise}")

if(__name__=="__main__"):
    main()




# print(f"word1: {tempWord1}")
# print(f"word2: {tempWord2}")
# print(f"similarities: {sims[-1]}\n")
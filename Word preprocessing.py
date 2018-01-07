import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer 

#####################################
########  DEFINED FUNCTIONS  ########
#####################################

# Stemming of all words in fileIn and writing the new text to fileOut
def stemming(fileIn, fileOut):
    lines = fileIn.readlines()
    for line in lines:
        words = line.split(" ")
        for word in words:
            fileOut.write(stemmer.stem(word) + " ")
            # if (stemmer.stem(word) != word):
            #    print('This is the original word: ' + word + ' and this the stemmed one: ' + stemmer.stem(word)) 
        fileOut.write("\n")             

# Lemmatization of the text according to WordNet's morphy function
def lemmatize(fileIn, fileOut):
    lines = fileIn.readlines()
    for line in lines:
        words = line.split(" ")
        for word in words:
            fileOut.write(wn.lemmatize(word) + " ")
            # if (wn.lemmatize(word) != word):
            #    print('This is the original word: ' + word + ' and this the lemmatized one: ' + wn.lemmatize(word)) 
        fileOut.write("\n")
            
#####################################
########      EXECUTION      ########
#####################################

# Initialization
wn = WordNetLemmatizer()
stemmer = EnglishStemmer()

# Stemming for the positive training set
with open("train-pos.txt", "r") as fileIn:
    with open("train-pos-stemmed.txt", "w") as fileOut:
        stemming(fileIn,fileOut)
        fileOut.close()
    fileIn.close()

# Stemming for the negative training set
with open("train-neg.txt", "r") as fileIn:
    with open("train-neg-stemmed.txt", "w") as fileOut:
        stemming(fileIn,fileOut)
        fileOut.close()
    fileIn.close()

# Lemmatizaion for the positive training set
with open("train-pos-stemmed.txt", "r") as fileIn:
    with open("train-pos-lemmatized.txt", "w") as fileOut:
        lemmatize(fileIn,fileOut)
        fileOut.close()
    fileIn.close()

# Lemmatizaion for the negative training set
with open("train-neg-stemmed.txt", "r") as fileIn:
    with open("train-neg-lemmatized.txt", "w") as fileOut:
        lemmatize(fileIn,fileOut)
        fileOut.close()
    fileIn.close()





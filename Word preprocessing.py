import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer 
from nltk.text import TextCollection
from nltk.corpus import wordnet as wn

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

# TODO: Include tokenization into the lemmatization process --> Better results!
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


# Returns tagges words
def tag(fileIn):
    return pos_tag(nltk.word_tokenize(fileIn.read()))

####################################
#######  EXTERNAL FUNCTIONS  #######
####################################

# Shall be used to enhance the lemmatization process
# Copied from stackoverflow: https://stackoverflow.com/questions/25534214/nltk-wordnet-lemmatizer-shouldnt-it-lemmatize-all-inflections-of-a-word
def is_noun(tag):
    return tag in ['NN', 'NNS', 'NNP', 'NNPS']

def is_verb(tag):
    return tag in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']

def is_adverb(tag):
    return tag in ['RB', 'RBR', 'RBS']

def is_adjective(tag):
    return tag in ['JJ', 'JJR', 'JJS']

def penn_to_wn(tag):
    if is_adjective(tag):
        return wn.ADJ
    elif is_noun(tag):
        return wn.NOUN
    elif is_adverb(tag):
        return wn.ADV
    elif is_verb(tag):
        return wn.VERB
    return None
            
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





import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer 
from nltk.text import TextCollection
from nltk.corpus import wordnet as wn
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

#####################################
########  DEFINED FUNCTIONS  ########
#####################################

# Stemming of all words in fileIn and writing the new text to fileOut
def stemming(fileIn, fileOut):
    lines = fileIn.readlines()
    for line in lines:
        words = line.split(" ")
        toWrite = ""
        for word in words:
            toWrite = toWrite + stemmer.stem(word) + " "
            # if (stemmer.stem(word) != word):
            #    print('This is the original word: ' + word + ' and this the stemmed one: ' + stemmer.stem(word))       
        # Remove trailing whitespace and write to file
        fileOut.write(toWrite.rstrip() + "\n")

# TODO: Include tokenization into the lemmatization process --> Better results!
# Lemmatization of the text according to WordNet's morphy function
def lemmatize(fileIn, fileOut):
    lines = fileIn.readlines()
    for line in lines:
        words = line.split(" ")
        toWrite = ""
        for word in words:
            toWrite = toWrite + wn.lemmatize(word) + " "
            # if (wn.lemmatize(word) != word):
            #    print('This is the original word: ' + word + ' and this the lemmatized one: ' + wn.lemmatize(word)) 
        # Remove trailing whitespace and write to file
        fileOut.write(toWrite.rstrip() + "\n")

# Remove all words that are part of the stopword list
def stopwordRemoval(fileIn, fileOut):
    lines = fileIn.readlines()
    for line in lines:
        words = line.split(" ")
        toWrite = ""
        for word in words:
            if (word not in sw):
                toWrite = toWrite + word + " "
            # else:
            #    print("Removed " + word)
        # Remove trailing whitespace and write to file
        fileOut.write(toWrite.rstrip() + "\n")


# Returns tagged words (Shall be used to enhance the lemmatization process)
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
sw = stopwords.words('english')

# TODO: Find optimal position for stopword removal
# Stopword removal for the positive training set
with open("test-pos.txt", "r") as fileIn:
    with open("train-pos-wostops.txt", "w") as fileOut:
        stopwordRemoval(fileIn,fileOut)
        fileOut.close()
    fileIn.close()

# Stopword removal for the negative training set
with open("test-neg.txt", "r") as fileIn:
    with open("train-neg-wostops.txt", "w") as fileOut:
        stopwordRemoval(fileIn,fileOut)
        fileOut.close()
    fileIn.close()

# Stemming for the positive training set
with open("train-pos-wostops.txt", "r") as fileIn:
    with open("train-pos-stemmed.txt", "w") as fileOut:
        stemming(fileIn,fileOut)
        fileOut.close()
    fileIn.close()

# Stemming for the negative training set
with open("train-neg-wostops.txt", "r") as fileIn:
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

# According to the example provided here: https://roshansanthosh.wordpress.com/2016/02/18/evaluating-term-and-document-similarity-using-latent-semantic-analysis/

# Perform tfidf vectorization --> term frequency * inverse document frequency
# TODO: Ueberpruefen ob das so richtig ist, dass man nur 2 Dimensionen hat?
with open ("train-pos-lemmatized.txt", "r") as fileIn:
    posData = fileIn.read()
    # print("This is the positive data: " + posData)
    fileIn.close()

with open ("train-neg-lemmatized.txt", "r") as fileIn:
    negData = fileIn.read()
    # print("This is the negative data: " + negData)
    fileIn.close()

data = posData.split("\n")
print("This is the data: ")
print(data)

# Funktioniert noch nicht so richtig
transformer = TfidfVectorizer()
tfidf = transformer.fit_transform(data)
print("This is the tfidf data: ")
print(tfidf)

# Perform SVD computation 
svd = TruncatedSVD(n_components = 2, algorithm="arpack")
print("This is the lsa data: ")
lsa = svd.fit_transform(tfidf.T)

print(lsa)

# TODO: Testen mit Datensplit







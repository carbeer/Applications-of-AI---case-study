import nltk
import csv
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import EnglishStemmer 
from nltk.text import TextCollection
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet as wnplain
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

#####################################
########  DEFINED FUNCTIONS  ########
#####################################

# Calls all the functions that are necessary to clean the data and returns list of reviews
def cleanData(fileIn):
	cleanedData = []
	with open(fileIn, "r") as file:
		lines = file.readlines()
		stemmed = stemming(lines)
		stopwordsRemoved = stopwordRemoval(stemmed)
		lemmatized = lemmatize(stopwordsRemoved)
		cleanedData = lemmatized
	file.close()
	return cleanedData

def cleanDataFromText(line):
	cleanedData = []
	lines = [line]
	stemmed = stemming(lines)
	stopwordsRemoved = stopwordRemoval(stemmed)
	lemmatized = lemmatizeWithTags(stopwordsRemoved)
	cleanedData = lemmatized
	return cleanedData

def cleanDataFromTextSplit(line):
	cleanedData = []
	lines = [line]
	stopwordsRemoved = stopwordRemoval(lines)
	lemmatized = lemmatizeWithTags(stopwordsRemoved)
	stemmed = stemming(lemmatized)
	lemmatized = lemmatizeWithTags(stemmed)
	cleanedData = lemmatized
	return cleanedData[0].split(" ")

# Stemming of all words in fileIn and writing the new text to fileOut
def stemming(fileIn, fileOut):
	lines = fileIn.readlines()
	for line in lines:
		words = line.split(" ")
		toWrite = ""
		for word in words:
			toWrite = toWrite + stemmer.stem(word) + " "
			# if (stemmer.stem(word) != word):
			#	print('This is the original word: ' + word + ' and this the stemmed one: ' + stemmer.stem(word))	   
		# Remove trailing whitespace and write to file
		fileOut.write(toWrite.rstrip() + "\n")

def stemming(lines):
	processedLines = []
	for line in lines:
		words = line.split(" ")
		toWrite = ""
		for word in words:
			toWrite = toWrite + stemmer.stem(word) + " "  
		# Remove trailing whitespace
		processedLines.append(toWrite.rstrip())
	return processedLines

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
			#	print("Removed " + word)
		# Remove trailing whitespace and write to file
		fileOut.write(toWrite.rstrip() + "\n")

def stopwordRemoval(lines):
	processedLines = []
	for line in lines:
		words = line.split(" ")
		toWrite = ""
		for word in words:
			if (word not in sw):
				toWrite = toWrite + word + " "
		# Remove trailing whitespace
		processedLines.append(toWrite.rstrip())
	return processedLines

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
			#	print('This is the original word: ' + word + ' and this the lemmatized one: ' + wn.lemmatize(word)) 
		# Remove trailing whitespace and write to file
		fileOut.write(toWrite.rstrip() + "\n")

def lemmatize(lines):
	processedLines = []
	for line in lines:
		words = line.split(" ")
		toWrite = ""
		for word in words:
			toWrite = toWrite + wn.lemmatize(word) + " "
		# Remove trailing whitespace
		processedLines.append(toWrite.rstrip())
	return processedLines

def lemmatizeWithTags(lines):
	processedLines = []
	count = 0
	for line in lines:
		words = tag(line)
		toWrite = ""
		for word in words:
			toWrite = toWrite + wn.lemmatize(word[0], penn_to_wn(word[1])) + " "
			# print('This is the original word: ' + word[0])
			# print('This is the category: ' + word[1])
			# print('And this is the lemmatized word: ' + wn.lemmatize(word[0], penn_to_wn(word[1])))
		# Remove trailing whitespace
		processedLines.append(toWrite.rstrip())
	return processedLines

# Creates a CSV file with lables
def lableDataFromFileToFile(fileInPos, fileInNeg, fileOut):
	linesPos = fileInPos.readlines()
	linesNeg = fileInNeg.readlines()
	csvwriter = csv.writer(fileOut)
	for line in linesPos:
		csvwriter.writerow([line, 'positive'])
	for line in linesNeg:
		csvwriter.writerow([line, 'negative'])

# Returns data with lables for further processing
def lableDataFromFile(fileInPos, fileInNeg):
	linesPos = fileInPos.readlines()
	linesNeg = fileInNeg.readlines()
	text = []
	lable = []
	for line in linesPos:
		text.append(line)
		lable.append('positive')
	for line in linesNeg:
		text.append(line)
		lable.append('negative')
	return text, lable

def lableData(linesPos, linesNeg):
	text = []
	lable = []
	for line in linesPos:
		text.append(line)
		lable.append('positive')
	for line in linesNeg:
		text.append(line)
		lable.append('negative')
	return text, lable

# Returns tagged words (Shall be used to enhance the lemmatization process)
def tag(line):
	return nltk.pos_tag(nltk.word_tokenize(line))


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
		return wnplain.ADJ
	elif is_noun(tag):
		return wnplain.NOUN
	elif is_adverb(tag):
		return wnplain.ADV
	elif is_verb(tag):
		return wnplain.VERB 
	return wnplain.NOUN
			
#####################################
########	  EXECUTION	  ########
#####################################

# Initialization
wn = WordNetLemmatizer()
stemmer = EnglishStemmer()
nb = MultinomialNB()
sw = stopwords.words('english')

# Adopted from https://medium.com/tensorist/classifying-yelp-reviews-using-nltk-and-scikit-learn-c58e71e962d9
text = []
lable = []

# Reads the reviews from the given files and lables them (according to the file as pos/ neg)
with open("Review files/train-pos.txt", "r") as filePos:
	with open ("Review files/train-neg.txt", "r") as fileNeg:
		text, lable = lableDataFromFile(filePos, fileNeg)
	fileNeg.close()
filePos.close()

# Bag of words method is used
# Creates a dictionary of words that occur in the raw documents
bow_transformer = CountVectorizer(analyzer=cleanDataFromTextSplit).fit(text)

# Creates document-term matrix
text = bow_transformer.transform(text)

# Splits our data into a training and a testing set. We used the common shares of 70% for training and 30% for testing. 
text_train, text_test, lable_train, lable_test = train_test_split(text, lable, test_size=0.3, random_state=101)

# Training
nb.fit(text_train, lable_train)

# Prediction
preds = nb.predict(text_test)
# print('These are the predictions:')
# print(preds)

# Evaluation: Uses the lable_test data to compare the values with the ones predicted by the data model
print('\n####################################')
print('##  See how our model performed!  ##')
print('####################################\n')
print('This is the confusion matrix:')
print(confusion_matrix(lable_test, preds))
print('\nThis is the classification report:')
print(classification_report(lable_test, preds))


# Predict values for the test data

testData = []
# Reads the reviews from the test file
with open("Review files/test.txt", "r") as testFile:
	testData = testFile.readlines()
testFile.close()

testDataTransformed = bow_transformer.transform(testData)
testPredictions = nb.predict(testDataTransformed)
with open("predictions.csv", "wb") as predictions:
	csvwriter = csv.writer(predictions)
	for x in range(0, (len(testData)-1)):
		csvwriter.writerow([testData[x], testPredictions[x]])
predictions.close()

print('You can find our predictions in the file "predictions.csv"')

####################################
#######  ITERATIVE CLEANING  #######
####################################



''' # TODO: Find optimal position for stopword removal
# Stopword removal for the positive training set
with open("Review files/train-pos.txt", "r") as fileIn:
	with open("Review files/train-pos-wostops.txt", "w") as fileOut:
		stopwordRemoval(fileIn,fileOut)
		fileOut.close()
	fileIn.close()

# Stopword removal for the negative training set
with open("Review files/train-neg.txt", "r") as fileIn:
	with open("Review files/train-neg-wostops.txt", "w") as fileOut:
		stopwordRemoval(fileIn,fileOut)
		fileOut.close()
	fileIn.close()

# Stemming for the positive training set
with open("Review files/train-pos-wostops.txt", "r") as fileIn:
	with open("Review files/train-pos-stemmed.txt", "w") as fileOut:
		stemming(fileIn,fileOut)
		fileOut.close()
	fileIn.close()

# Stemming for the negative training set
with open("Review files/train-neg-wostops.txt", "r") as fileIn:
	with open("Review files/train-neg-stemmed.txt", "w") as fileOut:
		stemming(fileIn,fileOut)
		fileOut.close()
	fileIn.close()

# Lemmatizaion for the positive training set
with open("Review files/train-pos-stemmed.txt", "r") as fileIn:
	with open("Review files/train-pos-lemmatized.txt", "w") as fileOut:
		lemmatize(fileIn,fileOut)
		fileOut.close()
	fileIn.close()

# Lemmatizaion for the negative training set
with open("Review files/train-neg-stemmed.txt", "r") as fileIn:
	with open("Review files/train-neg-lemmatized.txt", "w") as fileOut:
		lemmatize(fileIn,fileOut)
		fileOut.close()
	fileIn.close()

# Creates CSV with lables (used for Rapidminer)
with open("Review files/train-pos-lemmatized.txt", "r") as fileInPos:
	with open("Review files/train-neg-lemmatized.txt", "r") as fileInNeg:
		with open("Review files/train-labled.csv", "wb") as fileOut:
			lableDataFromFileToFile(fileInPos,fileInNeg,fileOut)
			fileOut.close()
		fileInNeg.close()
	fileInPos.close()

# Get data with lables
text = []
lables = []

with open("Review files/train-pos-lemmatized.txt", "r") as fileInPos:
	with open("Review files/train-neg-lemmatized.txt", "r") as fileInNeg:
			text, lables = lableDataFromFile(fileInPos,fileInNeg)
		fileInNeg.close()
	fileInPos.close()
'''
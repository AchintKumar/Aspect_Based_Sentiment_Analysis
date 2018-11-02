#Importing Files

import nltk
import ast
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
import os
import pandas as pd
import numpy as np
#nltk.download('sentiwordnet')

#Setting File Path

os.chdir('/media/achint/Practice/Functioniq/aspect')
print(os.getcwd())
print(os.listdir(os.getcwd()))

#Parsing Data

text = []

#For any other xml file containing review just change the file name
txt = open('restaurants_1.xml', 'r')
sent = []

#For taking out sentence id
for line in txt:
	#Sentence id tag will be called line by line
	if '<sentence id>' in line:
		line = line.strip()
		sent.append(line)

#Same here, just change the file name
txt = open('restaurants_1.xml', 'r')

#For taking out text in each sentence id
for line in txt:
	#text tag in each sentence id will be called
	if '<text>' in line:
		#Each line will contain tags etc. which needs to be stripped
		#So just replacing them with blank spaces
		line = line.replace('</text>', '')
		line = line.replace('<text>','')
		line = line.replace('pound', 'kilo')
		line = line.replace('#','')
		#Stripping each text line
		line=line.strip()
		#Appending in 'text' list
		text.append(line)

#Checking if parsed correctly
print(text)
print(text[-1])

#Writing reviews into a text file for future (not necessary)
reviewfile = open("restaurant", "w")
reviewfile.write("\n".join(map(lambda x: str(x), text)))
reviewfile.close()

#Now all the reviews are in one text file stripped of all tags
#Reading the review text file
inputFile = open('restaurant',"r").read()
print(inputFile)

#Removing Stopwords

#Stopwords included in the nltk dataset
StopWords = nltk.corpus.stopwords.words("english")
print(StopWords)

#Removing stopwords from reviews
result=(' '.join([word for word in inputFile.split() if word not in StopWords]))
print('Following are the Stop Words')
print(str(result))

#However, stopword removal leads to removal of meaning of sentences
#So, not using stopword removed reviews

#Tokenizing the reviews
tokenizedReviews={}

#NLTK Tokenizer
tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()

#Initializing a incrementer
Id=1;

#For every word in review file, that word will be tokenized
for sentence in tokenizer.tokenize(inputFile):      
	tokenizedReviews[Id]=sentence
	Id+=1

print(tokenizedReviews)

#POS-Tagging each tokenized value
outputPost={}

#Each tokenized value called one by one
for key,value in tokenizedReviews.items():
	outputPost[key]=nltk.pos_tag(nltk.word_tokenize(value))

#POS-tagged values
print(outputPost)


#Aspect Extraction

previous_word=''
previous_tag=''
current_word=''
aspects=[]
outputDict={}

#Extracting Aspects

for key,value in outputPost.items():
	for word,tag in value:
		#Extracting all nouns as nouns are power candidates for aspects
		if(tag=='NN' or tag=='NNP'):
			#If both words are nouns, then both words are a combined aspect
			if(previous_tag=='NN' or previous_tag=='NNP'):
				current_word= previous_word + ' ' + word
			else:
				aspects.append(previous_word.upper())
				current_word = word
		#Incrementing the word lable
		previous_word=current_word
		previous_tag=tag
print(aspects)

#Eliminating aspect which has 1 or less count
for aspect in aspects:
	if(aspects.count(aspect)>1):
		#Neglect all aspects coming once
		if(outputDict.keys()!=aspect):
			outputDict[aspect]=aspects.count(aspect)

outputAspect=sorted(outputDict.items(), key=lambda x: x[1],reverse = True)
print(outputDict)
#Aspect File with their counts in overall reviews
print(outputAspect)

#Polarity Detection

#orientation
#Made a function crosschecking each word's synonym in the dataset
def orientation(inputWord): 
	#wordnet is a lexical database
	#it has cognitive synonyms stored in synsets
	wordSynset=wordnet.synsets(inputWord)
	if(len(wordSynset) != 0):
		word=wordSynset[0].name()
		orientation=sentiwordnet.senti_synset(word)
		if(orientation.pos_score()>orientation.neg_score()):
			return True
		elif(orientation.pos_score()<orientation.neg_score()):
			return False

OpinionTuples={}
orientationCache={}
#Word set depicting negative behaviour (taken from internet)
negativeWordSet = {"don't","never", "nothing", "nowhere", "noone", "none", "not", "hasn't","hadn't","can't","couldn't","shouldn't","won't", "wouldn't","don't","doesn't","didn't","isn't","aren't","ain't"}

for aspect,no in outputAspect:
	#Tokenizing each aspect again
	aspectTokens= word_tokenize(aspect)
	count=0
	for key,value in outputPost.items():
		condition=True
		isNegativeSen=False
		for subWord in aspectTokens:
			if(subWord in str(value).upper()):
				condition = condition and True
			else:
				condition = condition and False
		if(condition):
			for negWord in negativeWordSet:
				#once senetence is negative no need to check this condition again and again
				if(not isNegativeSen):
					if negWord.upper() in str(value).upper():
						isNegativeSen=isNegativeSen or True
			#Setting up opinion tuple
			OpinionTuples.setdefault(aspect,[0,0])
			for word,tag in value:
				#Taking into account all adjectives and adverbs
				if(tag=='JJ' or tag=='JJR' or tag=='JJS'or tag== 'RB' or tag== 'RBR'or tag== 'RBS'):
					count+=1
					#Setting up synonyms of each word
					if(word not in orientationCache):
						orien=orientation(word)
						orientationCache[word]=orien
						orien=orientationCache[word]
					if(isNegativeSen and orien is not None):
						orien= not orien
					#Adding aspect polarity in positive and negative tuple values
					if orien:
						OpinionTuples[aspect][0]+=1
					else:
						OpinionTuples[aspect][1]+=1
					
	#Rounding up the overall positive and negative polarities of each aspect
	if(count>0):
		OpinionTuples[aspect][0]=round((OpinionTuples[aspect][0]/count)*100,2)
		OpinionTuples[aspect][1]=round((OpinionTuples[aspect][1]/count)*100,2)
		print(aspect,':\t\tPositive => ', OpinionTuples[aspect][0], '\tNegative => ',OpinionTuples[aspect][1])
		
print(OpinionTuples)

#Making a dataframe out of the results
output_file=pd.DataFrame(OpinionTuples)
output_file=output_file.transpose()
output_file.columns=['positive','negative']
output_file['polarity'] = np.where((output_file['positive'] >= output_file['negative']), 'positive', 'negative')
del output_file['positive']
del output_file['negative']
print(output_file)

#writing file
output_file.to_csv('output_file.csv')
import pandas as pd
import math
import spacy
nlp = spacy.load("en_core_web_md")
import numpy as np
from datetime import datetime
from collections import defaultdict
import pickle
from os.path import isfile 

import util
import autoslog

PATTERN_POOL_INIT_SIZE = 20
WORDS_PER_ROUND = 5
LOOPS = 15
MAX_TEXT_SIZE = 2000

def AvgLog(patterns, category, lexicon):
    # Creates a list of lists containing only category members
    AvgLogMembersExtracted = 0
    for pattern in patterns:
        categoryWords = [word for word in pattern[1] if word in lexicon]
        AvgLogMembersExtracted += math.log(len(categoryWords)+1, 2)
    p = len(patterns)

    score = AvgLogMembersExtracted/p
    return round(score, 3)

def RlogF(words, category, lexicon):
    n = len(words)
    f = len([word for word in words if word in lexicon])

    if n == 0 or f == 0:
        score = 0
    else:
        score = (f/n)*math.log(f,2)

    return round(score, 3)

def extract(doc, pattern):
    extractions = set()
    for word in doc:
        if (word.head.text, word.dep_) == pattern:
            # Make sure to add compound words together like boy choy instead of choy
            compounds = [child.text for child in word.lefts if child.dep_ == 'compound']
            compoundWord = ' '.join(compounds + [word.text])
            extractions.add(compoundWord)
    return extractions

def basilisk(category, output, path, picklePath, docsPath, development=False):
    print("Start Time:", datetime.now().strftime("%H:%M:%S"))
    seeds = util.fetchLines(f'seeds/{category}.seed')
    lexicon = set(seeds)
    print('Reading CSV')
    texts = pd.read_csv(path, squeeze=True)
    texts.dropna(inplace=True)
    # Split sampledTexts into series of size <= MAX_TEXT_SIZE and finalize them into a list of docs
    print('Performing Dependency Parsing')
    splitTexts = np.split(texts, [MAX_TEXT_SIZE*(i+1) for i in range(len(texts)//MAX_TEXT_SIZE)])
    joinedTexts = ['. '.join(text) for text in splitTexts]

    # Check to see if NLP has been done already
    if isfile(docsPath):
        with open(docsPath, 'rb') as file: 
            docs = pickle.load(file) 
    else:
        docs = [nlp(text) for text in joinedTexts]
        with open(docsPath, 'wb') as file:
            pickle.dump(docs, file) 
    
    # Check for already extraced patterns
    if isfile(picklePath):
        with open(picklePath, 'rb') as file: 
            extractedPatternsDict = pickle.load(file) 
    else:
        extractedPatternsDict = defaultdict(set)

    if development:
        print('Pre Processing Done')

    for iteration in range(LOOPS):
        
        if development:
            print(f'Starting Loop {iteration+1} for category {category}'), 

        allPatterns = set()
        for doc in docs:
            allPatterns.update(autoslog.extractPatterns(doc, lexicon))

        if development:
            print('1: Patterns Extracted')

        extractedPatterns = []
        for pattern in allPatterns:
            if pattern in extractedPatternsDict:
                extractedPatterns.append((pattern, extractedPatternsDict[pattern]))
            else:
                extractedPattern = set()
                for doc in docs:
                    extractedPattern.update(extract(doc, pattern))
                extractedPatternsDict[pattern] = extractedPattern
                extractedPatterns.append((pattern, extractedPattern))

        if development:
            print('2: Extraction Done on All Potential Patterns')

        scoredPatterns = []
        for patternSet in extractedPatterns:
            score = RlogF(patternSet[1], category, lexicon)
            scoredPatterns.append((patternSet[0], patternSet[1], score))
        scoredPatterns.sort(key=lambda x: x[2], reverse=True)

        if development:
            print('3: Patterns Scored and Trimmed')

        chosenPatterns = scoredPatterns[:PATTERN_POOL_INIT_SIZE + iteration]
        candidateWords = set().union(*[chosenPattern[1] for chosenPattern in chosenPatterns])
        candidateWords = candidateWords.difference(lexicon)

        scoredWords = []
        for word in candidateWords:
            patternsThatExtracted = [pattern for pattern in extractedPatterns if word in pattern[1]]
            score = AvgLog(patternsThatExtracted, category, lexicon)
            scoredWords.append((word, score))
        scoredWords.sort(key=lambda x: x[1], reverse=True)
        chosenWords = scoredWords[:WORDS_PER_ROUND]
        lexicon = lexicon.union({word[0] for word in chosenWords})

        if development:
            print('4: Words Scored and Trimmed')
            print(chosenWords)
            print()

    print('Extracted Words...')
    generatedWords = lexicon.difference(set(seeds))
    print(lexicon.difference(set(seeds)))
    print()

    # Write the output of generated words
    file = open(f'{output}{category}.txt','w')
    for word in generatedWords:
        file.write(f'{word}\n')
    file.close()

    # Save pattern extractions for next time
    with open(picklePath, 'wb') as file: 
        pickle.dump(extractedPatternsDict, file) 
    print("End Time:", datetime.now().strftime("%H:%M:%S"))
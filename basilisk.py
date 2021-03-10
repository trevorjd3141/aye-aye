import util
import autoslog
import pandas as pd
import math
import spacy
nlp = spacy.load("en_core_web_md")
import numpy as np
from datetime import datetime
from collections import defaultdict

PATTERN_POOL_INIT_SIZE = 20
WORDS_PER_ROUND = 5
LOOPS = 10
SAMPLE_SIZE = 10000
MAX_TEXT_SIZE = 1500

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
            extractions.add(word.text)
    return extractions

def basilisk(category, output, path, development=False):
    print("Start Time:", datetime.now().strftime("%H:%M:%S"))

    seeds = util.fetchLines(f'seeds/{category}.seed')
    lexicon = set(seeds)

    texts = pd.read_csv(path, squeeze=True)
    sampledTexts = texts.sample(n=SAMPLE_SIZE)
    # Split sampledTexts into series of size <= MAX_TEXT_SIZE and finalize them into a list of docs
    splitSampledTexts = np.split(sampledTexts, [MAX_TEXT_SIZE*(i+1) for i in range(SAMPLE_SIZE//MAX_TEXT_SIZE)])
    joinedSampledTexts = ['. '.join(text) for text in splitSampledTexts]
    docs = [nlp(text) for text in joinedSampledTexts]

    extractedPatternsDict = defaultdict(set)

    if development:
        print('Loading Done')

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
    file = open(f'{output}{category}.txt','w')
    for word in generatedWords:
        file.write(f'{word}\n')
    file.close()
    print("End Time:", datetime.now().strftime("%H:%M:%S"))
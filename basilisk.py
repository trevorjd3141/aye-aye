import util
import autoslog
import pandas as pd
import math
import spacy
nlp = spacy.load("en_core_web_sm")

SIZE = 1

def RlogF(words, category):
    categories = [util.categorize(word) for word in words]
    n = len(categories)
    f = categories.count(category)

    if n == 0:
        return 0

    score = (f/n)*math.log(f,2)
    return score

def extract(text, pattern):
    size = len(pattern)
    patternWords = [token[0] for token in pattern]

    # Exit early if word can't be found
    # in the text
    for word in patternWords:
        if word == '_WORD_':
            continue

        if word not in text:
            return set()

    doc = nlp(text)
    extractions = set()
    for sent in doc.sents:
        sentWords = list([word.text for word in sent])

        # Exit early if all pattern words aren't in the sentence
        if not all([word in sentWords for word in patternWords if word != '_WORD_']):
            continue

        for i in range(len(sent) - size + 1):
            window = sent[i: i + size]
            flag = False
            extraction = None
            for j in range(len(pattern)):
                windowToken = window[j]
                patternText = pattern[j][0]
                patternDep = pattern[j][1]
                if (windowToken.text == patternText) and patternText == '_WORD_':
                    extraction = windowToken.text
                elif windowToken.dep_ != patternDep or windowToken.text != patternText:
                    flag = True
                    break
            if not flag and extraction is not None:
                extractions.add(extraction)
    return extractions

def basilisk(category, development=False):
    seeds = util.fetchLines(f'seeds/noun.{category}.seed')
    lexicon = set(seeds)
    textDF = pd.read_csv('data/compiledText.csv')
    textDF = textDF.sample(n=500)
    textDF.dropna(inplace=True)
    iteration = 0

    if development:
        print('1: Loading Done')

    allPatterns = []
    for text in textDF['Text']:
        patterns = autoslog.extractPatterns(text, lexicon, SIZE)
        allPatterns.append(patterns)
    allPatterns = set().union(*allPatterns)

    if development:
        print('2: Patterns Extracted')

    patternSets = []
    for pattern in allPatterns:
        allExtractedWords = []
        for text in textDF['Text']:
            newlyExtractedWords = extract(text, pattern)
        allExtractedWords = set().union(*allExtractedWords)
        patternSets.append((pattern, allExtractedWords))

    if development:
        print('3: Extraction Done on All Potential Patterns')

    for setter in patternSets:
        print(setter[1])
    
    scoredPatterns = []
    for patternSet in patternSets:
        scoredPatterns.append((patternSet[0], RlogF(patternSet[1], category)))

    if development:
        print('4: Patterns Scored')
    
    #print(sorted(scoredPatterns, reverse=True))


basilisk('animal', True)
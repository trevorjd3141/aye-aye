import util

# Takes a maximized pattern template and returns a set
# of all subsets of it
def expandPatterns(patternTemplate):
    patterns = set()
    leftUnits, parentUnits, rightUnits = patternTemplate
    for i in range(len(leftUnits)+1):
        for j in range(len(parentUnits)+1):
            for k in range(len(rightUnits)+1):
                patterns.add((leftUnits[:i], parentUnits[:j], rightUnits[:k]))
    return patterns

# Patterns are in the form
# (unit) = (word, dep)
# (tuple) = (left units, parent unit, parent tuple, right units)
def extractPatterns(doc, lexicon, maxUp=2, maxLeft=2, maxRight=1):
    patterns = set()
    for sent in doc.sents:
        words = [word.text for word in sent]
        matches = [(words.index(word.text), word) for word in sent if word.dep_ != 'ROOT' and word.text in lexicon]
        for index, token in matches:
            leftUnits = tuple([(leftToken.text, leftToken.dep_) for leftToken in list(token.lefts)[-maxLeft:]])
            ancestorList = [token] + list(token.ancestors)[:maxUp]
            parentUnits = tuple([(parentToken.head.text, parentToken.dep_) for parentToken in ancestorList if parentToken.dep_ != 'ROOT'])
            rightUnits = tuple([(rightToken.text, rightToken.dep_) for rightToken in list(token.rights)[:maxRight]])
            patternTemplate = (leftUnits, parentUnits, rightUnits)
            patterns.update(expandPatterns(patternTemplate))
    return patterns
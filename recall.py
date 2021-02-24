import util
from os.path import join

def compileCategoryWords(path, category):
    categoryWords = set()
    for directory in util.allDirectories(path):
        for file in util.allFiles(join(path, directory)):
            filePath = join(path, directory, file)
            text = ' '.join(util.fetchLines(filePath))
            words = text.split()
            for word in words:
                if util.categorize(word) == category:
                    categoryWords.add(word)
    output = open(f'recall/{category}.txt', 'w')
    for word in categoryWords:
        output.write(f'{word}\n') 
    output.close() 
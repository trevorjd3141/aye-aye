import grader
import preprocessing

NEWS_PATH = 'data\\20news\\'

def main():
    preprocessing.extract(NEWS_PATH)
    grader.read('output/test.txt', 'noun.person')

if __name__ == "__main__":
    main()
import util

def extract(category):
    seeds = util.fetchLines(f'seeds/{category}.seed')
    lexicon = set(seeds)
    iteration = 0
from itertools import product

# Takes a maximized pattern template and returns a set
# of all subsets of it
def expand_patterns(pattern_template, min_pattern_complexity, max_pattern_complexity):
    patterns = set()
    target_dep, left_units, parent_units, right_units, left_sibling, right_sibling = pattern_template

    # Get all possible combinations of patterns from the pattern template
    # itertools creates a cartesian product between the factors of the pattern
    for i, j, k, l, m in product(range(len(left_units)+1), range(len(parent_units)+1), range(len(right_units)+1), range(len(left_sibling)+1), range(len(right_sibling)+1)):
        if i+j+k+l+m >= min_pattern_complexity and i+j+k+l+m <= max_pattern_complexity:
            patterns.add((target_dep, left_units[:i], parent_units[:j], right_units[:k], left_sibling[:l], right_sibling[:m]))
    return patterns

# Patterns are in the form
# (unit) = (word, dep)
# (tuple) = (target unit, left units, parent units, right_units, left sibling unit, right sibling unit)
def extract_patterns(doc, lexicon, max_left=2, max_up=2, max_right=1, left_sibling=True, right_sibling=True, min_pattern_complexity=2, max_pattern_complexity=3):
    patterns = set()
    for token in doc:
        if token.lower_ not in lexicon:
            continue

        left_tokens = [token for token in token.lefts if token.dep_ != 'det' and token.dep_ != 'punct']
        left_units = tuple([(left_token.lower_, left_token.dep_) for left_token in left_tokens[-max_left:]])

        parent_units = tuple([(parent_token.lower_, parent_token.dep_) for parent_token in list(token.ancestors)[:max_up] if parent_token.dep_ != 'ROOT'])

        right_token = [token for token in token.rights if token.dep_ != 'det' and token.dep_ != 'punct']
        right_units = tuple([(right_token.lower_, right_token.dep_) for right_token in right_token[:max_right]])

        if token.dep_ != 'ROOT':
            siblings = list([(sibling.lower_, sibling.dep_) for sibling in token.head.children])
            index = [sibling[0] for sibling in siblings].index(token.lower_)
            left_sibling = tuple([siblings[index-1]]) if index > 0 and left_sibling else ()
            right_sibling = tuple([siblings[index+1]]) if index+1 < len(siblings) and right_sibling else ()
        else:
            left_sibling = ()
            right_sibling = ()

        pattern_template = (token.dep_, left_units, parent_units, right_units, left_sibling, right_sibling)
        patterns.update(expand_patterns(pattern_template, min_pattern_complexity, max_pattern_complexity))
    return patterns
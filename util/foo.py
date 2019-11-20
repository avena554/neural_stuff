from sampling.gibs_dirichlet_hmm import HMMState

sentences = [['a', 'b'], ['c', 'd', 'e']]
tag_set = {'1', '2'}

s = HMMState(sentences, tag_set)
print(s)
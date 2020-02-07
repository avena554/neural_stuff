from sampling.gibbs_hmm import HMMCounts, naive_sampler, hierarchical_sampler, HMMAverage, HMMMAP, extract_classes, build_prepare_fn
from sampling.gibbs_sampling import collect_samples
from util.conll_io import CONLLWriter

sentences = [['a', 'c'], ['a', 'd'], ['b', 'c'], ['b', 'd']]
tag_set = ['1', '2']
s_lengths = [len(s) for s in sentences]

print('***\nstarting little testing\n***\n')
sampler, _ = hierarchical_sampler(sentences, tag_set, n=4, alphas=[1., 1., 1., 1.])
#sampler, _ = naive_sampler(sentences, tag_set, n=2)

#(w_to_i, i_to_w) = index(['a', 'b', 'c', 'd'])
#(t_to_i, i_to_t) = index(s.extended_set)

#collector = HMMAverage(4, 4, tag_to_i=t_to_i, word_to_i=w_to_i)

pretty_maker = build_prepare_fn(sentences, s_lengths)
map_collector = HMMMAP(extract_classes, pretty_maker)


summary = collect_samples(sampler, map_collector, 20000, 100)
sample = next(sampler)
print(sample)
#print(sample.storage._t_count)

print("and the result is: \n%s\n" % summary)
#tr = np.transpose(delta)
#print([sum(tr[i]) for i in range(4)])

#for k in range(1000):
#    sample = next(sampler)
#    print(sample._t_total)
#    print("%s\n" % str(sample))

print(map_collector.counter.values())

wr = CONLLWriter(summary)
print(wr.as_string())
wr.write('../tests/bla.text')





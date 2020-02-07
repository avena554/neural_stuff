import argparse
from nltk.corpus.reader.conll import ConllCorpusReader

from sampling.gibbs_hmm import naive_sampler, hierarchical_sampler, HMMMAP, extract_classes, build_prepare_fn
from sampling.gibbs_sampling import collect_samples
from util.conll_io import HackedConllCorpusReader
from util.conll_io import CONLLWriter, get_tagged_sents, s_fields
from pathlib import Path


parser = argparse.ArgumentParser(description=
                                 """Does the thing, you know.
                                 """)

parser.add_argument("-i", type=str, dest="input")
parser.add_argument("-o", type=str, dest="output")
parser.add_argument("-we", type=int, dest="warmup_epochs", default=100)
parser.add_argument("-ns", type=int, dest="n_samples", default=100)
parser.add_argument("-v", action="store_true", dest="verbose", default=False)
parser.add_argument("-s", type=str, dest="init_state")


if __name__ == "__main__":
    args = parser.parse_args()
    summary = None

    if args.input:
        print("Reading corpus")
        t_s = get_tagged_sents(args.input, fields=s_fields)
        sents = t_s.get_sents()

        tagset = set()
        ##make separate input for tagset, make unlabelled input for sampling
        for s in t_s.get_tags():
            for t in s:
                tagset.add(t)
        print(tagset)

        if args.init_state:
            print("retrieving initial state")
            tagged_sents = get_tagged_sents(args.init_state, fields=s_fields)
            init_tags = tagged_sents.get_tags()

        sampler, initial_state = hierarchical_sampler(sents, tagset,
                                                      verbose=args.verbose, initial_tags=init_tags, n=3
                                                      )
        print(initial_state.n_tokens)

        pretty_maker = build_prepare_fn(initial_state.get_sents(), initial_state.s_lengths)
        collector = HMMMAP(extract_classes, pretty_maker)

        warmup_samples = args.warmup_epochs*initial_state.n_tokens
        samples = args.n_samples

        def trace(_, i):
            if i % 100 == 0:
                print("Sample #%d" % i)

        summary = collect_samples(sampler, collector, warmup_samples, samples, pre_action=trace)
        print("argmax occurred %d times" % collector._max)

        w = CONLLWriter(summary)

        if args.output and summary is not None:
            w.write(args.output)

    else:
        parser.print_usage()

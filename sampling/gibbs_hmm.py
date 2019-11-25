from sampling.conditional_proba import NormalizingConditional
from util import default_rng
from collections import defaultdict


def uniform_init(tag_set, rng, n_sents, s_lengths, n_tokens):
    chosen_tags = rng.choice(tag_set, n_tokens)
    initial_tags = [None for _ in range(n_sents)]
    token_count = 0
    for i in range(n_sents):
        initial_tags[i] = [chosen_tags[token_count + j] for j in range(s_lengths[i])]
        token_count += s_lengths[i]
    return initial_tags

# TODO: Harmonize use of indices and various access functions
class HMMState:

    def __init__(self, sentences, tag_set, initial_tags=None, init_method=uniform_init, rng=default_rng):
        self.sentences = sentences
        self.s_lengths = [len(s) for s in self.sentences]
        self.n_tokens = sum(self.s_lengths)
        self.n_sents = len(self.s_lengths)

        tags_list = ['t_' + tag for tag in tag_set]
        if initial_tags is None:
            initial_tags = init_method(tags_list, rng, self.n_sents, self.s_lengths, self.n_tokens)

        self.tags = initial_tags

        self.eos = '<eos>'
        self.sos = '<sos>'

        tags_list.extend([self.eos, self.sos])
        self.tag_set = frozenset(tags_list)

        self._t_count = defaultdict(lambda:defaultdict(int))
        self._t_total = defaultdict(int)
        self._e_count = defaultdict(lambda:defaultdict(int))
        self._e_total = defaultdict(int)

        self.refresh()

    def refresh(self):
        self._t_count.clear()
        self._e_count.clear()
        self._t_total.clear()
        self._e_total.clear()

        for i in range(self.n_sents):
            for (src_tag, dest_tag) in self.transitions(i):
                self._t_count[src_tag][dest_tag] += 1
                self._t_total[src_tag] += 1
            for (tag, word) in self.emissions(i):
                self._e_count[tag][word] += 1
                self._e_total[tag] += 1

    def update(self, index, new_tag):
        former_tag = self.get_tag(index)
        preceding_tag = self.get_preceding_tag(index)
        following_tag = self.get_following_tag(index)
        word = self.get_word(index)

        self._t_count[preceding_tag][former_tag] -= 1
        self._t_count[former_tag][following_tag] -= 1
        self._e_count[former_tag][word] -= 1
        self._t_total[former_tag] -= 1
        self._e_total[former_tag] -= 1

        self._t_count[preceding_tag][new_tag] += 1
        self._t_count[new_tag][following_tag] += 1
        self._e_count[new_tag][word] += 1
        self._t_total[new_tag] += 1
        self._e_total[new_tag] += 1

    def transitions(self, i):
        return [(self.get_preceding_tag((i, j)), self.get_tag((i, j))) for j in range(self.s_lengths[i]+1)]

    def emissions(self, i):
        return [(self.get_tag((i, j)), self.get_word((i, j))) for j in range(self.s_lengths[i])]

    def get_word(self, index):
        (i,j) = index
        return self.sentences[i][j]

    def get_tag(self, index):
        if self.is_eos(index):
            return self.eos
        if self.is_sos(index):
            return self.sos

    def admissible(self, index):
        if self.is_eos(index):
            return frozenset(['<eos>'])
        else:
            if self.is_sos(index):
                return frozenset(['<sos>'])
            else:
                return self.tag_set

    def is_eos(self, index):
        return index[1] >= self.s_lengths[index[0]]

    def is_sos(self, index):
        return index[1] < 0

    def t_count(self, src_tag, dest_tag):
        return self._t_count[src_tag][dest_tag]

    def t_total(self, src_tag):
        return self._t_total[src_tag]

    def e_count(self, tag, word):
        return self._e_count[tag][word]

    def e_total(self, tag):
        return self._e_total[tag]

    def get_preceding_tag(self, index):
        (i, j) = index
        return self.get_tag((i, j-1))

    def get_following_tag(self, index):
        (i, j) = index
        return self.get_tag((i, j+1))

    def __str__(self):
        return '\n'.join(
            [' '.join([str((self.sentences[i][j], self.tags[i][j])) for j in range(self.s_lengths[i])]
                      )
             for i in range(self.n_sents)]
        )

    def __repr__(self):
        return str(self)


class HMMConditional(NormalizingConditional):

    def get_potentials(self, state:HMMState, index):
        tag_set = state.admissibleFor(index)
        current_tag = state.get_tag(index)
        p_tag = state.get_preceding_tag(index)
        n_tag = state.get_following_tag(index)
        word = state.get_word(index)
        for c_tag in tag_set:
            amend = (c_tag == current_tag)
            p_to_c = state.t_count(p_tag, c_tag) - amend
            c_to_n = state.t_count(c_tag, n_tag) - amend
            p_total = state.t_total(p_tag)
            c_total = state.t_total(c_tag)
            e = state.e_count(c_tag, word)
            e_total = state.e_total(c_tag)
            potential = (p_to_c / p_total) * (c_to_n / c_total) * (e / e_total)
            yield(potential)


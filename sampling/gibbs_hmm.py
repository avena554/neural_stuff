from sampling.gibbs_sampling import NormalizingConditional, gibbs_sampler, GibbsState, Statistic, Paperclip,\
    Transform, StatisticSetter, ProvidingState
from util import default_rng
from util.log_space import log, log_add
from collections import defaultdict
from structures import TaggedSents
from typing import TypeVar, Generic, Iterable, Tuple, List
from enum import IntEnum
import numpy as np


def uniform_init(tag_set, rng, n_sents, s_lengths, n_tokens):
    chosen_tags = rng.choice(tag_set, n_tokens)
    initial_tags = [None for _ in range(n_sents)]
    token_count = 0
    for i in range(n_sents):
        initial_tags[i] = [chosen_tags[token_count + j] for j in range(s_lengths[i])]
        token_count += s_lengths[i]
    return initial_tags


Index = Tuple[int, int]


class TagAssignment(TaggedSents, ProvidingState[Index, 'TagAssigment']):

    class Wrapper:

        def __init__(self, wrapped, mapping=None):
            self.wrapped = wrapped
            self.mapping = mapping

        def __getitem__(self, index):
            item = self.wrapped[index]
            if self.mapping:
                item = self.mapping(item)

            return item

        def __len__(self):
            return len(self.wrapped)

    class WrapperWrapper(Wrapper):

        def __init__(self, wrapped, inner_mapping=None):
            super(TagAssignment.WrapperWrapper, self).__init__(wrapped,
                                                               mapping=lambda x: TagAssignment.Wrapper(
                                                                    x, mapping=inner_mapping)
                                                               )

    class TagsWrapper(WrapperWrapper):

        def __init__(self, outer_instance):
            super(TagAssignment.TagsWrapper, self).__init__(outer_instance.tags,
                                                            inner_mapping=lambda x: outer_instance.tag_externer[x])

    class SentsWrapper(WrapperWrapper):

        def __init__(self, outer_instance):
            super(TagAssignment.SentsWrapper, self).__init__(outer_instance.sents, inner_mapping=None)

    def __init__(self, sentences, tagset, initial_tags=None, init_method=uniform_init, rng=default_rng):
        super(TagAssignment, self).__init__()
        self.sents = sentences
        self.s_lengths = [len(s) for s in self.sents]
        self.n_tokens = sum(self.s_lengths)
        self.n_sents = len(self.s_lengths)
        self.indices = tuple([(i, j) for i in range(self.n_sents) for j in range(self.s_lengths[i])])

        self.tag_interner = {t: 't_' + str(t) for t in tagset}
        self.tag_externer = {'t_' + str(t): t for t in tagset}

        self.tagset = list(self.tag_interner.values())
        self.eos = '<eos>'
        self.sos = '<sos>'
        self.una = '<?>'
        self.extended_set = self.tagset + [self.eos, self.sos, self.una]

        if initial_tags is None:
            initial_tags = init_method(tagset, rng, self.n_sents, self.s_lengths, self.n_tokens)

        self.tags = [[self.tag_interner[tag] for tag in tags] for tags in initial_tags]

        self.external_tags = TagAssignment.TagsWrapper(self)
        self.external_sents = TagAssignment.SentsWrapper(self)

    def get_word(self, index: Index):
        (i, j) = index
        return self.sents[i][j]

    def get_tag(self, index: Index):
        if self.is_eos(index):
            return self.eos
        if self.is_sos(index):
            return self.sos
        return self.tags[index[0]][index[1]]

    def get_value(self, index: Index):
        return self.get_tag(index)

    def admissible(self, index: Index):
        if self.is_eos(index):
            return ['<eos>'],
        else:
            if self.is_sos(index):
                return ['<sos>'],
            else:
                return self.tagset

    def is_eos(self, index: Index):
        return index[1] >= self.s_lengths[index[0]]

    # noinspection PyMethodMayBeStatic
    def is_sos(self, index: Index):
        return index[1] < 0

    def get_tags(self):
        return self.external_tags

    def get_sents(self):
        return self.external_sents

    def update(self, index: Index, tag):
        self.tags[index[0]][index[1]] = tag

    def __str__(self):
        return '\n'.join(
            [' '.join([str((self.sents[i][j], self.tags[i][j])) for j in range(self.s_lengths[i])]
                      )
             for i in range(self.n_sents)]
        )

    def __repr__(self):
        return str(self)

    def get(self) -> 'TagAssignment':
        return self


# TODO: move this where it belongs
class Event:

    def set_value(self, v):
        pass


E = TypeVar('E', bound=Event)


class Events(Generic[E], Event):

    def __init__(self, *events: E):
        self.events : List[E] = []
        for e in events:
            self.add(e)

    def set_value(self, v):
        for e in self.events:
            e.set_value(v)

    def add(self, e: E):
        self.events.append(e)

    def __getitem__(self, item) -> E:
        return self.events[item]

    def __str__(self):
        return str([str(e) for e in self.events])


# TODO: potentially different types stored in single array is ugly
class VariableNgram(Event):

    def __init__(self, i, ngram, variable=True):
        self._variable_index = i
        self.ngram = list(ngram)
        self._variable = variable

    def __getitem__(self, index):
        if index == 0:
            return tuple(self.ngram[:-1])
        if index == 1:
            return self.ngram[-1]
        else:
            raise IndexError()

    def set_value(self, v):
        if self._variable:
            # print('making the change:' + str(v))
            self.ngram[self._variable_index] = v

    def __str__(self):
        return str((self.ngram[:-1], self.ngram[-1]))

    def __repr__(self):
        return repr((self.ngram[:-1], self.ngram[-1]))


def transition_to(assignment: TagAssignment, index: Index, variable_index: Index, n: int, variable=True) \
        -> VariableNgram:
    (i, j) = index
    relative_v_index = None
    if variable:
        (vi, vj) = variable_index
        relative_v_index = n - 1 - (j - vj)
    ngram = [assignment.get_tag((i, j - n + k + 1)) for k in range(n - 1)]
    ngram.append(assignment.get_tag(index))
    return VariableNgram(relative_v_index, ngram, variable=variable)


def emission_at(assignment: TagAssignment, index: Index, variable_index: Index, n: int, variable=True) -> VariableNgram:
    (i, j) = index
    relative_v_index = None
    if variable:
        (vi, vj) = variable_index
        relative_v_index = n - 2 - (j - vj)
    ngram = [assignment.get_tag((i, j - n + 2 + k)) for k in range(n - 1)]
    ngram.append(assignment.get_word((i, j)))
    return VariableNgram(relative_v_index, ngram, variable=variable)


def sent_transitions(assignment: TagAssignment, i: int, n: int):
    return [
        transition_to(assignment, (i, j), None, n, variable=False)
        for j in range(assignment.s_lengths[i] + n - 1)
    ]


def sent_emissions(assignment: TagAssignment, i: int, n: int):
    return [emission_at(assignment, (i, j), None, n, variable=False) for j in range(assignment.s_lengths[i])]


D = TypeVar('D', bound=DecoratedAssignment)


class HmmTransitionClip(Generic[D], Paperclip[D, Index, Events[VariableNgram]]):

    def __init__(self, n=2):
        super(HmmTransitionClip, self).__init__()
        self.n = n

    def extract_event(self, decorated: D, index: Index) -> Events[VariableNgram]:
        assignment = decorated.get()
        (i, j) = index
        return Events[VariableNgram](
            *[transition_to(assignment, (i, k), index, self.n, variable=True) for k in range(j, j + self.n)]
        )


class HmmEmissionClip(Generic[D], Paperclip[D, Index, Events[VariableNgram]]):

    def __init__(self, n=2):
        super(HmmEmissionClip, self).__init__()
        self.n = n

    def extract_event(self, decorated: D, index: Index) -> Events[VariableNgram]:
        assignment = decorated.get()
        (i, j) = index
        return Events[VariableNgram](
            *[
                emission_at(assignment, (i, k), index, self.n, variable=True)
                for k in range(j, min(j + self.n - 1, assignment.s_lengths[i]))
            ]
        )


# TODO:
class NgramCount(Statistic[Iterable]):
    class Direction(IntEnum):
        INCREMENT = 1
        DECREMENT = -1

    def __init__(self):
        # FIXME: always default to 1 is weird, parametrize this at least
        self._count = defaultdict(lambda: defaultdict(lambda: 1))
        self._total = defaultdict(lambda: 1)

    def clear(self):
        self._count.clear()
        self._total.clear()

    def _update_one(self, g, direction: Direction = Direction.INCREMENT):
        (c, t) = g
        self._count[c] += direction
        self._total[c] += direction

    def _update(self, events: Iterable, direction: Direction = Direction.INCREMENT):
        for g in events:
            self._update_one(g, direction)

    def forget(self, events: Iterable):
        self._update(events, NgramCount.Direction.DECREMENT)

    def collect(self, events: Iterable):
        self._update(events, NgramCount.Direction.INCREMENT)

"""
class TruncatedNgram:

    def __init__(self, offset, ngram):
        self.full = ngram
        self.offset = offset

    def __getitem__(self, item):
        if item == 0:
            return tuple(self.full[0][self.offset:])
        else:
            return self.full[item]

    def __str__(self):
        return str(self.full)
"""


class TransitionCountSetter(StatisticSetter[TagAssignment, Iterable]):

    def __init__(self, n : int):
        self.n : int = n

    def shape_input(self, assignment: TagAssignment) -> Iterable:
        transitions = []
        for i in range(assignment.n_sents):
            transitions.extend(sent_transitions(assignment, i, self.n))

        return transitions


class EmissionCountSetter(StatisticSetter[TagAssignment, Iterable]):
    def __init__(self, n: int):
        self.n: int = n

    def shape_input(self, assignment: TagAssignment) -> Iterable:
        emissions = []
        for i in range(assignment.n_sents):
            emissions.extend(sent_emissions(assignment, i, self.n))

        return emissions


class Truncation(Transform[Events[VariableNgram, Iterable]]):

    def __init__(self, offset):
        self.offset = offset

    def apply(self, events: Events[VariableNgram]) -> Iterable:
        return ((e[0][self.offset:], e[1]) for e in events)


def make_fallback_state(n_t: int, n_e: int, assignment: TagAssignment) -> GibbsState[Index]:
    t_clip = HmmTransitionClip[TagAssignment](n_t)
    clipped_t: DecoratedAssignment = t_clip.clip(assignment)
    t_stats = [NgramCount() for _ in range(n_t)]
    t_transforms = [Truncation(k) for k in range(n_t)]
    t_setters = [TransitionCountSetter(k) for k in range(n_t)]
    for k in range(n_t):
        t_setters[k].set(t_stats[k])
    t_clip.add_stat(t_stats[k], t_transforms[k])

    e_clip = HmmEmissionClip(n_e)
    clipped_e: GibbsState[Index] = e_clip.clip(t_clip)
    e_stats = [NgramCount() for _ in range(n_e)]
    e_transforms = [Truncation(k) for k in range(n_e)]
    e_setters = [EmissionCountSetter(k) for k in range(n_e)]
    for k in range(n_e):
        e_setters[k].set(e_stats[k])
    e_clip.add_stat(e_stats[k], e_transforms[k])

    return clipped_e, t_stats, e_stats





"""
# TODO: Harmonize use of indices and various access functions
class NgramCount(Statistic):

    def __init__(self):
        self._t_count = defaultdict(lambda: defaultdict(lambda: 1))
        self._t_total = defaultdict(lambda: 1)
        self._e_count = defaultdict(lambda: defaultdict(lambda: 1))
        self._e_total = defaultdict(lambda: 1)

    def clear(self):
        self._t_count.clear()
        self._e_count.clear()
        self._t_total.clear()
        self._e_total.clear()

    def get_full_event(self, assignment : TagAssignment):

        for i in range(assignment.n_sents):
            for (src_tags, dest_tags) in self.sent_transitions(assignment, i):
                self._t_count[src_tags][dest_tags] += 1
                self._t_total[src_tags] += 1
            for (tags, word) in self.sent_emissions(assignment, i):
                self._e_count[tags][word] += 1
                self._e_total[tags] += 1

    def forget(self, data):
        (transitions, emissions) = data
        for (src_tags, dest_tag) in transitions:
            self._t_count[src_tags][dest_tag] -= 1
            self._t_total[src_tags] -= 1

        for (src_tags, word) in emissions:
            self._e_count[src_tags][word] -= 1
            self._e_total[src_tags] -= 1

    def store(self, event):
        (transitions, emissions) = event
        #print("transitions:" + str(transitions))
        for (src_tags, dest_tags) in transitions:
            self._t_count[src_tags][dest_tags] += 1
            self._t_total[src_tags] += 1

        #print("emissions:" + str(emissions))
        for (src_tags, word) in emissions:
            self._e_count[src_tags][word] += 1
            self._e_total[src_tags] += 1

    def get_event(self, assignment: TagAssignment, index) -> Event:
        return NgramCount.Events(self.t_event(assignment, index), self.e_event(assignment, index))

    def t_event(self, assignment, index):
        pass

    def e_event(self, assignment, index):
        pass

    def sent_transitions(self, assignment, i):
        pass

    def sent_emissions(self, assignment, i):
        pass

    def t_count(self, src_tag, dest_tag):
        return self._t_count[src_tag][dest_tag]

    def t_total(self, src_tag):
        return self._t_total[src_tag]

    def e_count(self, tag, word):
        return self._e_count[tag][word]

    def e_total(self, tag):
        return self._e_total[tag]
"""

class FallbackNgramCounts(NgramCount):

    class Ngram(Event):

        def __init__(self, i, ngram, variable=True):
            self._variable_index = i
            self.ngram = list(ngram)
            self._variable = variable

        def __getitem__(self, index):
            if index == 0:
                return tuple(self.ngram[:-1])
            if index == 1:
                return self.ngram[-1]
            else:
                raise IndexError()

        def set_value(self, v):
            if self._variable:
                #print('making the change:' + str(v))
                self.ngram[self._variable_index] = v

        def __str__(self):
            return str((self.ngram[:-1], self.ngram[-1]))

        def __repr__(self):
            return repr((self.ngram[:-1], self.ngram[-1]))

    def __init__(self, n=2):
        super(FallbackNgramCounts, self).__init__()
        self.n = n

    def transition_to(self, assignment, index, variable_index, variable=True):
        (i, j) = index
        relative_v_index = None
        if variable:
            (vi, vj) = variable_index
            relative_v_index = self.n - 1 - (j - vj)
        ngram = [assignment.get_tag((i, j - self.n + k + 1)) for k in range(self.n - 1)]
        ngram.append(assignment.get_tag(index))
        return FallbackNgramCounts.Ngram(relative_v_index, ngram, variable=variable)

    def emission_at(self, assignment, index, variable_index, variable=True):
        (i, j) = index
        relative_v_index = None
        if variable:
            (vi, vj) = variable_index
            relative_v_index = self.n - 2 - (j - vj)
        ngram = [assignment.get_tag((i, j - self.n + 2 + k)) for k in range(self.n - 1)]
        ngram.append(assignment.get_word((i, j)))
        return FallbackNgramCounts.Ngram(relative_v_index, ngram, variable=variable)

    def t_event(self, assignment, index):
        #print('getting transition for n=% d' % self.n)
        (i, j) = index
        return NgramCount.Events(
                *[self.transition_to(assignment, (i, k), index, variable=True) for k in range(j, j + self.n)]
        )

    def e_event(self, assignment, index):
        (i, j) = index
        return NgramCount.Events(
            *[
                self.emission_at(assignment, (i, k), index, variable=True)
                for k in range(j, min(j + self.n - 1, assignment.s_lengths[i]))
            ]
        )

    def sent_transitions(self, assignment, i):
        return [
            self.transition_to(assignment, (i, j), None, variable=False)
            for j in range(assignment.s_lengths[i] + self.n - 1)
        ]

    def sent_emissions(self, assignment, i):
        return [self.emission_at(assignment, (i, j), None, variable=False) for j in range(assignment.s_lengths[i])]


class AssignmentWithStorage(GibbsState[tuple[int]], TaggedSents):

    def __init__(self, decorated: TagAssignment, storage: NgramCount):
        self.base = decorated
        self.storage = storage
        self.una = self.base.una
        self.storage.refresh(self.base)
        self.s_lengths = self.base.s_lengths
        self.n_tokens = self.base.n_tokens

    def get_tags(self):
        return self.base.get_tags()

    def get_sents(self):
        return self.base.get_sents()

    def update(self, index, value):
        event = self.get_event(index)
        #print("transitions before setting: " + str(event[0]))
        event.set_value(self.base.get_tag(index))
        self.storage.forget(event)
        self.base.update(index, value)
        event.set_value(value)
        #print("transitions after setting: " + str(event[0]))
        self.storage.store(event)

    def admissible(self, index):
        return self.base.admissible(index)

    def get_event(self, index):
        return self.storage.get_event(self.base, index)

    def __str__(self):
        return str(self.base)


class NgramCountWithFallBack:

    class NgramWithFallBack(NgramCount.Event):

        class TruncatedNgram:

            def __init__(self, n, offset, ngram_event):
                self.full_event = ngram_event
                self.n = n
                self.offset = offset

            def __getitem__(self, item):
                if item == 0:
                    return tuple(self.full_event[0][self.offset:])
                else:
                    return self.full_event[item]

            def __str__(self):
                return str(self.full_event)

        def __init__(self, ngram_event: FallbackNgramCounts.Ngram, n):
            self.full_event = ngram_event
            self.fallback_events = [
                NgramCountWithFallBack.NgramWithFallBack.TruncatedNgram(n, offset, self.full_event)
                for offset in range(n-1)
            ]

        def __str__(self):
            return str(self.full_event)

        def set_value(self, v):
            self.full_event.set_value(v)

        def __getitem__(self, item):
            return self.fallback_events[item]

    class EventsWithFallBack(NgramCount.Events):

        def __init__(self, n, *events):
            super(NgramCountWithFallBack.EventsWithFallBack, self).__init__(
                *[NgramCountWithFallBack.NgramWithFallBack(e, n) for e in events]
            )

    def __init__(self, storages):
        self.storages = list(storages)
        self.n = len(self.storages)+1

    def forget(self, event):
        for (i, storage) in enumerate(self.storages):
            to_forget = NgramCount.Events(NgramCount.Events(*[e[i] for e in event[0]]),
                                          NgramCount.Events(*[e[i] for e in event[1]])
                                          )
            storage.forget(to_forget)

    def store(self, event):
        for (i, storage) in enumerate(self.storages):
            to_store = NgramCount.Events(NgramCount.Events(*[e[i] for e in event[0]]),
                                         NgramCount.Events(*[e[i] for e in event[1]])
                                         )
            storage.store(to_store)

    def refresh(self, assignment: TagAssignment):
        for (i, storage) in enumerate(self.storages):
            storage.refresh(assignment)

    def get_event(self, assignment: TagAssignment, index):
        original_event = self.storages[0].get_event(assignment, index)
        #print("original event: %s" % str(original_event))
        res = NgramCount.Events(
            NgramCountWithFallBack.EventsWithFallBack(self.n, *original_event[0]),
            NgramCountWithFallBack.EventsWithFallBack(self.n, *original_event[1])
            )
        #print("returning: %s" % str(res))
        return res


class HMMConditional(NormalizingConditional):

    def get_potentials(self, state: AssignmentWithStorage, index):
        tag_set = state.admissible(index)
        storage = state.storage
        event = state.get_event(index)

        def _potential(c_tag):
            event.set_value(c_tag)
            #print("transitions after: "+str(event[0]))
            #print("emissions after: "+str(event[1]))
            logp = 0.0
            local_t_counts = defaultdict(lambda: defaultdict(int))
            local_t_totals = defaultdict(int)
            local_e_counts = defaultdict(lambda: defaultdict(int))
            local_e_totals = defaultdict(int)

            for (s, t) in event[0]:
                logp += log(storage.t_count(s, t) + local_t_counts[s][t]) - log(storage.t_total(s) + local_t_totals[s])
                local_t_counts[s][t] += 1
                local_t_totals[s] += 1
                #print("trans logp:"+str(logp))

            for (s, w) in event[1]:
                logp += log(storage.e_count(s, w) + local_e_counts[s][w]) - log(storage.e_total(s) + local_e_totals[s])
                local_e_counts[s][w] += 1
                local_e_totals[s] += 1
                #print("em p:"+str(logp))

            return logp

        return tag_set, tuple([_potential(c_tag) for c_tag in tag_set])


class DirichletP:

    def __init__(self, base_p, alpha):
        self.base_p = base_p
        self.alpha = alpha

    def logp(self, event):
        (o, t) = event.counts()

        return log_add(log(self.alpha) + self.base_p.logp(event), log(o)) - log(self.alpha + t)


class RecursiveP:

    def __init__(self, offset, actual_p):
        self.offset = offset
        self.actual_p = actual_p

    def logp(self, event, *args, **kwargs):
        return self.actual_p.logp(event[self.offset], *args, **kwargs)


class UniformP:

    def __init__(self, k):
        self.logdensity = -log(k)

    def logp(self, _):
        return self.logdensity


def make_backoff_dirichlet(k, alphas, n, offset=0):
    if offset == n-1:
        return UniformP(k)
    else:
        base_p = RecursiveP(offset + 1, make_backoff_dirichlet(k, alphas, n, offset=offset + 1))
        return DirichletP(base_p, alphas[offset])


class EventForBackoffP:

    class View:

        def __init__(self, views, value):
            self.views = views
            self.value = value

        def __getitem__(self, item):
            return self.views[item]

        def counts(self):
            return self.value

        def __len__(self):
            return len(self.views)

    def __init__(self, events):
        self.events = [EventForBackoffP.View(self, counts) for counts in events]

    def counts(self):
        return self[0].counts()

    def __getitem__(self, item):
        return self.events[item]

    def __len__(self):
        return len(self.events)


class NgramFallBackConditional(NormalizingConditional):

    def __init__(self, alphas, k, n):
        super(NgramFallBackConditional, self).__init__()
        self.hierachical_p = make_backoff_dirichlet(k, alphas, n, offset=0)
        self.n = n

    def get_potentials(self, state: AssignmentWithStorage, index):
        tag_set = state.admissible(index)
        context = state.storage
        event = state.get_event(index)

        def _potential(c_tag):
            event.set_value(c_tag)

            logp = 0.0
            local_t_counters = [defaultdict(lambda: defaultdict(int)) for d in range(0, self.n)]
            local_t_totals = [defaultdict(int) for d in range(0, self.n)]
            local_e_counters = [defaultdict(lambda: defaultdict(int)) for d in range(0, self.n)]
            local_e_totals = [defaultdict(int) for d in range(0, self.n)]

            for fb_evnt in event[0]:
                r_t_counts = [
                    (context.storages[i].t_count(*fb_evnt[i]) + local_t_counters[i][fb_evnt[i][0]][fb_evnt[i][1]],
                     context.storages[i].t_total(fb_evnt[i][0]) + local_t_totals[i][fb_evnt[i][0]]
                    )
                    for i in range(0, self.n - 1)
                ]
                r_t_counts.append((None, None))
                h_t_event = EventForBackoffP(r_t_counts)

                logp += self.hierachical_p.logp(h_t_event)

                for i in range(self.n-1):
                    gram = fb_evnt[i]
                    local_t_counters[i][gram[0]][gram[1]] += 1
                    local_t_totals[i][gram[0]] += 1
                #print("trans logp:"+str(logp))

            for fb_evnt in event[1]:
                r_e_counts = [
                    (context.storages[i].e_count(*fb_evnt[i]) + local_e_counters[i][fb_evnt[i][0]][fb_evnt[i][1]],
                     context.storages[i].e_total(fb_evnt[i][0]) + local_e_totals[i][fb_evnt[i][0]]
                     )
                    for i in range(0, self.n - 1)
                ]
                r_e_counts.append((None, None))
                h_e_event = EventForBackoffP(r_e_counts)

                logp += self.hierachical_p.logp(h_e_event)

                for i in range(self.n-1):
                    gram = fb_evnt[i]
                    local_e_counters[i][gram[0]][gram[1]] += 1
                    local_e_totals[i][gram[0]] += 1
                #print("em p:"+str(logp))

            return logp

        return tag_set, tuple([_potential(c_tag) for c_tag in tag_set])


def uniform_index(assignment, rng):
    while True:
        yield assignment.indices[rng.randint(len(assignment.indices))]


def naive_sampler(sents, tagset, *args, initial_tags=None, rng=default_rng, n=2, **kwargs):
    assignment = TagAssignment(sents, tagset, initial_tags=initial_tags)
    index_sampler = uniform_index(assignment, rng)
    conditional_family = HMMConditional()
    storage = FallbackNgramCounts(n=n)
    state = AssignmentWithStorage(assignment, storage)
    return gibbs_sampler(state, index_sampler, conditional_family, *args, rng, **kwargs), state


def hierarchical_sampler(sents, tagset, *args, initial_tags=None, rng=default_rng, n=2, alphas=None, **kwargs):
    assignment = TagAssignment(sents, tagset, initial_tags=initial_tags)
    index_sampler = uniform_index(assignment, rng)
    if not alphas:
        alphas = [1.0 for _ in range(n-1)]

    conditional_family = NgramFallBackConditional(alphas, len(tagset), n)
    storage = NgramCountWithFallBack(FallbackNgramCounts(n - i) for i in range(n - 1))
    state = AssignmentWithStorage(assignment, storage)
    return gibbs_sampler(state, index_sampler, conditional_family, *args, rng, **kwargs), state


def extract_classes(state: AssignmentWithStorage):
    assignment = state.base
    quotient = defaultdict(set)
    for index in assignment.indices:
        tag = assignment.get_tag(index)
        quotient[tag].add(index)

    return frozenset({frozenset(q) for q in quotient.values()})


def build_prepare_fn(sents, s_lengths):
    def prepare(q):
        tags = [s_lengths[i]*[None] for i in range(len(sents))]
        l = len(q)
        for k, s in enumerate(q):
            for i in s:
                tags[i[0]][i[1]] = str(k)

        return TagAssignment(sents, ['%d' % i for i in range(l)], initial_tags=tags)
    return prepare


class HMMMAP:

    def __init__(self, collapse, prepare):
        self.collapse = collapse
        self.prepare = prepare
        self._max = 0
        self._argmax = None
        self.counter = defaultdict(int)

    def collect(self, state: NgramCount):
        collapsed = self.collapse(state)
        count = self.counter[collapsed] + 1
        self.counter[collapsed] = count
        if count > self._max:
            self._argmax = collapsed
            self._max = count

    def summary(self):
        return self.prepare(self._argmax)


class HMMAverage:

    def __init__(self, l, v, tag_to_i=lambda x: x, word_to_i=lambda x: x):
        self.l = l
        self.v = v

        self.tag_to_i = tag_to_i
        self.word_to_i = word_to_i

        self.delta = np.zeros(shape=(self.l, self.l))
        self.e = np.zeros(shape=(self.v, self.l))
        self._norm = np.zeros(shape=self.l,)
        self.n = 0

        self._sample_delta = np.zeros(shape=(self.l, self.l))
        self._sample_e = np.zeros(shape=(self.v, self.l))

    def collect(self, state: NgramCount):
        self._sample_delta.fill(0)
        self._sample_e.fill(0)
        self._norm.fill(0)

        for i in range(state.n_sents):
            for (t1, t2) in state.transitions(i):
                self._sample_delta[self.tag_to_i(t2), self.tag_to_i(t1)] += 1
                self._norm[self.tag_to_i(t1)] += 1
            for (t, w) in state.emissions(i):
                self._sample_e[self.word_to_i(w), self.tag_to_i(t)] += 1

        self._norm = self._norm + np.array([float(self._norm[i] == 0.) for i in range(self.l)])

        self.delta += (self._sample_delta / self._norm)
        self.e += (self._sample_e / self._norm)

        self.n += 1

    def summary(self):
        e_delta = self.delta / self.n
        e_e = self.e / self.n
        return e_delta, e_e








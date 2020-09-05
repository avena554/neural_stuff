from typing import Generic, TypeVar
from util.log_space import log_add, exp


class ConditionalProba:

    def get_conditional(self, state, index):
        raise NotImplemented


class NormalizingConditional(ConditionalProba):

    def get_potentials(self, state, index):
        raise NotImplemented

    def get_conditional(self, state, index):
        (candidates, potentials) = self.get_potentials(state, index)
        logfactor = log_add(*potentials)
        return candidates, [potential - logfactor for potential in potentials]


ID = TypeVar('ID')


class GibbsState(Generic[ID]):

    una = '<una>'

    def update(self, index: ID, value):
        pass

    def remove_assignment(self, index: ID):
        self.update(index, self.una)

    def get_value(self, index: ID):
        pass

    def admissible(self, index: ID):
        pass


S = TypeVar('S', bound=GibbsState)
E = TypeVar('E')
F = TypeVar('F')


class Transform(Generic[E, F]):

    def apply(self, e: E) -> F:
        pass


# FIXME: Put events in type specs
# FIXME: that iterable vs single thing shouldn't be imposed at this level of abstraction.
class Statistic(E):

    def forget(self, events: E):
        pass

    def collect(self, events: E):
        pass

    def clear(self):
        pass


class StatisticSetter(Generic[S, E]):

    def shape_input(self, assignment: S) -> E:
        raise NotImplemented

    def set(self, stat: Statistic[E], assignment: S):
        stat.clear()
        stat.collect(self.shape_input(assignment))


class Provider(Generic[S]):

    def get(self) -> S:
        raise NotImplemented


P = TypeVar('P')


class ProvidingState(Generic[ID, P], GibbsState[ID], Provider[P]):
    pass


# todo : refine type-parametrization
class ClippedState(Generic[S, ID, E], ProvidingState[ID, S]):

    def __init__(self, decorated: S, clip: 'Paperclip[S, ID, E]', collect_fn, forget_fn):
        self.base = decorated
        self.clip = clip
        self.una = self.base.una
        self.collect_fn = collect_fn
        self.forget_fn = forget_fn

    def update(self, index: ID, value):
        event: E = self.clip.extract_event(self, index)
        # print("transitions before setting: " + str(event[0]))
        event.set_value(self.base.get_value(index))
        self.forget_fn(event)
        self.base.update(index, value)
        event.set_value(value)
        # print("transitions after setting: " + str(event[0]))
        self.collect_fn(event)

    def admissible(self, index):
        return self.base.admissible(index)

    def get_value(self, index):
        return self.base.get_value

    def get(self):
        return self.base

    def __str__(self):
        return str(self.base)


# Sadly found no way in current python type-hinting
# to specify that S should inherit GibbsState[ID]
# (No generic type in upper bound).
class Paperclip(Generic[S, ID, E]):

    def __init__(self):
        self.collect_fns = []
        self.forget_fns = []

    # For synchronizing counts with state's updates
    def extract_event(self, state: S, index: ID) -> E:
        pass

    def add_stat(self, stat: Statistic[S, ID, F], transform: Transform[E, F]):
        def collect_fn(e: E):
            stat.collect(transform.apply(e))

        def forget_fn(e: E):
            stat.forget(transform.apply(e))
        self.collect_fns.append(collect_fn)
        self.forget_fns.append(forget_fn)

    def clip(self, state: S) -> ClippedState[S, ID]:
        return ClippedState[S, ID, E](state, self, self._collect, self._forget)

    @staticmethod
    def _apply_all(fns, event: E):
        for fn in fns:
            fn(event)

    def _collect(self, event: E):
        Paperclip._apply_all(self.collect_fns, event)

    def _forget(self, event: E):
        Paperclip._apply_all(self.forget_fns, event)


# GibbsSampler = Generator[GibbsState]
def gibbs_sampler(state: GibbsState[ID], index_sampler, conditional_family: ConditionalProba,
                  rng, verbose=True):
    current_state = state
    while True:
        yield current_state
        next_index = next(index_sampler)
        if verbose:
            print("resampling index %s" % str(next_index))
        state.remove_assignment(next_index)
        candidates, logits = conditional_family.get_conditional(current_state, next_index)
        next_value = rng.choice(candidates, p=[exp(logp) for logp in logits])
        if verbose:
            print("next value: %s" % str(next_value))
        current_state.update(next_index, next_value)
        # print(current_state.storage._e_count)


def print_index(_, index):
    print("sampling #%d\n" % index)


def collect_samples(sampler, collector, warmup: int, samples: int, pre_action=print_index):
    for i in range(warmup):
        pre_action(next(sampler), i)
    print("Warmup is over")
    for i in range(samples):
        pre_action(next(sampler), i)
        collector.collect(next(sampler))
    return collector.summary()

from typing import Generator, Generic, TypeVar
from util.log_space import log_add, exp, log


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

    def update(self, index: ID, value):
        raise NotImplemented

    def remove_assignment(self, index: ID):
        self.update(index, self.una)


#GibbsSampler = Generator[GibbsState]

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
        #print(current_state.storage._e_count)


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







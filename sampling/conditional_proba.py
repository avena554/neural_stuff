class ConditionalProba:

    def get_conditional(self, state, index):
        raise NotImplemented


class NormalizingConditional(ConditionalProba):

    def get_potentials(self, state, index):
        raise NotImplemented

    def get_conditional(self, state, index):
        potentials = self.get_potentials(state, index)
        factor = sum(potentials)
        return [potential/factor for potential in potentials]





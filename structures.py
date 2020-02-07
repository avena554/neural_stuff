class TaggedSents:

    def __init__(self, sents, tags):
        self._sents = sents
        self._tags = tags

    def get_sents(self):
        return self._sents

    def get_tags(self):
        return self._tags

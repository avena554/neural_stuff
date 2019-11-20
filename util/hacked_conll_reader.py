import nltk
from nltk.corpus.reader.conll import ConllCorpusReader


class HackedConllCorpusReader(ConllCorpusReader):
    def __init__(self, comment_symbol, *args, **kwargs):
        self.comment_symbol = comment_symbol
        super(HackedConllCorpusReader, self).__init__(*args, **kwargs)

    def hack_stream(self, stream):
        old_method = stream.readline

        def hacked_read_line():
            line = None
            while (not line) or (line[0] == self.comment_symbol):
                line = old_method()
            return line

        stream.readline = hacked_read_line

    def _read_grid_block(self, stream):
        self.hack_stream(stream)
        return super(HackedConllCorpusReader, self)._read_grid_block(stream)



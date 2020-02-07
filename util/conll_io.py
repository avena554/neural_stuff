from nltk.corpus.reader.conll import ConllCorpusReader
from pathlib import Path
from structures import TaggedSents
import re


u_fields = [
            ConllCorpusReader.IGNORE, ConllCorpusReader.WORDS,
            ConllCorpusReader.IGNORE, ConllCorpusReader.POS
        ]
s_fields = [
    ConllCorpusReader.WORDS,
    ConllCorpusReader.POS
]


class HackedConllCorpusReader(ConllCorpusReader):
    def __init__(self, comment_symbol, *args, **kwargs):
        super(HackedConllCorpusReader, self).__init__(*args, **kwargs)
        self.comment_symbol = comment_symbol
        self.ignore = re.compile(r'^(#|\d+-\d+\t).*$')

    def hack_stream(self, stream):
        if not('_hacked' in dir(stream)):
            old_method = stream.readline

            def hacked_read_line():
                line = old_method()
                while self.ignore.match(line):
                    line = old_method()

                return line

            stream.readline = hacked_read_line
            stream._hacked = True

    def _read_grid_block(self, stream):
        self.hack_stream(stream)
        return super(HackedConllCorpusReader, self)._read_grid_block(stream)


def get_tagged_sents(input_file, fields=s_fields):
    f_path = Path(input_file)
    directory = str(f_path.parent)
    base_name = f_path.name
    reader = HackedConllCorpusReader(
        '#',
        str(directory), [base_name],
        fields,
        separator='\t'
        )

    sents, tags = [[w[0] for w in s] for s in reader.tagged_sents()], [[w[1] for w in s] for s in reader.tagged_sents()]
    tagged_pair = TaggedSents(sents, tags)
    return tagged_pair


class CONLLWriter:

    def __init__(self, tagged_sent: TaggedSents):
        self.tagged_sent = tagged_sent

    def as_string(self):
        return '\n\n'.join(
            [
                '\n'.join(
                    ['\t'.join((sent[i], tags[i])) for i in range(len(sent))]
                )
                for sent, tags in zip(self.tagged_sent.get_sents(), self.tagged_sent.get_tags())
            ]
        )

    def write(self, file):
        with open(file, 'w+') as desc:
            desc.write(self.as_string())
            desc.close()

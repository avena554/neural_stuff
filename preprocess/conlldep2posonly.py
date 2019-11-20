import argparse
import re
from nltk.corpus.reader.conll import ConllCorpusReader
from util.hacked_conll_reader import HackedConllCorpusReader
from util.print_utils import print_on_err
from pathlib import Path

rtn_strip = re.compile("(^.*)(\n?)")


def get_mapping(lines, separator='\t'):
    pos_map={}
    for l in lines:
        if l.strip():
            fields = l.split(separator)
            if fields[0] in pos_map:
                print_on_err("Warning: source TAG twice in map file. Keeping second only")
            pos_map[fields[0]] = rtn_strip.match(fields[1]).group(1)
    return pos_map


def get_tagged_sents(input_file):
    f_path = Path(input_file)
    directory = str(f_path.parent)
    base_name = f_path.name
    reader = HackedConllCorpusReader(
        '#',
        str(directory), [base_name],
        [
            ConllCorpusReader.IGNORE, ConllCorpusReader.WORDS,
            ConllCorpusReader.IGNORE, ConllCorpusReader.POS
        ]
        )
    return reader.tagged_sents()


parser = argparse.ArgumentParser(description=
                                 """Shamelessly tear apart a dependency corpus in Conll format into a mere
                                 POS annotated corpus
                                 May apply some clever TAGSet mapping in the meantime,
                                  for a reduced (and hopefully more universal) 
                                 TAGSet
                                 """)

parser.add_argument("-i", type=str, dest="input")
parser.add_argument("-m", type=str, dest="map")

if __name__ == "__main__":
    args = parser.parse_args()

    if args.input:
        tagged_sents = get_tagged_sents(args.input)
        print(tagged_sents)

    else:
        parser.print_usage()
    if args.map:
        with open(args.map) as map_desc:
            print(get_mapping(map_desc))






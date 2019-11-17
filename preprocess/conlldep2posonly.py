import argparse


def get_conll(lines):
    for l in lines:
        if l.strip():
            fields = l.split("\t")
            print(fields[0] + "\t" + fields[1] + "\t" + fields[2])


parser = argparse.ArgumentParser(description=
                                 """Shamelessly tear appart a dependcy corpus in Conll format into a mere
                                 POS annotated corpus
                                 May apply some clever TAGSET mapping in the meantimne,
                                  for a reduced (and hopefully more universal) 
                                 TAGset
                                 """)

parser.add_argument("-i", type=str, dest="input")
parser.add_argument("-m", type=str, dest="map")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.input:
        with open(args.input) as input:
            get_conll(input)






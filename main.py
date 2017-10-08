import sys
import torch

import block_seq2seq
import prepare_ptb_postags

from torch.autograd import Variable
from torch import LongTensor
from torch.optim import Adam
from torch.nn import NLLLoss

from block_seq2seq import SimpleBlockSeq2Seq
from block_seq2seq import train_blockSeq2Seq

from prepare_ptb_postags import LexicalMap, PTBPostagReader
from prepare_ptb_postags import remap, pad_right

train = False

#Hyperparameters
IN_EMB_DIM = 500
OUT_EMB_DIM = 100
H_DIM = 500
N_ENC_LAYERS = 5
N_DEC_LAYERS = 5
BATCH_SIZE = 10

NUM_EPOCH = 50

#get data
print("reading data")
reader = PTBPostagReader()
((batched_sentences, batched_tags), s_length, length_batches)  = reader.get_batched_postags(BATCH_SIZE, reader.train_files)

print("Number of instances: %d"%(len(batched_sentences)*len(batched_sentences[0])))

s_lm = LexicalMap()
t_lm = LexicalMap()

print("building vocabulary")

in_snt = LongTensor(remap(batched_sentences, s_lm, True))
gold_tags = LongTensor(remap(batched_tags, t_lm, True))

in_voc_size = s_lm.size
out_voc_size = t_lm.size

print("%d words, %d tags"%(in_voc_size, out_voc_size))

if(train):

    print("building model")

    model = SimpleBlockSeq2Seq(in_voc_size, out_voc_size, IN_EMB_DIM, OUT_EMB_DIM, H_DIM, N_ENC_LAYERS, N_DEC_LAYERS)

    print("starting training")

    """
    weigths = []
    for k in range len(batched_sentences):
        p_bw = []
        for l in range s_length:
            p_bw.append([])
            
        for l in range s_length:    
            for m in range BATCH_SIZE:
                if(length_batches[k][m] <= l):
                    p_bw[l].append(1)
                else:
                    p_bw[l].append(0)


        weights.append(p_bw)

    weights = Variable(Tensor(weigths))
    """
    
    model, loss = train_blockSeq2Seq(model, in_snt, gold_tags, lambda x: Adam(x), NLLLoss(ignore_index=t_lm.retrieve_index(reader.eos_symb)), True, NUM_EPOCH, t_lm)



    torch.save(model, "pos_tagger_model")
    #torch.save(tagger.state_dict(), "model")

else:
    model = torch.load("pos_tagger_model")
    model = model.cpu()

    print("ready for input.")

    for line in sys.stdin:
        input = [reader.sos_symb] + line.replace("\n", "").split(" ")
        pad_right(s_length, input, reader.eos_symb)
        print(input)
        print("\n")
        
        input = [ [ [tok] for tok in input ] ]
        print(input)
        print("\n")

        input = remap(input, s_lm, True)
        print(input)
        print("\n")

        input = input[0]
        print(input)
        print("\n")

        
        input = Variable(LongTensor(input)).cpu()
        print(input)
        print("\n")

        gold = Variable(LongTensor([[t_lm.retrieve_index(reader.sos_symb)]]))
        print("gold:")
        print(gold)
        
        out = model.forward(input, gold, [True]*s_length, model.init_hidden(1))
        out = torch.topk(out, 1)[1]
        print("output: ")
        print(out)

        postags = remap([out.view(s_length, 1).data],t_lm, False)
        print(postags)
                      

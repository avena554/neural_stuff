import torch
import torch.nn as nn

from torch.autograd import Variable
from torch import Tensor

from prepare_ptb_postags import remap

class SimpleBlockSeq2Seq(nn.Module):
    def __init__(self, in_voc_size, out_voc_size, in_emb_dim, out_emb_dim, hidden_dim, num_enc_layers, num_dec_layers):
        super(SimpleBlockSeq2Seq, self).__init__()

        #modules
        self.lhs_emb_layer = nn.Embedding(in_voc_size, in_emb_dim)
        self.encoder = nn.LSTM(input_size=in_emb_dim, hidden_size=hidden_dim, num_layers=num_enc_layers)

        self.rhs_emb_layer = nn.Embedding(out_voc_size, out_emb_dim)
        self.decoder = nn.LSTM(input_size=out_emb_dim, hidden_size=hidden_dim, num_layers=num_dec_layers)

        self.reSize = nn.Linear(hidden_dim, out_voc_size)
        self.softmax = nn.LogSoftmax()


        #hyperparameters
        self.hidden_dim = hidden_dim
        
        self.num_enc_layers = num_enc_layers
        self.num_dec_layers = num_dec_layers

        #constants
        self.out_voc_size = out_voc_size
        self.in_voc_size = in_voc_size
        
    def forward(self, in_batched_seq, gold_batched_seq, feed_previous, hidden_and_cell):
        #in_batched_seq and gold_batched_seq are both of shape (sentence_length, batch-size)
        

        #Embed each token of each input
        embedded_seqs = self.lhs_emb_layer(in_batched_seq)

        in_shape = in_batched_seq.data.size() 
        sentence_length = in_shape[0]
        batch_size = in_shape[1]
        
        #encode input seq
        (_, (contexts_tensor, cells_tensor)) = self.encoder(embedded_seqs, hidden_and_cell)

        #decoding. need to actually write the loop to reenter last decoded symbol as the next decoder input whenever feed previous tells us to.

        decoder_input = gold_batched_seq[0]

        outputs = []
        
        #one step computation
        for i in range (0, sentence_length):
            '''
            print("decoder:")
            print(self.decoder)
            print("\n")
            print("input")
            print(decoder_input)
            print("\n")
            print("states") 
            print((contexts_tensor, cells_tensor))
            '''
            '''
            print("decoder input size")
            print(decoder_input.data.size())
            '''
            decoder_input = self.rhs_emb_layer(decoder_input.view(batch_size,1))


            
            
            #should now be batch_size , 1, emb_dim
            '''
            print("decoder embedded input size")
            print(decoder_input.data.size())
            '''
            
            decoder_input = decoder_input.view(1, batch_size, -1)
            '''
            print("reviewed decoder embedded input size")
            print(decoder_input.data.size())
            
            print("context size:")
            print(contexts_tensor.size())

            print("cell size")
            print(cells_tensor.size())
            '''
            (_, (contexts_tensor, cells_tensor)) = self.decoder(decoder_input, (contexts_tensor, cells_tensor))

            scores_batched = self.reSize(contexts_tensor[self.num_dec_layers-1])
            predictions = self.softmax(scores_batched)

            outputs.append(predictions)
                
            if(feed_previous[i]):
                decoder_input = torch.topk(predictions, 1)[1]
            else:
                #forced teaching
                decoder_input = gold_batched_seq[i]
        #print([x.size() for x in outputs])
        return torch.stack(outputs).cuda()

    def init_hidden(self, batch_size):         
         return (Variable(Tensor(self.num_enc_layers, batch_size, self.hidden_dim).zero_()), Variable(Tensor(self.num_enc_layers, batch_size, self.hidden_dim).zero_()))



def gpu_cpu_switch(cuda):
    if(cuda):
        return lambda x : x.cuda()
    else:
        return lambda x : x.cpu()

    
"""
in_data: (num_batches, seq_length, batch_size) Long tensor
out_data: (num_batches, seq_length, batch_size) Long tensor
loss: loss function
cuda: boolean
"""        
def train_blockSeq2Seq(model, in_data, gold_data, optimizer_factory, loss_fn, cuda, epochs = 1, l_map = None, output_period = 10):

    dims = in_data.size()
    
    num_batches = dims[0]
    seq_length = dims[1]
    batch_size = dims[2]
    
    switch = gpu_cpu_switch(cuda)

    model = switch(model)
    loss_fn = switch(loss_fn)

    optimizer = optimizer_factory(model.parameters())
    
    for e in range(epochs):
        total_loss = 0
        batch_loss = 0
        batches_done = 0
        
        for i in range(num_batches):
            batches_done = batches_done + 1
            optimizer.zero_grad()
        
            seq_batched = Variable(switch(in_data[i]))
            gold_batched = Variable(switch(gold_data[i]))
            '''
            print("dims of enc input:")
            print(seq_batched.data.size())
            print("dims of dec input:")
            print(gold_batched.data.size())
            '''
            hidden_and_cell = [switch(model.init_hidden(batch_size)[k]) for k in [0,1]]
            '''
            print("dim hidden:")
            print(hidden_and_cell[0].data.size())

            print("dim cell:")
            print(hidden_and_cell[1].data.size())
            '''
            feed_previous = [False]*seq_length
        
            predictions = model(seq_batched, gold_batched, feed_previous, hidden_and_cell)

            #print("gold: ")
            #print(gold_batched)

            '''
            if(l_map):
                print("human readable gold:")
                print(remap([gold_batched.data], l_map, False))
                
            for m in range(seq_length):
                for z in range(batch_size):
                    if(gold_batched.data[m][z] < 0 or gold_batched.data[m][z] > (model.out_voc_size-1) ):
                        print("Ouch! Found inconsistant gold data: %d"%gold_batched.data[m][z])
            '''

            #print("prediction: ")
            #print(predictions)
            
            loss = loss_fn(predictions.view(seq_length*batch_size, -1), gold_batched.view(seq_length*batch_size))
    
            #average here before next line?
            loss.backward()
            optimizer.step()

            total_loss += loss.data[0]
            batch_loss += loss.data[0]

            if(batches_done % output_period == 0):
                print("\taverage loss over the last %d batches: %f\n"%(output_period, (batch_loss/output_period)))
                batch_loss = 0
            
            

        print("[after epoch %d] mean loss = %f\n" % (e+1, total_loss/num_batches))

    return model, loss.data[0]
    

    
        

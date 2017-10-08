import torch
from nltk.corpus import ptb
from prefixtrees import TrieDict
import re

default_train_sections = range(2, 22)
default_dev_sections = [1,22,24]
default_test_sections = [23]

class LexicalMap:

    def __init__(self):
        self._string_2_int = TrieDict()
        self._int_2_string = []
        self.size = 0

    #add if not present    
    def retrieve_index(self, w):
        try:
            return self._string_2_int.get_value(w)
        except(KeyError):
            self.size = self.size + 1
            self._string_2_int.set_value(w, self.size-1)
            self._int_2_string.append(w)
            return (self.size-1)

    def retrieve_symbol(self, i):
        return self._int_2_string[i]


            
class PTBPostagReader:

    def _extract_wsj_sec_num(self, fileID):
        matcher = self._wsj_pattern.match(fileID)
        if(matcher):
            return int(matcher.group(1))
        else:
            return None
        
    
    def _admits(self, file_id, sec_lists, extraction_fn):
        return extraction_fn(file_id) in sec_lists
                
        
    def __init__(self, train_sections=default_train_sections, dev_sections=default_dev_sections, test_sections=default_test_sections, sos_symb = "<SOS>", eos_symb="</EOS>"):
        self._wsj_pattern = re.compile(r"WSJ/(\d\d)/")

        self.train_sections = train_sections
        self.dev_sections = dev_sections
        self.test_sections = test_sections

        self.sos_symb = "<SOS>"
        self.eos_symb = "</EOS>"
        
        (self.train_files, self.dev_files, self.test_files) = [ [file_id for file_id in ptb.fileids() if self._admits(file_id, sections, lambda x: self._extract_wsj_sec_num(x))] for sections in (self.train_sections, self.dev_sections, self.test_sections)]


    def _pad_right(self, obj_length, seq_to_pad):
        pad_right(obj_length, seq_to_pad, (self.eos_symb, self.eos_symb))
        
    def get_batched_postags(self, batch_size, files):
        print("obtaining parsed snts")
        parsed_sents = ptb.parsed_sents(files)
        
        num_sentences = len(parsed_sents)
    
        num_batches = (num_sentences + num_sentences % batch_size ) // batch_size
        print("num batches: %d"%num_batches)

        

        max_length = 0
       
        batches = []
        length_batches = []
       
        for i in range(num_batches):
            batch = [ tree.pos() for tree in parsed_sents[i*batch_size:(i+1)*batch_size] ]
            if((i+1)*batch_size > num_sentences):
                batch.extend(parsed_sents[0:num_sentences%batch_size])

            batched_length = []
            for tagged_words in batch:
               #add start of sentence token (and corresponding tag to the left)
                tagged_words.insert(0, (self.sos_symb, self.sos_symb))       
                            
                l = len(tagged_words)
                batched_length.append(l)
                if(max_length < l):
                    max_length = l
                    
            batches.append(batch)
            length_batches.append(batched_length) 

        print("Max length: %d"%max_length)

        for batch in batches:
            for tagged_words in batch:
                #print("len before: %d"%len(tagged_words))
                self._pad_right(max_length, tagged_words)
                #print("padded seq: %s"%tagged_words)
                #print("len after: %d"%len(tagged_words))
        
        #padd everything
        final_batch = [ [ [ [ tagged_words[i][j] for i in range(max_length) ] for tagged_words in batch ] for batch in batches ] for j in [0,1] ]

        return (final_batch, max_length, length_batches)

    
def remap(batches, l_map, string2int):
    if(string2int):
        fn = lambda x:l_map.retrieve_index(x)
    else:
        fn = lambda x: l_map.retrieve_symbol(x)
        

    remapped = [ [ [ fn(token)  for token in batched_timestep ] for batched_timestep in batched_seq ] for batched_seq in batches  ]
    return remapped



def pad_right(obj_length, seq_to_pad, symb):
        #print("objective length: %d"%obj_length)
        padding_size = obj_length - len(seq_to_pad)
        #print("addding %d symbols"%padding_size)
        seq_to_pad.extend(padding_size*[symb])

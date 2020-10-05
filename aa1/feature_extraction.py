
#basics
import pandas as pd
import torch
import spacy


def extract_features(data:pd.DataFrame, max_sample_length:int, pos_tags:list, vocab:list, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    
    def get_vector(ix:int):
        '''Returns the vector of a token_id'''
        string = vocab[ix]
        vector = nlp.vocab.get_vector(string)
        return vector

    
    train_X = []
    val_X = []
    test_X = []
    
    nlp = spacy.load('en_core_sci_md')
    data['pos'] = pos_tags  # add the POS-tags to the dataframe for easier access
    
    # POS-tags mapping
    pos2id = {p:i+1 for i,p in enumerate(set(pos_tags))}  # reserve 0 for padding
    
    for sid in set(data.sentence_id):
        sent = data[data['sentence_id']==sid]
        splt = sent['split'].iloc[0]
        pad_len = max_sample_length - len(sent)
        sample = list(sent['token_id']) + [1]*pad_len  # token_id, pad with 1 for <pad>
        prev_token = [0] + [sample[i-1] for i in range(1,len(sample))]  # previous token_id; the first token gets 0
        next_token = [sample[i+1] for i in range(len(sample)-1)] + [0]  # next token_id; the last token gets 0
        pos = [pos2id[p] for p in sent['pos']] + [0]*pad_len  # POS-tags, pad with 0 as reserved in pos2id
        vec = [get_vector(i) for i in sample]  # FastText vectors
              
        features = [[sample[i]] + [prev_token[i]] + [next_token[i]] + [pos[i]] + list(vec[i]) for i in range(len(sample))]
        
        if splt == 'Train':
            train_X.append(features)
        elif splt == 'Val':
            val_X.append(features)
        elif splt == 'Test':
            test_X.append(features)
            
    return torch.tensor(train_X).to(device), torch.tensor(val_X).to(device), torch.tensor(test_X).to(device)

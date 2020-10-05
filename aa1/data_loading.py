
#basics
import random
import pandas as pd
import torch
import xml.etree.ElementTree as ET
import os
from collections import Counter
from torchtext.vocab import Vocab
import spacy
import matplotlib.pyplot as plt



class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for i,t_row in sample_tokens.iterrows():

            is_ner = False
            for i, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()



class DataLoader(DataLoaderBase):


    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

    
    def _parse_data(self, data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.


        def get_paths(top):
            '''Returns a list of paths to all xml files in all sub folders of the input top folder.'''

            # Walk through data dirs to collect paths to train data
            train_paths = []
            for directory in next(os.walk(os.path.join(data_dir, 'Train')))[1]:
                for file in next(os.walk(os.path.join(data_dir, 'Train', directory)))[2]:
                    if file.endswith('.xml'):
                        train_paths.append(os.path.join(data_dir, 'Train', directory, file))

            # Reserve 10% as val data
            random.shuffle(train_paths)
            val_ix = len(train_paths)//10
            val_paths = train_paths[:val_ix]
            train_paths = train_paths[val_ix:]

            # Repeat for test data
            test_paths = []
            for directory in next(os.walk(os.path.join(data_dir, 'Test', 'Test for DrugNER task')))[1]:
                for file in next(os.walk(os.path.join(data_dir, 'Test', 'Test for DrugNER task', directory)))[2]:
                    if file.endswith('.xml'):
                        test_paths.append(os.path.join(data_dir, 'Test', 'Test for DrugNER task', directory, file))
            
            return train_paths, val_paths, test_paths


        def update_data_df(sentence, data_df, max_len, pos_tags):
            '''Updates metadata from an etree sentence element'''
            spacy_parsed = nlp(sentence.attrib['text'])
            tokenized = [token.text for token in spacy_parsed]
            max_len = max(len(tokenized), max_len)
            
            #update dataframe
            for token in spacy_parsed:
                new_data_row = pd.Series([sentence.get('id'), token.text, int(token.idx), int(token.idx+len(token.text)), splt])
                data_df = data_df.append(new_data_row, ignore_index=True)
                pos_tags.append(token.tag_)
                
            return data_df, max_len, pos_tags
        
        
        def update_ner_df(sentence, ner_df):
            '''Updates ner metadata from an etree sentence elment'''
            entities = sentence.findall('entity')
            
            for ent in entities:
                ner_type = ent.get('type')
                ner_spans = ent.get('charOffset').split(';')
                
                #update dataframe
                for span in ner_spans:
                    ner_char_start_id = span.split('-')[0]
                    ner_char_end_id = span.split('-')[1]
                    new_ner_row = pd.Series([sentence.get('id'), ner_type, int(ner_char_start_id), int(ner_char_end_id)])
                    ner_df = ner_df.append(new_ner_row, ignore_index=True)
                    
            return ner_df
        

        print('Initializing...')
        
        pos_tags = []
        max_len = 0
        
        data_df = pd.DataFrame()
        ner_df = pd.DataFrame()
        
        nlp = spacy.load('en_core_sci_md')
        
        #get files to process
        train_paths, val_paths, test_paths = get_paths(data_dir)

        #parse xml
        for paths in [train_paths, val_paths, test_paths]:
            splt = 'Train' if paths==train_paths else 'Val' if paths==val_paths else 'Test'
            print('Processing {} data...'.format(splt))
            for file in paths:
                tree = ET.parse(file)
                root = tree.getroot()

                for sentence in root.findall('sentence'):
                    data_df, max_len, pos_tags = update_data_df(sentence, data_df, max_len, pos_tags)
                    ner_df = update_ner_df(sentence, ner_df)
        
        # Finalize data_df & vocab
        data_df.columns=["sentence_id", "token_id", "char_start_id", "char_end_id", "split"]
        counter = Counter(data_df.token_id)  #token_id are actual tokens at this point
        vocab = Vocab(counter)
        word2id = vocab.stoi
        id2word = vocab.itos
        data_df.token_id = [word2id[w] for w in data_df.token_id]  #convert tokens to ids
        
        # Finalize ner_df
        ner_df.columns=["sentence_id", "ner_id", "char_start_id", "char_end_id"]
        ner2id = {'NEG':0, 'drug':1, 'drug_n':2, 'brand':3, 'group':4}
        id2ner = {i:n for (n,i) in ner2id.items()}
        ner_df.ner_id = [ner2id[w] for w in ner_df.ner_id]  #convert entities to ids
        
        # set variables
        setattr(self, 'data_df', data_df)
        setattr(self, 'ner_df', ner_df)
        setattr(self, 'vocab', id2word)
        setattr(self, 'id2ner', id2ner)
        setattr(self, 'max_sample_length', max_len)
        setattr(self, 'id2word', id2word)
        setattr(self, 'pos_tags', pos_tags)
        
        print('Done!')
        

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        train_y = []
        val_y = []
        test_y = []
        
        for sid in set(self.data_df.sentence_id):
            sent = self.data_df[self.data_df['sentence_id']==sid]
            ners = self.ner_df[self.ner_df['sentence_id']==sid]
            sample_y = []
            for i, token in sent.iterrows():
                y = 0
                for i, ner in ners.iterrows():
                    if token['char_start_id'] >= ner['char_start_id'] and token['char_start_id'] <= ner['char_end_id']:
                        y = ner['ner_id']
                sample_y.append(y)
                
            sample_y = sample_y + [0 for i in range(self.max_sample_length - len(sample_y))]  #end padding with negative class
            
            if token['split'] == 'Train':
                train_y.append(sample_y)
            elif token['split'] == 'Val':
                val_y.append(sample_y)
            elif token['split'] == 'Test':
                test_y.append(sample_y)

        return torch.tensor(train_y).to(self.device), torch.tensor(val_y).to(self.device), torch.tensor(test_y).to(self.device)
    

    def plot_split_ner_distribution(self):
        # should plot a histogram displaying ner label counts for each split
        labels = self.get_y()
        sequence = [[self.id2ner[i] for i in torch.flatten(y).tolist() if i!=0] for y in labels]
        
        plt.hist(sequence, edgecolor="black")
        plt.legend(['Train', 'Val', 'Test'])
        plt.xlabel('NER type')
        plt.ylabel('Count')
        plt.show()


    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        sequence = []
        for sid in set(self.data_df.sentence_id):
            sent = self.data_df[self.data_df['sentence_id']==sid]
            sequence.append(len(sent))
            
        plt.hist(sequence, align='left', bins=max(sequence), edgecolor='black')
        plt.xlabel('# Tokens')
        plt.ylabel('# Sentences')
        plt.show()


    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        train_y, val_y, test_y = self.get_y()
        samples = train_y.tolist() + val_y.tolist() + test_y.tolist()
        sequence = []
        for sample in samples:
            sequence.append(len([n for n in sample if n!=0]))
            
        plt.hist(sequence, align='left', bins=max(sequence), edgecolor='black')
        plt.xlabel('# NERs')
        plt.ylabel('# Sentences')
        plt.show()
        

    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        pass




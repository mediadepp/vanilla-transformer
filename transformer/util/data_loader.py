from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None 

    def __init__(self, ext, tokenize_en, tokenize_de, init_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en 
        self.tokenize_de = tokenize_de 
        self.init_token = init_token 
        self.eos_token = eos_token 
        print("Initialising dataset ...")
    
    def make_dataset(self):
        # This is for (".de", ".en")
        self.source = Field(tokenize=self.tokenize_de, init_token=self.init_token, eos_token=self.eos_token, 
                            lower=True, batch_first=True)
        self.target = Field(tokenize=self.tokenize_en, init_token=self.init_token, eos_token=self.eos_token, 
                            lower=True, batch_first=True)

        if self.ext == (".en", ".de"):   
            self.source, self.target = self.target, self.source 
        
        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data 

    def build_vocab(self, train_data, min_freq):
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq) 
    
    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test), 
                                                                              batch_size=batch_size, device=device) 
        print("Initialising the dataset is done!")
        return train_iterator, valid_iterator, test_iterator 

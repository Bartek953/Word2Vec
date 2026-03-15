import Trainer
import DataDownloader
import numpy as np

gen = DataDownloader.get_text8_generator()
for i in range(1000):
    print(next(gen), end=' ')

word_to_ind, ind_to_word = DataDownloader.get_dics()

def get_embeding(word: str):
    if word in word_to_ind:
        index = word_to_ind[word]
        
        return Trainer.model.get_embeding(index)
    else:
        print(f"Error: word {word} no in dictionary!")
        



import numpy as np
from DataDownloader import data_downloader
from collections import deque
from typing import Iterator, Tuple, List
import random

class DataLoader:
    def __init__(self, context_size: int = 2):
        self.generator = data_downloader.get_text_generator()
        self.word_to_ind, self.ind_to_word = data_downloader.get_dics()
        self.context_size = context_size
        self.vocab_size = data_downloader.get_vocab_size()
        self.dataset_size = data_downloader.get_dataset_size()


    def generate_samples(self) -> Iterator[Tuple[np.ndarray, int]]:
        sliding_window = deque(maxlen = 2 * self.context_size + 1)
        
        while True:
            sliding_window.clear()
            
            word_counter:int  = 0
            
            for word in self.generator:
                word_ind: int = 0
                
                if word in self.word_to_ind:
                    word_ind = self.word_to_ind[word]
                else:
                    word_ind = 0 # [unknown]
                    
                sliding_window.append(word_ind)
                
                if len(sliding_window) == sliding_window.maxlen:
                    full_window = list(sliding_window)
                    target: int = full_window[self.context_size]
                    context: list = full_window[:self.context_size] + full_window[self.context_size + 1:]
                    yield np.array(context), target
                    
                word_counter += 1
                if word_counter >= self.dataset_size:
                    break
                
    def generate_batches(self, shuffle_size: int = 128, batch_size: int = 16):
        sample_gen = self.generate_samples()
        
        while True:
            buffer = []
            
            while len(buffer) < shuffle_size:
                buffer.append(next(sample_gen))
            
            random.shuffle(buffer)
            for i in range(0, len(buffer), batch_size):
                # i = random.randint(0, shuffle_size - batch_size)
                batch = buffer[i: i + batch_size]
                
                if len(batch) == batch_size:
                    yield (np.array([p[0] for p in batch]), np.array([p[1] for p in batch]))
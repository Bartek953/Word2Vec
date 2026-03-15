import os
import urllib.request
import zipfile
from collections.abc import Iterator

class DataDownloader:
    def __init__(self, text_lim: int = -1, extracted_path: str = "text8", url: str = "http://mattmahoney.net/dc/text8.zip"):
        self.extracted_path = extracted_path
        self.url = url
        
        self.text_lim = text_lim
        
        self.__download_data__()
        self.vocab_size = 0
        self.dataset_size = 0
        self.__create_dics__()
        
    # downloads dataset if not already downloaded
    def __download_data__(self) -> None:
        zip_path: str = f"{self.extracted_path}.zip"

        # 1. Checks whether file already exists
        if os.path.exists(self.extracted_path):
            print("File text8 already exists")
            return

        # 2. File doesn't exists, download the zip file
        if not os.path.exists(zip_path):
            print("Downloading text8")
            urllib.request.urlretrieve(self.url, zip_path)
            print("Succesfully text8 downloaded")

        # 3. Extract the file
        print("Extracting...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall()
       
    # creates vocabulary dictionaries 
    def __create_dics__(self) -> None:
        text_gen = self.get_text_generator(infinite=False)
        
        self.word_to_ind = dict()
        self.ind_to_word = dict()
        
        curr_word_index: int = 0
        
        self.vocab_size = 0
        self.dataset_size = 0
        
        for word in text_gen:
            self.dataset_size += 1
            if word not in self.word_to_ind:
                self.vocab_size += 1
                
                self.word_to_ind[word] = curr_word_index
                self.ind_to_word[curr_word_index] = word
                
                curr_word_index += 1
                
    # returns created dictionaries: word_to_ind, ind_to_word
    def get_dics(self) -> tuple[dict, dict]:
        return self.word_to_ind, self.ind_to_word
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_dataset_size(self) -> int:
        return self.dataset_size
    
    # returns dataset generator (to not load whole dataset to RAM at once)
    def get_text_generator(self, chunk_size = 1024 * 1024, infinite = True) -> Iterator[str]:
        while True:
            remaining = ""
            word_counter = 0
            with open(self.extracted_path, 'r') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk:
                        if remaining:
                            yield remaining
                        break
                    
                    chunk = remaining + chunk
                    words = chunk.split(' ')
                    #in case last word isn't fully loaded
                    remaining = words.pop()
                    
                    for word in words:
                        if word: 
                            yield word
                        word_counter += 1
                        if word_counter >= self.text_lim and self.text_lim > 0:
                            break
                    if word_counter >= self.text_lim and self.text_lim > 0:
                        break
            if infinite == False:
                break
            
data_downloader = DataDownloader(text_lim = 500000)
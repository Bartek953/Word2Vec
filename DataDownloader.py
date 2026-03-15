import os
import urllib.request
import zipfile

extracted_path: str = 'text8'
vocab_size = 0
dataset_size = 0

words_limit = 10000

# Download the data
def download_text8_data():
    url: str = 'http://mattmahoney.net/dc/text8.zip'
    zip_path: str = 'text8.zip'

    # 1. Checks whether file already exists
    if os.path.exists(extracted_path):
        print("File text8 already exists")
        return

    # 2. File doesn't exists, download the zip file
    if not os.path.exists(zip_path):
        print("Downloading text8")
        urllib.request.urlretrieve(url, zip_path)
        print("Succesfully text8 downloaded")

    # 3. Extract the file
    print("Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall()
      
# Yields words, to not load whole dataset into RAM 
def get_text8_generator(file_path: str = extracted_path, chunk_size = 1024 * 1024, infinite=True):
    download_text8_data()
    while True:
        remaining = ""
        word_counter = 0
        with open(file_path, 'r') as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    if remaining: yield remaining
                    break
                
                chunk = remaining + chunk
                words = chunk.split(' ')
                #in case last word isn't fully loaded
                remaining = words.pop()
                
                for word in words:
                    if word: yield word
                    word_counter += 1
                    if word_counter >= words_limit and words_limit > 0:
                        break
                if word_counter >= words_limit and words_limit > 0:
                    break
        if infinite == False:
            break
                
# Returns word dics: word_to_ind and ind_to_word
def get_dics(file_path: str = extracted_path):
    global vocab_size, dataset_size
    text8_gen = get_text8_generator(file_path=file_path, infinite=False)
    
    word_to_ind = dict()
    ind_to_word = dict()
    curr_word_index: int = 0
    
    vocab_size = 0
    dataset_size = 0
    
    for word in text8_gen:
        dataset_size += 1
        if word not in word_to_ind:
            vocab_size += 1
            
            word_to_ind[word] = curr_word_index
            ind_to_word[curr_word_index] = word
            
            curr_word_index += 1
    return (word_to_ind, ind_to_word)

def get_dataset_size(file_path: str = extracted_path) -> int:
    res: int = 0
    text8_gen = get_text8_generator(file_path=file_path, infinite=False)
    
    for word in text8_gen:
        res += 1
        
    return res

def get_vocab_size(file_path: str = extracted_path) -> int:
    res: int = 0
    text8_gen = get_text8_generator(file_path=file_path, infinite=False)
    
    set_of_words = set()
    for word in text8_gen:
        if word not in set_of_words:
            set_of_words.add(word)
            res += 1
    return res
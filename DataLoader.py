import os
import urllib.request
import zipfile

extracted_path: str = 'text8'

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
def get_text8_generator(file_path: str = extracted_path, chunk_size = 1024 * 1024):
    download_text8_data()
    remaining = ""
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
                
# Returns word dics: word_to_ind and ind_to_word
def get_dics(file_path: str = extracted_path):
    text8_gen = get_text8_generator(file_path=file_path)
    
    word_to_ind = dict()
    ind_to_word = dict()
    curr_word_index: int = 0
    
    for word in text8_gen:
        if word not in word_to_ind:
            word_to_ind[word] = curr_word_index
            ind_to_word[curr_word_index] = word
            
            curr_word_index += 1
    return (word_to_ind, ind_to_word)

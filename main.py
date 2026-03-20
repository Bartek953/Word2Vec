import Trainer
from DataDownloader import data_downloader
import Architecture
import numpy as np

model = Architecture.Model()
Trainer.train(model)

model.save('data.npz')
#model.load('data.npz')

word_to_ind, ind_to_word = data_downloader.get_dics()
        
def find_closest(target: str, n: int = 7) -> [str]:
    if target not in word_to_ind:
        print("Not in dictionary")
        return
    target_vec = model.get_word_embeding(target)
    target_idx = word_to_ind[target]
    
    all_vecs = model.hidden_layer.weights
    
    norm_target = np.linalg.norm(target_vec)
    norm_all = np.linalg.norm(all_vecs, axis=1)
    
    similarities = np.dot(all_vecs, target_vec) / (norm_all * norm_target + 1e-9)
    
    # to discard UNKNOWN token
    similarities[0] = -np.inf
    similarities[target_idx] = -np.inf
    
    # Sort and pick best options
    closest_indices = np.argsort(similarities)[-n:][::-1]
    
    return [ind_to_word[idx] for idx in closest_indices][:n]


test_words = ["king", "queen", "french", "revolution", "computer", "science", "modern"]
for word in test_words:
    print(f"{word.upper()}:")
    print(find_closest(word))
    print()
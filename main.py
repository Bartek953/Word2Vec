import Trainer
from DataDownloader import data_downloader
import numpy as np

word_to_ind, ind_to_word = data_downloader.get_dics()
        
def find_closest(target: str, n: int = 7):
    if target not in word_to_ind:
        print("Not in dictionary")
        return
    target_vec = Trainer.model.get_word_embeding(target)
    target_idx = word_to_ind[target]
    
    all_vecs = Trainer.model.hidden_layer.weights
    
    norm_target = np.linalg.norm(target_vec)
    norm_all = np.linalg.norm(all_vecs, axis=1)
    
    similarities = np.dot(all_vecs, target_vec) / (norm_all * norm_target + 1e-9)
    
    # to discard UNKNOWN token
    similarities[0] = -np.inf
    similarities[target_idx] = -np.inf
    
    # Posortuj i wybierz najlepsze (pomijając samo słowo 'target')
    closest_indices = np.argsort(similarities)[-n:][::-1]
    
    return [(ind_to_word[idx], similarities[idx]) for idx in closest_indices][:n]

print("KING:")
print(find_closest("king"))
print()

print("FRENCH:")
print(find_closest("french"))
print()

print("REVOLUTION:")
print(find_closest("revolution"))
print()

print("MODERN:")
print(find_closest("modern"))
print()

print("individualist:")
print(find_closest("individualist"))
print()


print("individualist:")
print(find_closest("individualist"))
print()

print("paris:")
print(find_closest("paris"))
print()

print("queen:")
print(find_closest("queen"))
print()

print("computer:")
print(find_closest("computer"))
print()

print("science:")
print(find_closest("science"))
print()
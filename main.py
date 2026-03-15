import Trainer
import DataDownloader
import numpy as np

word_to_ind, ind_to_word = DataDownloader.get_dics()

def get_embeding(word: str):
    if word in word_to_ind:
        index = word_to_ind[word]
        
        return Trainer.model.get_embeding(index)
    else:
        print(f"Error: word {word} no in dictionary!")
        
def find_closest(target: str, n: int = 5):
    if target not in word_to_ind:
        print("Not in dictionary")
        return
    target_vec = get_embeding(target)
    
    all_vecs = Trainer.model.hidden_layer.weights 
    
    norm_target = np.linalg.norm(target_vec)
    norm_all = np.linalg.norm(all_vecs, axis=1)
    
    similarities = np.dot(all_vecs, target_vec) / (norm_all * norm_target + 1e-9)
    
    # Posortuj i wybierz najlepsze (pomijając samo słowo 'target')
    closest_indices = np.argsort(similarities)[-(n+1):-1][::-1]
    
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

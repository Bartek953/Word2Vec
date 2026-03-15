import numpy as np
from DataDownloader import data_downloader

def sigmoid(batch_embed: np.ndarray) -> np.ndarray:
    #batch_embed: (batch_size, context_size)
    return 1 / (1 + np.exp(-np.clip(batch_embed, -15, 15)))

def cross_entropy(labels, predictions):
    epsilon = 1e-12
    predictions = np.clip(predictions, epsilon, 1.0 - epsilon)
    
    loss = labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
    
    return -np.mean(loss)

class HiddenLayer:
    def __init__(self, vocab_size: int, embed_size: int, lr: float):
        # Weights: (vocab_size, embed_size)
        # Halves so it doesn't explode during exponentiation
        self.weights = np.random.uniform(-0.5, 0.5, (vocab_size, embed_size)) / embed_size
        self.lr = lr
        
    # Takes in batch of arrays of indexes correspoding to given words
    def forward(self, batch_indices: np.ndarray) -> np.ndarray:
        # batch_indices: (batch_size, context_size)
        
        embeded_words = self.weights[batch_indices]
        # embeded_words: (batch_size, context_size, embed_size)
        
        self.last_indices = batch_indices
        #(batch_size, context_size)
        
        return np.mean(embeded_words, axis = 1)
        # (batch_size, embed_size)
        
    def backpropagate(self, prev_grad: np.ndarray):
        #prev_grad: (batch_size, embed_size)
        
        context_size = self.last_indices.shape[1]
        
        grad_per_word = prev_grad / context_size
        
        # for i, indices in enumerate(self.last_indices):
        #     self.weights[indices] -= self.lr * grad_per_word[i]
        grad_broadcasted = grad_per_word[:, np.newaxis, :] # Kształt: (batch, 1, embed)
        np.add.at(self.weights, self.last_indices, -self.lr * grad_broadcasted)
        

class OutputLayer:
    def __init__(self, vocab_size: int, embed_size: int, lr: float):
        # Weights: (vocab_size, embed_size)
        self.weights = np.random.uniform(-0.5, 0.5, (vocab_size, embed_size)) / embed_size
        self.lr = lr
    
    # Takes in batch of embedings from HiddenLayer and word indices to calculate
    def forward(self, batch_embed: np.ndarray, batch_indices: np.ndarray) -> np.ndarray:
        #bacth_embed: (batch_size, embed_size)
        #batch_indices: (batch_size, eval_size)
        
        self.last_indices = batch_indices
        self.last_embed = batch_embed
        
        part_matrix = self.weights[batch_indices]
        # (batch_size, eval_size, embed_size)
        
        output = np.einsum('bkn,bn->bk', part_matrix, batch_embed)
        #(batch_size, eval_size)
        
        return sigmoid(output)
    
    def backpropagate(self, labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
        # labels: (batch_size, eval_size)
        # predictions: (batch_size, eval_size)
        
        # sigmoid and cross entropy derivative
        loss_grad = predictions - labels
        # (batch_size, eval_size)
        
        # loss_grad: (batch, eval, 1) * last_embed: (batch, 1, embed)
        # -> (batch_size, eval_size, embed_size)
        grad_out = loss_grad[:, :, np.newaxis] * self.last_embed[:, np.newaxis, :]
        
        # Gradient for hidden layer
        part_matrix = self.weights[self.last_indices]
        # (batch_size, eval_size, embed_size)
        grad_hidden = np.einsum('bk,bkn->bn', loss_grad, part_matrix)
        # (batch_size embed_size)
        
        # for i, indices in enumerate(self.last_indices):
        #     self.weights[indices] -= self.lr * grad_out[i]
        np.add.at(self.weights, self.last_indices, -self.lr * grad_out)
        
        return grad_hidden
        # (batch_size, embed_size)
        

class Model:
    def __init__(self, vocab_size: int, embed_size: int, eval_size: int, lr: float):
        self.lr = lr
        self.hidden_layer = HiddenLayer(vocab_size, embed_size, lr)
        self.output_layer = OutputLayer(vocab_size, embed_size, lr)
        self.eval_size = eval_size
        self.vocab_size = vocab_size
        self.neg_probs = data_downloader.get_neg_samples_probs()
    
    def forward(self, batch_indices: np.ndarray, target_words: np.ndarray) -> np.ndarray:
        batch_size: int = batch_indices.shape[0]
        
        # creating negative samples
        #self.eval_ind = np.random.randint(1, self.vocab_size - 1, (batch_size, self.eval_size - 1))
        # chance of randomly getting target word is near zero
        self.eval_ind = np.random.choice(
            self.vocab_size, 
            size=(batch_size * (self.eval_size - 1)), 
            p=self.neg_probs
        ).reshape(batch_size, self.eval_size - 1)
        
        # self.eval_ind[self.eval_ind >= target_words[:, np.newaxis]] += 1
        self.eval_ind = np.column_stack([target_words, self.eval_ind])
        
        
        hidden_out = self.hidden_layer.forward(batch_indices)
        
        
        self.predictions = self.output_layer.forward(hidden_out, self.eval_ind)
        
        labels = np.zeros(self.eval_ind.shape)
        labels[:, 0] = 1
        
        return cross_entropy(labels, self.predictions)
    
    def backpropagate(self):
        labels = np.zeros(self.eval_ind.shape)
        labels[:, 0] = 1
        
        out_grad = self.output_layer.backpropagate(labels, self.predictions)
        self.hidden_layer.backpropagate(out_grad)
    
    def get_word_embeding(self, word: str):
        if word not in data_downloader.word_to_ind:
            return self.hidden_layer.weights[0]
        else:
            return self.hidden_layer.weights[data_downloader.word_to_ind[word]]
        
    def get_ith_embeding(self, index: int) -> np.ndarray:
        return self.hidden_layer.weights[index]

    def update_lr(self, new_lr):
        self.lr = new_lr
        self.hidden_layer.lr = new_lr
        self.output_layer.lr = new_lr
        
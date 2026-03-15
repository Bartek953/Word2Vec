import DataDownloader
import DataLoader
import Architecture

# setup
shuffle_size = 4096
batch_size = 128
learning_rate = 0.005
optimizer = 0.005
embed_size = 128
eval_size = 8
epochs = 300
context_size = 3

# objects declarations

data_loader = DataLoader.DataLoader(context_size = context_size)
batch_gen = data_loader.generate_batches(shuffle_size, batch_size)

vocab_size: int = DataDownloader.vocab_size
dataset_size: int = DataDownloader.dataset_size

model = Architecture.Model(vocab_size, embed_size, eval_size, learning_rate)

n_batches: int = dataset_size // batch_size

# training loop
print(f"Starting training, batches: {n_batches}, epochs {epochs}, lr {learning_rate}")
for e in range(epochs):
    loss = 0
    for i in range(n_batches):
        batch_x, batch_y = next(batch_gen)
        
        loss += model.forward(batch_x, batch_y) / n_batches
        
        model.backpropagate()
    
    model.update_lr(learning_rate / (1 + e * optimizer))
        
    print(f"Epoch: {e}, Loss: {loss}, LR: {model.lr}")
        
        

from DataDownloader import data_downloader
import DataLoader
import Architecture
import Config

# setup
shuffle_size = Config.shuffle_size
batch_size = Config.batch_size
learning_rate = Config.learning_rate
optimizer = Config.optimizer
embed_size = Config.embed_size
eval_size = Config.eval_size
epochs = Config.epochs
context_size = Config.context_size

# objects declarations

data_loader = DataLoader.DataLoader(context_size = context_size)
batch_gen = data_loader.generate_batches(shuffle_size, batch_size)

vocab_size: int = data_downloader.get_vocab_size()
dataset_size: int = data_downloader.get_dataset_size()

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
        
        

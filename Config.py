# Project config


# Data Download config
file_path = "text8"
dataset_url = "http://mattmahoney.net/dc/text8.zip"

# set to -1 to disable limit
text_len_lim = 1000000
ignore_ratio = 4

# Trainer
shuffle_size = 4096
batch_size = 128
learning_rate = 0.005
optimizer = 0.005
embed_size = 128
eval_size = 8
epochs = 100
context_size = 3
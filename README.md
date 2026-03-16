Word2Vec implementation in numpy

Overview:
This implementation uses CBOW (continous bag of words - predicting middle word based on surrouding ones) architecture. To make it faster it uses sparse matrix multiplication and negative sampling (randomly choosing a few outputs in opposite to calculate answers for whole vocabulary). The negative sampling probability distribution is word_frequency^(3/4), as in original paper. For training it utilizes text8 dataset, however it can be easily changed to other one - data are loaded using generators, to not load whole dataset into RAM. Training igores words that appear very rarely.

Project consits of:
- DataDownloader - loads dataset, creates dictionaries and generators
- DataLoader - creates batch generator
- Architecture - implemts actual model architecture providing "Model" class
- Trainer - training loop
- Config - contains model hyperparameters and download config
- main - entry point, allows for some experiments/tests

Usage:
Modify config if needed. You can add/change test to main.py.
Run: pip install numpy
Run: python main.py
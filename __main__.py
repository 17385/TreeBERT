import argparse
import copy

import torch
from torch.utils.data import DataLoader

from dataset import BPE, TokenVocab, TreeBERTDataset
from model import Decoder, Encoder, Seq2Seq
from trainer import BERTTrainer


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("-td", "--train_dataset", type=str, required=True, help="train set")
    parser.add_argument("-vd", "--valid_dataset", type=str, default=None, help="validation set")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="vocab path")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="model save path")

    parser.add_argument("-fs", "--feed_forward_hidden", type=int, default=4096, help="hidden size of feed-forward network")
    parser.add_argument("-hs", "--hidden", type=int, default=1024, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=6, help="number of transformer layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-p", "--path_num", type=int, default=100, help="a AST's maximum path num")
    parser.add_argument("-n", "--node_num", type=int, default=20, help="a path's maximum node num")
    parser.add_argument("-c", "--code_len", type=int, default=200, help="maximum code len")

    parser.add_argument("-al", "--alpha", type=int, default=0.75, help="loss weight")
    parser.add_argument("-b", "--batch_size", type=int, default=4096, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker num")

    parser.add_argument("--with_cuda", type=bool, default=False, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = TokenVocab.load_vocab(args.vocab_path)
    # source and target corpus share the vocab
    print("Vocab Size: ", len(vocab))
 
    print("Loading Train Dataset")
    train_dataset = TreeBERTDataset(vocab, args.train_dataset, path_num=args.path_num, node_num=args.node_num, 
                                    code_len=args.code_len, is_fine_tune=False, corpus_lines=args.corpus_lines)

    print("Loading valid Dataset")
    valid_dataset = TreeBERTDataset(vocab, args.valid_dataset, path_num=args.path_num, node_num=args.node_num, 
                                    code_len=args.code_len, is_fine_tune=False, corpus_lines=args.corpus_lines) \
        if args.valid_dataset is not None else None

    # Creating Dataloader
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    valid_data_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if valid_dataset is not None else None

    print("Building model")
    dropout = 0.1
    enc= Encoder(len(vocab), args.node_num, args.hidden, args.layers, args.attn_heads, args.feed_forward_hidden, 
                                                    dropout,  max_length = args.path_num)
    dec= Decoder(len(vocab), args.hidden, args.layers, args.attn_heads, args.feed_forward_hidden, 
                                                    dropout,  max_length = args.code_len+2)
    
    PAD_IDX = vocab.pad_index
    transformer = Seq2Seq(enc, dec, args.hidden, PAD_IDX)

    print("Creating Trainer")
    trainer = BERTTrainer(transformer, args.alpha, len(vocab), train_dataloader=train_data_loader, test_dataloader=valid_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    min_loss = 10
    loss = 0
    best_model = None
    for epoch in range(args.epochs):
        trainer.train(epoch) 

        if valid_data_loader is not None:
            loss = trainer.test(epoch)
        
        if min_loss>loss:
            best_model = copy.deepcopy(trainer.transformer)

    trainer.save(epoch, best_model, args.output_path)
    
train()

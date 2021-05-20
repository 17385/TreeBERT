import argparse

import torch 
from torch.utils.data import DataLoader

from finetuneTrainer import Trainer
import sys 
from dataset import TreeBERTDataset, TokenVocab, BPE


def train():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_save_path", required=True, type=str, help="save processed dataset")
    parser.add_argument("-s", "--source_train_dataset", type=str, default=None, help="train source set for evaluate train set")
    parser.add_argument("-t", "--target_train_dataset", type=str, default=None, help="teain target set for evaluate train set")
    parser.add_argument("-sd", "--source_test_dataset", type=str, default=None, help="test source set for evaluate test set")
    parser.add_argument("-td", "--target_test_dataset", type=str, default=None, help="test target set for evaluate test set")
    parser.add_argument("-m", "--pretrain_model_path", required=True, type=str, help="pre-train model load path")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="vocab path")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="model save path")

    parser.add_argument("-fs", "--feed_forward_hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-p", "--path_num", type=int, default=100, help="AST's maximum path num")
    parser.add_argument("-n", "--node_num", type=int, default=20, help="A path's maximum node num")
    parser.add_argument("-c", "--code_len", type=int, default=200, help="maximum code len")

    parser.add_argument("-b", "--batch_size", type=int, default=2, help="number of batch_size")
    parser.add_argument("-e", "--epochs", type=int, default=5, help="number of epochs")
    parser.add_argument("-w", "--num_workers", type=int, default=0, help="dataloader worker size")

    parser.add_argument("--with_cuda", type=bool, default=False, help="training with CUDA: true, or false")
    parser.add_argument("--log_freq", type=int, default=10, help="printing loss every n iter: setting n")
    parser.add_argument("--corpus_lines", type=int, default=None, help="total number of lines in corpus")
    parser.add_argument("--cuda_devices", type=int, nargs='+', default=None, help="CUDA device ids")

    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate of adam")
    parser.add_argument("--adam_weight_decay", type=float, default=0.01, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="adam first beta value")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="adam first beta value")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = TokenVocab.load_vocab(args.vocab_path)
    print("Vocab Size: ", len(vocab))

    print("Loading Train Dataset")
    train_dataset = TreeBERTDataset(vocab, path_num=args.path_num, node_num=args.node_num, code_len=args.code_len, 
                                dataset_save_path=args.dataset_save_path, is_dataset_processed=True, is_fine_tune=True,
                                test_source_path=args.source_test_dataset, test_target_path=args.target_test_dataset)
        
    print("Loading Test Dataset")
    test_dataset = TreeBERTDataset(vocab, path_num=args.path_num, node_num=args.node_num, code_len=args.code_len, 
                                dataset_save_path=args.dataset_save_path, is_dataset_processed=True, is_fine_tune=True,
                                test_source_path=args.source_test_dataset, test_target_path=args.target_test_dataset, is_test=True) \
        if args.source_test_dataset is not None else None


    print("Creating Dataloader")
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
    test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
        if test_dataset is not None else None

    print("Load TreeBERT")
    device = torch.device("cuda:0" if args.with_cuda else "cpu")
    model = torch.load(args.pretrain_model_path, map_location=device)

    print("Creating Trainer")
    trainer = Trainer(model, args.path_num, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                          lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                          with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq)

    print("Training Start")
    for epoch in range(args.epochs):
        trainer.train(epoch)
        trainer.save(epoch, args.output_path)

        if test_data_loader is not None:
            trainer.test(epoch)

train()
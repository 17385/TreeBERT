import argparse

import torch 
from torch.utils.data import DataLoader

import sys 
from dataset import TreeBERTDataset, TokenVocab, BPE, utils


def inference():
    parser = argparse.ArgumentParser()

    parser.add_argument("-d", "--dataset_save_path", required=True, type=str, help="save processed dataset")
    parser.add_argument("-s", "--inference_source_data", type=str, required=True, help="inference source data path")
    parser.add_argument("-t", "--inference_target_data", type=str, required=True, help="inference target data path")
    parser.add_argument("-v", "--vocab_path", required=True, type=str, help="vocab path")
    parser.add_argument("-o", "--output_path", required=True, type=str, help="result save path")
    parser.add_argument("-m", "--finetune_model_path", required=True, type=str, help="fine-tune model load path")

    parser.add_argument("-fs", "--feed_forward_hidden", type=int, default=512, help="hidden size of transformer model")
    parser.add_argument("-hs", "--hidden", type=int, default=256, help="hidden size of transformer model")
    parser.add_argument("-l", "--layers", type=int, default=8, help="number of layers")
    parser.add_argument("-a", "--attn_heads", type=int, default=8, help="number of attention heads")
    parser.add_argument("-p", "--path_num", type=int, default=100, help="AST's maximum path num")
    parser.add_argument("-n", "--node_num", type=int, default=20, help="A path's maximum node num")
    parser.add_argument("-c", "--code_len", type=int, default=200, help="maximum code len")

    parser.add_argument("--with_cuda", type=bool, default=False, help="training with CUDA: true, or false")

    args = parser.parse_args()

    print("Loading Vocab", args.vocab_path)
    vocab = TokenVocab.load_vocab(args.vocab_path)

    print("Loading Train Dataset")
    dataset = TreeBERTDataset(vocab, path_num=args.path_num, node_num=args.node_num, code_len=args.code_len, 
                                dataset_save_path=args.dataset_save_path, is_dataset_processed=True, is_fine_tune=True,
                                test_source_path=args.inference_source_data, test_target_path=args.inference_target_data)

    print("Loading TreeBERT")
    device = torch.device("cuda:0" if args.with_cuda else "cpu")
    model = torch.load(args.finetune_model_path, map_location=device)
    
    bleu_score = utils.calculate_bleu(dataset, model, vocab, device=device, max_len=args.code_len)

    print("bleu score:",bleu_score)
    
    




inference()
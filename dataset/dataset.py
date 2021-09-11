import json
import os

import torch
from torch.utils.data import Dataset

from dataset import BPE


class TreeBERTDataset(Dataset):
    def __init__(self, vocab, corpus_path, path_num, node_num, code_len, 
                        is_fine_tune=False, 
                        corpus_lines=None, max_subtoken_len=3):

        self.corpus_lines = corpus_lines
        self.corpus_path = corpus_path
        self.path_num = path_num
        self.code_len = code_len
        self.node_num = node_num
        self.vocab = vocab
        self.max_subtoken_len = max_subtoken_len

        self.files = os.listdir(corpus_path)
        if self.corpus_lines is None:
            self.corpus_lines = 0
        for index, tmp_file in enumerate(self.files):
            with open(corpus_path+tmp_file, 'r', encoding='utf-8') as f:
                for _ in f.readlines():
                    self.corpus_lines += 1

        # Start the first file
        self.file_index = 0
        self.file = open(corpus_path+self.files[self.file_index], 'r', encoding='utf-8')
                
    def __len__(self):
        return self.corpus_lines

    def __getitem__(self, item):
        line = self.get_corpus_line()
        data = json.loads(line)
        lan_type = data['lan_type']
        AST = data['encoder_input'][:self.path_num]
        code_mask = data['decoder_input'][:self.code_len]
        code = data['decoder_output'][:self.code_len]
        coeff = data['node_pos_em_coeff'][:self.path_num]

        padding_token = [self.vocab.pad_index for _ in range(self.max_subtoken_len)]
        padding_path = [padding_token for _ in range(self.node_num)]
        padding_coeff = [0 for _ in range(self.node_num)]

        for index, path in enumerate(AST):
            path = BPE.encode(self.vocab.vocab_tokenization,self.vocab.tokenize_word,self.vocab.sorted_tokens,texts=path[:self.node_num])
            for i, token in enumerate(path):
                for j, subtoken in enumerate(token):
                    path[i][j] = self.vocab.stoi.get(subtoken, self.vocab.unk_index)

            # Padding to the same number of nodes per path
            if(len(path)<self.node_num):
                padding1 = [padding_token for _ in range(self.node_num - len(path))]
                padding2 = [0 for _ in range(self.node_num - len(coeff[index]))]
                path.extend(padding1)
                coeff[index].extend(padding2)
                
            AST[index] = path                   
        
        # Padding to the same number of paths per AST
        if(len(AST)<self.path_num):
            padding1 = [padding_path for _ in range(self.path_num-len(AST))]
            padding2 = [padding_coeff for _ in range(self.path_num-len(AST))]
            AST.extend(padding1)
            coeff.extend(padding2)

            
        for i, token in enumerate(code):
            code[i] = self.vocab.stoi.get(token.lower() + '</w>', self.vocab.unk_index)
            code_mask[i] = self.vocab.stoi.get(code_mask[i].lower() + '</w>', self.vocab.unk_index)

        # Padding to the same length of code
        padding = [self.vocab.pad_index for _ in range(self.code_len - len(code))]
        code_mask.extend(padding)
        code.extend(padding)

        # add <sos> and <eos>
        if(lan_type=="py"):
            sos_token = self.vocab.sos_index_python
        elif(lan_type=="java"):
            sos_token = self.vocab.sos_index_java
        else:
            sos_token = self.vocab.unk_index

        code_mask = [sos_token] + code_mask + [self.vocab.cls_index]
        code = code + [self.vocab.cls_index] + [self.vocab.eos_index]

        output = {"encoder_input": AST,
                "decoder_input":  code_mask,
                "decoder_output": code,
                "is_ast_order": data['is_ast_order'],
                "node_pos_em_coeff": coeff}
        
        return {key: torch.tensor(value) for key, value in output.items()}

    def get_corpus_line(self):
        line = self.file.readline()

        # start reading the next file
        if line is '':
            self.file.close()
            self.file_index = (self.file_index + 1) % len(self.files)
            self.file = open(self.corpus_path+self.files[self.file_index], "r", encoding='utf-8')
            line = self.file.readline()
        return line

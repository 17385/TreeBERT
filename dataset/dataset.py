from torch.utils.data import Dataset
import numpy as np
import torch
import random
import json
import copy
import tqdm
from dataset import BPE

class TreeBERTDataset(Dataset):
    def __init__(self, vocab, path_num, node_num, code_len, dataset_save_path, is_dataset_processed=True, 
                         test_source_path=None, test_target_path=None, corpus_lines=None, is_test=False,is_fine_tune=False):
        self.vocab = vocab
        self.path_num = path_num
        self.node_num = node_num
        self.code_len = code_len
        self.is_fine_tune = is_fine_tune
        self.encoder_input = []
        self.decoder_input = []
        self.label = []
        self.is_path_order = []
        ASTs = []
        codes = []

        if(is_test):
            dataset_save_path = dataset_save_path + "/TestDataset.json"
        else:
            dataset_save_path = dataset_save_path + "/TrainDataset.json"

        if(is_dataset_processed == False):

            if(test_source_path != None and test_target_path != None):
                with open(test_source_path, "r", encoding='utf-8') as f:  
                    for paths in tqdm.tqdm(f, desc="Loading Source Dataset"):
                        path = paths.split("\t")
                        path_list = []
                        for nodes in path:
                            node_list = []
                            for tmp in nodes.replace("|", " ").replace("\/?", "").replace("/", " / ").split():
                                tmp = tmp + '</w>'
                                node_list.append(tmp)
                            node_list = BPE.encode(vocab.vocab_tokenization,vocab.tokenize_word,vocab.sorted_tokens,texts=node_list)
                            if(len(node_list) > 2):
                                path_list.append(node_list)
                        ASTs.append(path_list)

                with open(test_target_path, "r", encoding='utf-8') as f:
                    for code in tqdm.tqdm(f, desc="Loading Dataset"):
                        tmp_tokens = []
                        for tmp in code.split():
                            tmp = tmp + '</w>'
                            tmp_tokens.append(tmp)

                        code = BPE.encode(vocab.vocab_tokenization,vocab.tokenize_word,vocab.sorted_tokens,texts=tmp_tokens)
                        codes.append(code)
            else:
                ASTs = vocab.ASTs
                codes = vocab.codes

            # Temporary save to file
            with open(dataset_save_path, "w", encoding="utf-8") as f:
                for index, AST in enumerate(ASTs):
                    t, is_path_order = self.change_node(ASTs[index])
                    AST, code, token_list = self.PMLM(t, codes[index])
                    self.encoder_input.append(AST)
                    self.decoder_input.append(code)
                    self.label.append(token_list)
                    self.is_path_order.append(is_path_order)
                    
                    output = {"encoder_input": AST,
                    "decoder_input":  code,
                    "label": token_list,
                    "is_path_order": is_path_order}
                    f.write(json.dumps(output))
                    f.write("\n")

        else:
            # If the dataset has already been processed, read the file directly.
            with open(dataset_save_path, "r", encoding="utf-8") as f:
                for data in tqdm.tqdm(f, desc="Loading Processed Dataset", total=corpus_lines):
                    data = json.loads(data)
                    self.encoder_input.append(data["encoder_input"])
                    self.decoder_input.append(data["decoder_input"])
                    self.label.append(data["label"])
                    self.is_path_order.append(data["is_path_order"])
        

        self.source_corpus_lines = len(self.encoder_input)
        self.target_corpus_lines = len(self.decoder_input)              

    def __len__(self):
        return self.target_corpus_lines

    def __getitem__(self, item):
        code = [self.vocab.sos_index] + self.decoder_input[item] + [self.vocab.eos_index]
        token_list = [self.vocab.sos_index] + self.label[item] + [self.vocab.eos_index]

        # padding 
        AST = self.encoder_input[item][:self.path_num]
        if(len(AST) < self.path_num):
            tmp = [self.vocab.pad_index for _ in range(self.node_num)]
            padding = [tmp for _ in range(self.path_num - len(AST))]
            AST.extend(padding)
        # padding number of AST Paths
        for index, path in enumerate(AST):
            if(len(path) >= self.node_num):
                AST[index] = path[:self.node_num]
            else:
                padding = [self.vocab.pad_index for _ in range(self.node_num - len(path))]
                AST[index].extend(padding)

        # padding length of decoder inputs and outputs
        code = code[:self.code_len]
        if(len(code) < self.code_len):
            padding = [self.vocab.pad_index for _ in range(self.code_len - len(code))]
            code.extend(padding)

        token_list = token_list[:self.code_len]
        if(len(token_list) < self.code_len):
            padding = [self.vocab.pad_index for _ in range(self.code_len - len(token_list))]
            token_list.extend(padding)

        if(self.is_fine_tune == False):
            output = {"encoder_input": AST,
                    "decoder_input":  code,
                    "label": token_list,
                    "is_path_order": self.is_path_order[item]}
        else:
            output = {"encoder_input": AST,
                  "label": token_list}
                  
        return {key: torch.tensor(value) for key, value in output.items()}

    def PMLM(self, AST, code):
        mask_node = []
        if(self.is_fine_tune == False):
            # AST path disrupted, up to mask 0.15 nodes
            max_mask_num = len(AST) * len(AST[0]) * 0.15
            random.shuffle(AST)
            for path in AST:
                d = len(path)
                tmp = np.array(range(d))
                select_node_prob_dis = np.exp(tmp-d) / np.sum(np.exp(tmp-d))

                for index, node in enumerate(path):
                    prob = random.random()
                    if (prob < select_node_prob_dis[index]) and (len(mask_node) < max_mask_num):
                        mask_node.append(path[index])
                        path[index] = "<mask>"
     
        # Decoder Inputs and Outputs
        token_list = []
        for m, token in enumerate(code):
            token_list.append(self.vocab.stoi.get(token, self.vocab.unk_index))
            if(self.is_fine_tune == False):
                if (token not in mask_node):
                    code[m] = self.vocab.mask_index
                else:
                    code[m] = self.vocab.stoi.get(token, self.vocab.unk_index)

        # Encoder Inputs
        for m, path in enumerate(AST):
            for n, node in enumerate(path):
                AST[m][n] = self.vocab.stoi.get(node, self.vocab.unk_index)
            
        return AST, code, token_list

    def change_node(self, AST):
        t = copy.deepcopy(AST) 

        # output_text, label(disorder:0, order:1)
        # AST has 0.5 probability of exchanging nodes
        if(self.is_fine_tune == False):
            if random.random() > 0.5:
                    return t, 1
            else:
                # Randomly select a path
                path = t[random.randint(0, len(t)-1)]
                path = t[random.randint(0, len(t)-1)]
                position = random.sample(range(0,len(path)-1), 2)
                path[position[0]], path[position[1]] = path[position[1]], path[position[0]]
                return t, 0
        return t, 0
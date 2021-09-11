import sys

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from .optim_schedule import ScheduledOptim

sys.path.append("..") 
import tqdm
from model import TreeBERT, transformer


class BERTTrainer:
    def __init__(self, transformer, alpha, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):

        self.alpha = alpha
        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.transformer = transformer
        self.model = TreeBERT(self.transformer, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for TreeBERT" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.transformer.hidden, n_warmup_steps=warmup_steps)

        self.criterionNOP = nn.BCEWithLogitsLoss()
        self.criterionTMLM = nn.NLLLoss(ignore_index=-2)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        loss = self.iteration(epoch, self.train_data)
        return loss

    def test(self, epoch):
        loss = self.iteration(epoch, self.test_data, train=False)
        return loss

    def iteration(self, epoch, data_loader, train=True):
        str_code = "train" if train else "test"

        data_iter = tqdm.tqdm(enumerate(data_loader),
                              desc="EP_%s:%d" % (str_code, epoch),
                              total=len(data_loader),
                              bar_format="{l_bar}{r_bar}")

        avg_loss = 0.0
        total_correct = 0
        total_element = 0

        for i, data in data_iter:
            data = {key: value.to(self.device) for key, value in data.items()}

            node_order_predic, tree_mask_predic = self.model.forward(data["encoder_input"], data["node_pos_em_coeff"], data["decoder_input"])
            NOP_loss = self.criterionNOP(node_order_predic.squeeze(dim=-1), data["is_ast_order"].float())
            tree_mask_predic = tree_mask_predic.reshape(-1,tree_mask_predic.shape[-1])
            target = data["decoder_output"].reshape(-1)
            TMLM_loss = self.criterionTMLM(tree_mask_predic, target)
            
            loss = (1-self.alpha) * NOP_loss + self.alpha * TMLM_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            NOP_prediction = (node_order_predic > 0.5).squeeze(-1)
            correct = NOP_prediction.eq(data["is_ast_order"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_ast_order"].nelement()

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "NOP_total_acc=",
              total_correct * 100.0 / total_element)

        return avg_loss / len(data_iter)

    def save(self, epoch, model, file_path="output/bert_trained.model"):
        output_path = file_path + ".ep%d" % epoch
        torch.save(model, output_path)
        model.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

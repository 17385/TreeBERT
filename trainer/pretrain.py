import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from .optim_schedule import ScheduledOptim
import sys
sys.path.append("..") 
from model import TreeBERT, transformer


import tqdm


class BERTTrainer:
    def __init__(self, transformer, path_num, vocab_size: int,
                 train_dataloader: DataLoader, test_dataloader: DataLoader = None,
                 lr: float = 1e-4, betas=(0.9, 0.999), weight_decay: float = 0.01, warmup_steps=10000,
                 with_cuda: bool = True, cuda_devices=None, log_freq: int = 10):
        """
        :param transformer: transformer model which you want to train
        :param vocab_size: total word vocab size
        :param train_dataloader: train dataset data loader
        :param test_dataloader: test dataset data loader [can be None]
        :param lr: learning rate of optimizer
        :param betas: Adam optimizer betas
        :param weight_decay: Adam optimizer weight decay param
        :param with_cuda: traning with cuda
        :param log_freq: logging frequency of the batch iteration
        """

        cuda_condition = torch.cuda.is_available() and with_cuda
        self.device = torch.device("cuda:0" if cuda_condition else "cpu")

        self.transformer = transformer
        self.model = TreeBERT(self.transformer, path_num, vocab_size).to(self.device)

        # Distributed GPU training if CUDA can detect more than 1 GPU
        if with_cuda and torch.cuda.device_count() > 1:
            print("Using %d GPUS for transformer" % torch.cuda.device_count())
            self.model = nn.DataParallel(self.model, device_ids=cuda_devices)

        self.train_data = train_dataloader
        self.test_data = test_dataloader

        # Setting the Adam optimizer with hyper-param
        self.optim = Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        self.optim_schedule = ScheduledOptim(self.optim, self.transformer.hidden, n_warmup_steps=warmup_steps)

        self.criterion = nn.NLLLoss(ignore_index=0)

        self.log_freq = log_freq

        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))

    def train(self, epoch):
        self.iteration(epoch, self.train_data)

    def test(self, epoch):
        self.iteration(epoch, self.test_data, train=False)

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
            # batch_data will be sent into the device(GPU or cpu)
            data = {key: value.to(self.device) for key, value in data.items()}

            next_sent_output, mask_lm_output = self.model.forward(data["encoder_input"], data["decoder_input"][:,:-1])
            next_loss = self.criterion(next_sent_output, data["is_path_order"])
            mask_loss = self.criterion(mask_lm_output.transpose(1, 2), data["label"][:,1:])
            loss = next_loss + mask_loss

            if train:
                self.optim_schedule.zero_grad()
                loss.backward()
                self.optim_schedule.step_and_update_lr()

            correct = next_sent_output.argmax(dim=-1).eq(data["is_path_order"]).sum().item()
            avg_loss += loss.item()
            total_correct += correct
            total_element += data["is_path_order"].nelement()

            post_fix = {
                "epoch": epoch,
                "iter": i,
                "avg_loss": avg_loss / (i + 1),
                "avg_acc": total_correct / total_element * 100,
                "loss": loss.item()
            }

            if i % self.log_freq == 0:
                data_iter.write(str(post_fix))

        print("EP%d_%s, avg_loss=" % (epoch, str_code), avg_loss / len(data_iter), "total_acc=",
              total_correct * 100.0 / total_element)

    def save(self, epoch, file_path="output/bert_trained.model"):
        """
        Saving the current transformer model on file_path

        :param epoch: current epoch number
        :param file_path: model output path which gonna be file_path+"ep%d" % epoch
        :return: final_output_path
        """
        output_path = file_path + ".ep%d" % epoch
        torch.save(self.transformer, output_path)
        self.transformer.to(self.device)
        print("EP:%d Model Saved on:" % epoch, output_path)
        return output_path

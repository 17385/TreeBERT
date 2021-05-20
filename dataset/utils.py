import torch
import torch.nn as nn

from torchtext.data.metrics import bleu_score

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import tqdm


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)


def train(model, iterator, optimizer, criterion, clip):

    model.train()
    
    epoch_loss = 0
    
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        try:
            output, _ = model(src, trg[:,:-1])
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                raise exception
  
        output_dim = output.shape[-1]
            
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:,1:].contiguous().view(-1)
    
        loss = criterion(output, trg)
        
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        optimizer.step()
        
        epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    
    model.eval()
    
    epoch_loss = 0
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg

            output, _ = model(src, trg[:,:-1])
 
            output_dim = output.shape[-1]
            
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:,1:].contiguous().view(-1)

            loss = criterion(output, trg)

            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def generate_target(src, model, vocab, device, max_len = 50):
    
    model.eval()
    
    src_tensor = src.unsqueeze(0).to(device)
    
    src_mask = model.make_src_mask(src_tensor)
    
    with torch.no_grad():
        enc_src = model.encoder(src.unsqueeze(0), src_mask)

    trg_indexes = [vocab.sos_index]

    for i in range(max_len):

        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(0).to(device)

        trg_mask = model.make_trg_mask(trg_tensor)
        
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        
        pred_token = output.argmax(2)[:,-1].item()
        
        trg_indexes.append(pred_token)

        if pred_token == vocab.eos_index:
            break
    
    trg_tokens = [vocab.itos[i] for i in trg_indexes]
    trg_tokens = "".join(trg_tokens[1:])
    trg_tokens = trg_tokens.split("</w>")
    
    return trg_tokens, attention

def display_attention(sentence, translation, attention, n_heads = 8, n_rows = 4, n_cols = 2):
    
    assert n_rows * n_cols == n_heads
    
    fig = plt.figure(figsize=(15,25))
    
    for i in range(n_heads):
        
        ax = fig.add_subplot(n_rows, n_cols, i+1)
        
        _attention = attention.squeeze(0)[i].cpu().detach().numpy()

        cax = ax.matshow(_attention, cmap='bone')

        ax.tick_params(labelsize=12)
        ax.set_xticklabels(['']+['<sos>']+[t.lower() for t in sentence]+['<eos>'], 
                           rotation=45)
        ax.set_yticklabels(['']+translation)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()
    plt.close()


def calculate_bleu(dataset, model, vocab, device, max_len = 50):
    dataset = tqdm.tqdm(enumerate(dataset),
                              desc="inference and calculate BLEU",
                              total=len(dataset),
                              bar_format="{l_bar}{r_bar}")
    trgs = []
    pred_trgs = []
    for i, data in dataset:
        # data will be sent into the device(GPU or cpu)
        data = {key: value.to(device) for key, value in data.items()}
        
        src = data['encoder_input']
        trg_index = data['label']
        trg = [vocab.itos[i] for i in trg_index]
        trg = "".join(trg[1:])
        trg = trg.split("</w>")

        
        pred_trg, _ = generate_target(src, model, vocab, device, max_len)
        
        #cut off <eos> token       
        pred_trgs.append(pred_trg[:-1])
        trgs.append(trg[:-1])
        
    return bleu_score(pred_trgs, trgs)

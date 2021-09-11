#coding=utf-8

import collections
import json
import os
import pickle
import re

from tqdm import tqdm


class BPE(object):
    def __init__(self, corpus_path, BPE_path="data/BPEObject.small", except_list=None, num_merges=5):
        print("Read in files")
        print('==========')
        vocab = self.get_vocab(corpus_path, except_list)
        # self.tokens_frequencies, self.vocab_tokenization = self.get_tokens_from_vocab(vocab)
        print('==========')
        print("Adopt BPE")
        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            if(i % 100 == 0):
                print('Iter: {}'.format(i))
            # print('Number of tokens: {}'.format(len(self.tokens_frequencies.keys())))
                print('==========')

        self.tokens_frequencies, self.vocab_tokenization = self.get_tokens_from_vocab(vocab)

        sorted_tokens_tuple = sorted(self.tokens_frequencies.items(), key=lambda item: (self.measure_token_length(item[0]), item[1]), reverse=True)
        self.sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]
        
        print('==========')


    @staticmethod
    def encode(vocab_tokenization, tokenize_word, sorted_tokens, texts):
        encode_output = []
        for word_given in texts:
            # Tokenization of the known word
            word_given = word_given.lower() + '</w>'
            if word_given in vocab_tokenization:
                tmpWord = vocab_tokenization[word_given]
            # Tokenizating of the unknown word
            else:
                tmpWord = tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='<unk>')
            encode_output = encode_output + tmpWord
        return encode_output

        
    @staticmethod
    def decode():
        pass
    @staticmethod
    def load_BPE(BPE_path: str):
        with open(BPE_path, "rb") as f:
            return pickle.load(f)

    def save_BPE(self, BPE_path):
        with open(BPE_path, "wb") as f:
            pickle.dump(self, f)

    def get_vocab(self, corpus_path, except_list):
        vocab = collections.Counter()
        files= os.listdir(corpus_path)

        for index, file in enumerate(files):   
            with open(corpus_path+file, 'r', encoding='utf-8') as f:
                s = f.readlines()
                for line in tqdm(s,
                            desc="file %d" % (index),
                            total=len(s),
                            bar_format="{l_bar}{r_bar}"):
                    data = json.loads(line)
                    AST = data['encoder_input']
                    code = data['decoder_output']
                    for path in AST:
                        for node in path:
                            if (node in except_list):
                                vocab[node.lower() + '</w>'] += 1
                            else:
                                vocab[' '.join(list(node.lower())) + ' </w>'] += 1
                    for token in code:
                        vocab[token.lower() + '</w>'] += 1

        return vocab

    def get_stats(self, vocab):
        pairs = collections.Counter()
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols)-1):
                pairs[symbols[i],symbols[i+1]] += freq
        return pairs

    def merge_vocab(self, pair, v_in):
        v_out = {}
        bigram = re.escape(' '.join(pair))
        p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
        for word in v_in:
            w_out = p.sub(''.join(pair), word)
            v_out[w_out] = v_in[word]
        return v_out

    def get_tokens_from_vocab(self, vocab):
        self.tokens_frequencies = collections.Counter()
        self.vocab_tokenization = {}
        for word, freq in vocab.items():
            word_tokens = word.split()
            for token in word_tokens:
                self.tokens_frequencies[token] += freq
            self.vocab_tokenization[''.join(word_tokens)] = word_tokens
        return self.tokens_frequencies, self.vocab_tokenization

    def measure_token_length(self, token):
        if token[-4:] == '</w>':
            return len(token[:-4]) + 1
        else:
            return len(token)

    def tokenize_word(self, string, sorted_tokens, unknown_token='<unk>'):

        if string == '':
            return []
        if sorted_tokens == []:
            return [unknown_token]

        string_tokens = []
        for i in range(len(sorted_tokens)):
            token = sorted_tokens[i]
            token_reg = re.escape(token.replace('.', '[.]'))

            matched_positions = [(m.start(0), m.end(0)) for m in re.finditer(token_reg, string)]
            if len(matched_positions) == 0:
                continue
            substring_end_positions = [matched_position[0] for matched_position in matched_positions]

            substring_start_position = 0
            for substring_end_position in substring_end_positions:
                substring = string[substring_start_position:substring_end_position]
                string_tokens += self.tokenize_word(string=substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
                string_tokens += [token]
                substring_start_position = substring_end_position + len(token)
            remaining_substring = string[substring_start_position:]
            string_tokens += self.tokenize_word(string=remaining_substring, sorted_tokens=sorted_tokens[i+1:], unknown_token=unknown_token)
            break
        return string_tokens

class TorchVocab(object):
    """Defines a vocabulary object that will be used to numericalize a field.
    Attributes:
        freqs: A collections.Counter object holding the frequencies of tokens
            in the data used to build the Vocab.
        stoi: A collections.defaultdict instance mapping token strings to
            numerical identifiers.
        itos: A list of token strings indexed by their numerical identifiers.
    """

    def __init__(self, counter, max_size=None, min_freq=1, specials=['<pad>', '<unk>'],
                 vectors=None, unk_init=None, vectors_cache=None):
        """Create a Vocab object from a collections.Counter.
        Arguments:
            counter: collections.Counter object holding the frequencies of
                each value found in the data.
            max_size: The maximum size of the vocabulary, or None for no
                maximum. Default: None.
            min_freq: The minimum frequency needed to include a token in the
                vocabulary. Values less than 1 will be set to 1. Default: 1.
            specials: The list of special tokens (e.g., padding or eos) that
                will be prepended to the vocabulary in addition to an <unk>
                token. Default: ['<pad>']
            vectors: One of either the available pretrained vectors
                or custom pretrained vectors (see Vocab.load_vectors);
                or a list of aforementioned vectors
            unk_init (callback): by default, initialize out-of-vocabulary word vectors
                to zero vectors; can be any function that takes in a Tensor and
                returns a Tensor of the same size. Default: torch.Tensor.zero_.
            vectors_cache: directory for cached vectors. Default: '.vector_cache'.
        """
        self.freqs = counter
        counter = counter.copy()
        min_freq = max(min_freq, 1)

        self.itos = list(specials)
        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in specials:
            del counter[tok]

        max_size = None if max_size is None else max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)

        # stoi is simply a reverse dict for itos
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}

        self.vectors = None
        if vectors is not None:
            self.load_vectors(vectors, unk_init=unk_init, cache=vectors_cache)
        else:
            assert unk_init is None and vectors_cache is None

    def __eq__(self, other):
        if self.freqs != other.freqs:
            return False
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if self.vectors != other.vectors:
            return False
        return True

    def __len__(self):
        return len(self.itos)

    def vocab_rerank(self):
        self.stoi = {word: i for i, word in enumerate(self.itos)}

    def extend(self, v, sort=False):
        words = sorted(v.itos) if sort else v.itos
        for w in words:
            if w not in self.stoi:
                self.itos.append(w)
                self.stoi[w] = len(self.itos) - 1


class Vocab(TorchVocab):
    def __init__(self, counter, max_size=None, min_freq=1):
        self.pad_index = 0
        self.unk_index = 1
        self.sos_index_python = 2
        self.sos_index_java = 3
        self.eos_index = 4
        self.mask_index = 5
        self.cls_index = 6
        # <mask> is automatically included when creating a vocab
        super().__init__(counter, specials=["<pad></w>", "<unk></w>", "<sosp></w>", "<sosj></w>","<eos></w>", "<cls></w>"],
                         max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentece, seq_len, with_eos=False, with_sos=False) -> list:
        pass

    def from_seq(self, seq, join=False, with_pad=False):
        pass

    @staticmethod
    def load_vocab(vocab_path: str) -> 'Vocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)

    def save_vocab(self, vocab_path):
        with open(vocab_path, "wb") as f:
            pickle.dump(self, f)


class TokenVocab(Vocab):
    def __init__(self, BPEObject, max_size=None,min_freq=1):
        print("Building Vocab")

        self.sorted_tokens = BPEObject.sorted_tokens
        self.tokenize_word = BPEObject.tokenize_word
        self.vocab_tokenization = BPEObject.vocab_tokenization
        counter = BPEObject.tokens_frequencies
        
        super().__init__(counter, max_size=max_size, min_freq=min_freq)

    def to_seq(self, sentence, seq_len=None, with_eos=False, with_sos=False, with_len=False):
        if isinstance(sentence, str):
            sentence = sentence.split()

        seq = [self.stoi.get(word, self.unk_index) for word in sentence]

        if with_eos:
            seq += [self.eos_index]
        if with_sos:
            seq = [self.sos_index] + seq

        origin_seq_len = len(seq)

        if seq_len is None:
            pass
        elif len(seq) <= seq_len:
            seq += [self.pad_index for _ in range(seq_len - len(seq))]
        else:
            seq = seq[:seq_len]

        return (seq, origin_seq_len) if with_len else seq

    def from_seq(self, seq, join=False, with_pad=False):
        words = [self.itos[idx]
                 if idx < len(self.itos)
                 else "<%d>" % idx
                 for idx in seq
                 if not with_pad or idx != self.pad_index]

        return " ".join(words) if join else words

    @staticmethod
    def load_vocab(vocab_path: str) -> 'TokenVocab':
        with open(vocab_path, "rb") as f:
            return pickle.load(f)


def build():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--corpus_path", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)
    parser.add_argument("-s", "--vocab_size", type=int, default=None)
    parser.add_argument("-f", "--min_freq", type=int, default=2)
    parser.add_argument("-m", "--num_merges", type=int, default=5000)
    args = parser.parse_args()

    # load type nodes
    type_path = "data/type_list/"
    files = os.listdir(type_path)
    type_node = []
    for file in files:
        with open(type_path+file, "r", encoding='utf-8') as f:
            s = f.read()
            type_node += s.split()

    except_list = ['False','None', 'True','and','as', 'assert','break',
                    'class','continue', 'def','del','elif', 'else','except',
                    'finally', 'for', 'from','global','if','import','in','is',
                    'lambda', 'nonlocal','not','or','pass','raise', 'return',
                    'try','while','with','yield','NEWLINE','INDENT','<mask>'] + [
                    "abstract", "assert", "boolean", "break", "byte", "case", "catch", 
                    "char", "class", "const", "continue", "default", "do", "double", 
                    "else", "enum", "extends", "final", "finally", "float", "for", "goto",
                    "if", "implements", "import", "instanceof", "int", "interface", "long", 
                    "native", "new", "package", "private", "protected", "public", "return", 
                    "strictfp", "short", "static", "super", "switch", "synchronized", "this", 
                    "throw", "throws", "transient", "try", "void", "volatile", "while"]

    except_list.extend(type_node)
    # except_list include type nodes and key word in code, which do not need to adopt BPE
    except_list = list(set(except_list))

    # Create BPE object
    BPE_object = BPE(args.corpus_path,except_list=except_list,num_merges=args.num_merges)
    vocab = TokenVocab( BPE_object,max_size=args.vocab_size,min_freq=args.min_freq)

    print("VOCAB SIZE:", len(vocab))
    vocab.save_vocab(args.output_path)

# build()

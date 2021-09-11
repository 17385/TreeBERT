import re, collections

class BPE(object):
    def __init__(self, sourceFile, targetFile, num_merges=5):
        print('==========')
        print("Adopt BPE")
        vocab = self.get_vocab(sourceFile, targetFile)

        self.tokens_frequencies, self.vocab_tokenization = self.get_tokens_from_vocab(vocab)

        for i in range(num_merges):
            pairs = self.get_stats(vocab)
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            vocab = self.merge_vocab(best, vocab)
            if(i % 100 == 0):
                print('Iter: {}'.format(i))
                print('Number of tokens: {}'.format(len(self.tokens_frequencies.keys())))
                print('==========')

            self.tokens_frequencies, self.vocab_tokenization = self.get_tokens_from_vocab(vocab)

        sorted_tokens_tuple = sorted(self.tokens_frequencies.items(), key=lambda item: (self.measure_token_length(item[0]), item[1]), reverse=True)
        self.sorted_tokens = [token for (token, freq) in sorted_tokens_tuple]

        print('==========')


    @staticmethod
    def encode(vocab_tokenization, tokenize_word, sorted_tokens, texts, max_subtoken_len=3):
        encode_output = []
        for word_given in texts:
            word_given = word_given.lower() + '</w>'

            if word_given in vocab_tokenization:
                tmpWord = vocab_tokenization[word_given]

            else:
                if(word_given=="<mask>"):
                    tmpWord = ["<mask></w>"]
                else:
                    tmpWord = tokenize_word(string=word_given, sorted_tokens=sorted_tokens, unknown_token='<unk></w>')

            tmpWord = tmpWord[:max_subtoken_len]
            if (len(tmpWord)<max_subtoken_len):
                padding = ["<pad></w>" for _ in range(max_subtoken_len-len(tmpWord))]
                tmpWord.extend(padding)
            encode_output.append(tmpWord) 
        return encode_output
        
    @staticmethod
    def decode():
        '''
        TODO: concatenate all the tokens together and replace "<\w> with space
        '''
        pass

    def get_vocab(self, sourceFile, targetFile):
        vocab = collections.Counter()
        with open(sourceFile, 'r', encoding='utf-8') as fhand:
            for line in fhand:
                words = line.replace("\n", "").replace("\t", " ").replace("|", " ").replace("\/?", "").replace("/", " / ").split()
                for word in words:
                    vocab[' '.join(list(word)) + ' </w>'] += 1
        with open(targetFile, 'r', encoding='utf-8') as fhand:
            for line in fhand:
                words = line.replace("\n", "").replace("\t", " ").replace("\/?", "").replace("/", " / ").split()
                for word in words:
                    vocab[' '.join(list(word)) + ' </w>'] += 1
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
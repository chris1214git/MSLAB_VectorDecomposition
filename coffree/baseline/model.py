import torch
import torch.nn as nn
import random

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [1, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)

class Seq2Seq(nn.Module):
    def __init__(self, decoder, device):
        super().__init__()

        self.device = device
        self.decoder = decoder
        self.apply(init_weights)
        
    def forward(self, doc_emb, trg, teacher_forcing_ratio = 0.5):
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #doc_emb = [batch size, embedding_dim]
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        hidden = torch.unsqueeze(doc_emb, 0)
        cell = torch.unsqueeze(doc_emb, 0)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs

    def predict(self, doc_emb, word2idx, idx2word, tfidf_word2idx, max_len=50):
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #doc_emb = [batch size, embedding_dim]
        trg_len = max_len
        batch_size = len(doc_emb)
        trg_vocab_size = len(word2idx)
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        prediction = torch.zeros(trg_len, batch_size)
        predict_voc = torch.zeros(batch_size, len(tfidf_word2idx))
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]
        hidden = torch.unsqueeze(doc_emb, 0)
        cell = torch.unsqueeze(doc_emb, 0)
        
        #first input to the decoder is the <sos> tokens
        input = torch.LongTensor([word2idx["<SOS>"]] * len(doc_emb)).to(self.device)
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #get the highest predicted token from our predictions
            input = output.argmax(1)

            prediction[t] = input

            for b in range(batch_size):
                if (input[b] != word2idx["<SOS>"] and input[b] != word2idx["<PAD>"]
                     and input[b] != word2idx["<EOS>"] and input[b] != word2idx["<UNK>"]):
                    input_idx = int(input[b])
                    predict_label = tfidf_word2idx.get(idx2word[input_idx], -1)
                    if (predict_label != -1):
                        predict_voc[b][predict_label] += 1

        return prediction.transpose(0, 1), predict_voc
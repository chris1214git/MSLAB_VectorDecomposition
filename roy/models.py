import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, pretrianed_embedding=None):
        super().__init__()
        if pretrianed_embedding is None:
            self.word_embedding = nn.Parameter(torch.zeros(hidden_dim, output_dim))
        else:
            self.word_embedding = nn.Parameter(torch.FloatTensor(pretrianed_embedding))
        self.transform = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        docvec = self.transform(x)
        decoded = torch.sigmoid(torch.matmul(docvec,self.word_embedding))
        return decoded

class W2vDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.word2vec = EmbeddingModel(output_dim,hidden_dim)
        self.transform = nn.Sequential(
            nn.Linear(input_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim,hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.Sigmoid(),
        )
        self.post_layers = nn.Sequential(
            nn.Sigmoid(),
        )
    def forward(self, x):
        docvec = self.transform(x)
        word_embedding = self.word2vec.out_embed.weight.T
        #decoded = self.post_layers(sim_matrix(docvec,self.word_embedding.T))
        decoded = self.post_layers(torch.matmul(docvec,word_embedding))
        return decoded
    
    def load_pretrianed(self,word_embedding):
        self.word_embedding = nn.Parameter(torch.sigmoid(torch.FloatTensor(word_embedding)))
        
    def word2vec_loss(self,input_labels, pos_labels, neg_labels):
        return self.word2vec(input_labels, pos_labels, neg_labels)


class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(EmbeddingModel, self).__init__()
        '''
        Word2vec embedding model
        '''
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
#         self.in_embed = nn.Embedding(self.vocab_size, self.embed_size)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size)
        
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' 
        Args:
            input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels：negative words, [batch_size, (window_size * 2 * K)]
        Return:
            loss: [batch_size]
        '''
        input_embedding = torch.sigmoid(self.out_embed(input_labels)) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]
        
        neg_dot = torch.bmm(neg_embedding, -input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]
        
        pos_dot = torch.clamp(pos_dot, max=10, min=-10)
        neg_dot = torch.clamp(neg_dot, max=10, min=-10)
        log_pos = F.logsigmoid(pos_dot).sum(1) # .sum()結果只為一個數，.sum(1)結果是一維的張量
        log_neg = F.logsigmoid(neg_dot).sum(1)
        
        loss = log_pos + log_neg
        
        return -loss
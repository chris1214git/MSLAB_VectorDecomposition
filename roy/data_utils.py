import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class IDEDataset(Dataset):
    def __init__(self, 
                 doc_vectors,
                 weight_ans):
        '''
        Args:
            doc_vectors: document vectors
            weight_ans: importance socre of each word in each document
        '''
        
        assert len(doc_vectors) == len(weight_ans)

        self.doc_vectors = torch.FloatTensor(doc_vectors)
        self.weight_ans = torch.FloatTensor(weight_ans)        
        self.weight_ans_s = torch.argsort(self.weight_ans, dim=1, descending=True)
        self.topk = torch.sum(self.weight_ans > 0, dim=1)
        
    def __getitem__(self, idx):
        return self.doc_vectors[idx], self.weight_ans[idx], self.weight_ans_s[idx], self.topk[idx]

    def __len__(self):
        return len(self.doc_vectors)


class WordEmbeddingDataset(Dataset):
    def __init__(self, text_encoded, word_freqs, window_size=3, num_negs=5):
        ''' 
        Args:
            text: a list of words, all text from the training dataset
            word_freqs: the frequency of each word
            window_size: word2vec window size
            num_negs: # of negative samples
        '''
        super(WordEmbeddingDataset, self).__init__()
        self.text_encoded = text_encoded 
        self.word_freqs = torch.Tensor(word_freqs)
        self.window_size = window_size
        self.num_negs = num_negs
        self.init_training_data()
        
    def init_training_data(self):
        self.center_words = []
        self.context_words = []
        for doc in tqdm(self.text_encoded,desc="Init word2vec training instances"):
            for center_wid in range(self.window_size,len(doc) - self.window_size):
                self.center_words.append(doc[center_wid])
                neighbor_words = list(doc[center_wid-self.window_size:center_wid]) + list(doc[center_wid+1:center_wid+self.window_size+1])
                self.context_words.append(neighbor_words)
        self.center_words = torch.LongTensor(self.center_words)
        self.context_words = torch.LongTensor(self.context_words)
        
    def __len__(self):
        return len(self.center_words) # 返回所有單詞的總數，即item的總數
    
    def __getitem__(self, idx):
        ''' 這個function返回以下資料用於訓練
            - 中心詞
            - 這個單詞附近的positive word
            - 隨機取樣的K個單詞作為negative word
        '''
        center_words, pos_words = self.center_words[idx], self.context_words[idx]
        
        neg_words = torch.multinomial(self.word_freqs, self.num_negs * pos_words.shape[0], True)
        # torch.multinomial作用是對self.word_freqs做K * pos_words.shape[0]次取值，輸出的是self.word_freqs對應的下標
        # 取樣方式採用有放回的取樣，並且self.word_freqs數值越大，取樣概率越大
        # 每取樣一個正確的單詞(positive word)，就取樣K個錯誤的單詞(negative word)，pos_words.shape[0]是正確單詞數量
        return center_words, pos_words, neg_words
    
    
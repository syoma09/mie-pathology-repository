import torch
import torch.nn as nn
import numpy as np

# Extracter + TransformerEncoder
class ExtTrans(torch.nn.Module):
    def __init__(self,ext ,est ,PE):
        super().__init__()
        self.ext = ext
        self.est = est
        self.PE = PE
        self.flatten = nn.Flatten()
        self.network = net = nn.Sequential(
        nn.Linear(8*128, 8*64, bias=True),
        nn.BatchNorm1d(8*64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(8*64, 8*8, bias=True),
        nn.BatchNorm1d(8*8),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(8*8, 4, bias=True),
        )
    def forward(self, sort_features_list,sort_clusters_list):
        #x = self.ext(x)
        #sort_features,sort_clusters = features_sort(x)
        x = self.PE(sort_features_list,sort_clusters_list).to(device)
        #print(x.shape)
        x = self.est(x)
        #x = self.flatten(x)
        #x = self.network(x)
        #print(x.shape)
        return x

# Clustering PE
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        #self.register_buffer('pe', pe)

    def forward(self, x,sort_clusters_list):
        
        #Args:
        #    x: Tensor, shape [batch_size, seq_len, embedding_dim]
        
        #batch_size = self.batch_size
        #print(len(sort_clusters_list[0]))
        pe = torch.zeros(1,len(sort_clusters_list[0]) , self.d_model).to(device)
        
        for i, position in enumerate(sort_clusters_list):
            for j in range(len(position)):
                pe[0, j, 0::2] = torch.sin(position[j] * self.div_term)
                pe[0, j, 1::2] = torch.cos(position[j] * self.div_term)
            x[i] = x[i]*math.sqrt(self.d_model) + pe[:x.size(1)]
            
            #print(a)
        #print(x.shape)
        
        return self.dropout(x)
    
class TransformerModel(nn.Module):

    def __init__(self, d_model: int, nhead: int, d_hid: int, 
                    nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        #self.pos_encoder = PositionalEncoding(claster, d_model, dropout)
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, d_hid, dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        #self.encoder = nn.Embedding(ntoken, d_model)
        self.d_model = d_model
        
        #self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        

    def forward(self, src):
        """
        Args:
            src: Transformerへの入力データ
        Returns:
            Transformerの出力
        """
        
        #src = self.pos_encoder(src,claster)
        output = self.transformer_encoder(src)
        
        return output

def features_sort(features):

    clusters = KMeans(n_clusters = 4,n_init='auto').fit(features.cpu().detach().numpy())
    sort_clusters, sort_clusters_index = torch.sort(torch.tensor(clusters.labels_),dim=0)
    sort_features = features[sort_clusters_index]
    return sort_features, sort_clusters
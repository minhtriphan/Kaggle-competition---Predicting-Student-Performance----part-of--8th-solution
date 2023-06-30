import os, pickle, copy, math
import torch
from torch import nn
import torch.nn.functional as F

from custom_config import cfg, NUM_COLS, TXT_COLS, LEVEL2QUESTION

# Custom transformer-encoder layer with attention weights
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def attention(query, key, value, attn_weight = None, mask = None, dropout = None):
    "Compute 'Scaled Dot Product Attention'"
    batch_size, seq_len, d_k = query.shape[0], query.shape[2], query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        mask = mask.view(batch_size, 1, 1, seq_len)
        scores = scores.masked_fill(mask, float('-inf'))    # Masking at mask == True
    if attn_weight is not None:
        # Check the dimmension
        assert len(attn_weight.shape) == 4, "'attn_weight' should have 4 dimensions, (batch_size, num_head, T, T)."
        scores -= attn_weight
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

class WeightedMultiHeadedAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout = 0.1):
        "Take in model size and number of heads."
        super(WeightedMultiHeadedAttention, self).__init__()
        assert d_model % nhead == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // nhead
        self.nhead = nhead
        self.linears = clones(nn.Linear(d_model, d_model, bias = False), 4) # Q, K, V, last
        self.attn = None
        self.dropout = nn.Dropout(p = dropout)

    def forward(self, query, key, value, attn_weight = None, mask = None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.nhead, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, attn_weight = attn_weight, mask = mask, dropout = self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.nhead * self.d_k)
        return self.linears[-1](x)

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
    
class TransformerEncoderLayer(nn.Module):
    """
    Single Encoder Block
    """
    def __init__(self, d_model, nhead, dim_feedforward = 1024, dropout = 0.1, batch_first = False):
        super(TransformerEncoderLayer, self).__init__()
        self._self_attn = WeightedMultiHeadedAttention(d_model, nhead, dropout)
        self._ffn = PositionwiseFeedForward(d_model, dim_feedforward, dropout)
        self._layernorms = clones(nn.LayerNorm(d_model, eps = 1e-6), 2)
        self._dropout = nn.Dropout(dropout)

    def forward(self, src, attn_weight = None, src_key_padding_mask = None):
        """
        query: question embeddings
        key: interaction embeddings
        """
        # self-attention block
        src2 = self._self_attn(query = src, key = src, value = src, attn_weight = attn_weight, mask = src_key_padding_mask)
        src = src + self._dropout(src2)
        src = self._layernorms[0](src)
        src2 = self._ffn(src)
        src = src + self._dropout(src2)
        src = self._layernorms[1](src)
        return src
    
class TransformerEncoder(nn.Module):
    """Stack of N single transformer-encoder blocks"""
    def __init__(self, encoder_layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.encoder_layers = clones(encoder_layer, num_layers)
        
    def forward(self, src, attn_weight = None, src_key_padding_mask = None):
        for layer in self.encoder_layers:
            src = layer(src, attn_weight = attn_weight, src_key_padding_mask = src_key_padding_mask)
        return src
        
# Pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x, mask = None):
        if mask is None:
            return torch.mean(x, dim = 1)    # Mean pooling over the time
        else:
            return (x * (~mask).unsqueeze(-1)).sum(dim = 1) / (~mask).unsqueeze(-1).sum(dim = 1)

# Main model
class PSPModel(nn.Module):
    def __init__(self, cfg, TXT_COL_MAPS, level = '0-4', hidden_size = 128):
        super().__init__()
        self.cfg = cfg
        self.level = level
        # Categorical variables embeddings
        self.txt_embeddings = {}
        TXT_DIM = 0
        for col, maps in TXT_COL_MAPS.items():
            self.txt_embeddings[col] = nn.Embedding(num_embeddings = len(maps), 
                                                    embedding_dim = 8)
            TXT_DIM += 8
        self.txt_embeddings = nn.ModuleDict(self.txt_embeddings)
        
        # Numerical normalization
        self.num_batch_norm = nn.BatchNorm1d(len(NUM_COLS) - 1)
        
        # The first layer
        self.gru = nn.GRU(input_size = len(NUM_COLS) + TXT_DIM - 1,
                          hidden_size = hidden_size // 2,
                          num_layers = 3,
                          batch_first = True,
                          dropout = 0.2,
                          bidirectional = True)
        
        # Transformer-encoder
        self.attn_weight = nn.Parameter(torch.randn(hidden_size // 32).view(hidden_size // 32, 1, 1).pow(2).to(cfg.device))
        
        encoder = TransformerEncoderLayer(d_model = hidden_size, 
                                          nhead = hidden_size // 32, 
                                          dim_feedforward = hidden_size, 
                                          batch_first = True)
        self.encoder = TransformerEncoder(encoder, num_layers = 3)
               
        
        # Pooling and output
        self.pooler = MeanPooling()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size * 4),
            nn.ReLU(),
        )
        
        self.output = nn.Linear(hidden_size * 4, len(LEVEL2QUESTION[level]))
        self.aux_output = nn.Linear(hidden_size * 4, 18 - len(LEVEL2QUESTION[level]))
        if level != '13-22':
            self.pred_answering_time = nn.Linear(hidden_size * 4, 1)
        
    def _get_time_attn_weight(self, elapsed_time):
        time_diff = elapsed_time.unsqueeze(-1) - elapsed_time.unsqueeze(1)
        time_diff = torch.clip(time_diff, min = 1e-6, max = 3.6e6)    # Clip to 1 hour between actions
        time_diff /= 60 * 1e3
        time_diff = torch.log(time_diff)
        return time_diff.unsqueeze(1)    # Shape: (batch_size, 1, seq_len, seq_len)
        
    def loss_fn(self, pred, true, aux_pred = None, aux_true = None, pred_answering_time = None, true_answering_time = None):
        if pred_answering_time is not None:
            return nn.BCEWithLogitsLoss()(pred, true) + nn.BCEWithLogitsLoss()(aux_pred, aux_true) + nn.MSELoss()(pred_answering_time, true_answering_time)
        return nn.BCEWithLogitsLoss()(pred, true) + nn.BCEWithLogitsLoss()(aux_pred, aux_true)
        
    def forward(self, inputs, mask = None, label = None, aux_label = None):
        # Embed the features
        num_features = torch.cat([inputs[col].unsqueeze(-1) for col in NUM_COLS if col != 'index'], dim = -1)
        num_features = self.num_batch_norm(num_features.permute(0, 2, 1)).permute(0, 2, 1)
        txt_features = torch.cat([self.txt_embeddings[col](inputs[col]) for col in TXT_COLS], dim = -1)
        
        features = torch.cat([num_features, txt_features], dim = -1)
        
        # Trimming the sequences
        if mask is not None:
            local_len = (~mask).sum(axis = 1).max().item()
            features = features[:,:local_len]
            time_diff = inputs['time_diff'][:,:local_len]
            mask = mask[:,:local_len]
            lengths = (~mask).sum(axis = 1).cpu()
        else:
            time_diff = inputs['time_diff']
        
        if mask is not None:
            features = nn.utils.rnn.pack_padded_sequence(features, lengths = lengths, batch_first = True, enforce_sorted = False)
            features, _ = self.gru(features)
            features, _ = nn.utils.rnn.pad_packed_sequence(features, batch_first = True, padding_value = -1.)
        else:
            features, _ = self.gru(features)
        
        # Time attention weight
        time_attn_weight = self._get_time_attn_weight(time_diff) * self.attn_weight
        
        # Transformer-encoder
        features = self.encoder(features, attn_weight = time_attn_weight, src_key_padding_mask = mask)
        
        # Pooling and output
        features = self.pooler(features, mask = mask)
        pooled_features = features.contiguous()
        features = self.classifier(features)
        output = self.output(features)
        
        aux_output = self.aux_output(features)
        
        if self.level != '13-22':
            pred_answering_time = self.pred_answering_time(features).unsqueeze(-1)
        else:
            pred_answering_time = None
        
        if label is not None:
            loss = self.loss_fn(output, label, aux_output, aux_label, pred_answering_time, inputs['answering_time'])
        else:
            loss = None
        return loss, output, aux_output, pooled_features

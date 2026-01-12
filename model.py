import math

import torch
import torch.nn as nn


class InputEmbeddings(nn.Module):
    def __init__(self, d_model:int, vocab_size:int) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size,self.d_model)

    def forward(self,x):
        return self.embedding(x) * math.sqrt(self.d_model)

class PostionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len:int , dropout:float) -> None:
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # postional encoding of size d_model rows and seq
        # Each row represents a position in the sentence, Each column represents a dimension of the embedding
        # If you used (d_model, seq_len), you would have to transpose your data every time you wanted to add the
        # encoding to your word embeddings. Keeping it as (seq_len, d_model) allows you to simply do: embeddings + pos_encoding.
        self.pos_encoding = torch.zeros(seq_len,d_model)

        # create a vector of shape seq_len filled with 0
        pos = torch.arange(0,seq_len,dtype=torch.float)
        # convert vector into vector of vector like this [[0],[0],[0]] i.e flat row into a column:
        # represents for each word value
        # add a new dimesion at index 1 with value 1
        pos = pos.unsqueeze(1)

        # as we are going to fill odd and even values sepratly we only need d_model/2 div_term
        # data doesnt change just for even places we take sin and for odd placed we take cos
        div_term = torch.exp(
                torch.arange(0,d_model,2).float() * (-math.log(10000.0)/d_model)
                )

        self.pos_encoding[:,0::2] = torch.sin(pos * div_term)
        self.pos_encoding[:,1::2] = torch.cos(pos * div_term)

        # [rows (seq_len), column(d_model)] -> [batch_size(1),rows (seq_len), column(d_model)]
        # model porcess data in batches
        # add a new dimesion at index 0 with value 1
        self.pos_encoding = self.pos_encoding.unsqueeze(0)

        # save data in file
        self.register_buffer("PostionalEncoding", self.pos_encoding)

    def forward(self,x):
        # slicing opration
        # 1st ':' selects all elements along the first dimension
        # :x.shape[1] select elements along the second axis (columns) from the beginning (default start index 0)
        #   up to (but not including) the index specified by the length of the second dimension of the array x itself
        # 3rd ':' selects all elements along the 3rd dimension
        x =  x + (self.pos_encoding[:, :x.shape[1], :]).to(x.device).requires_grad_(False)
        return self.dropout(x)

class LayerNormalizatin(nn.Module):
    def __init__(self, esp: float = 10**-6) -> None:
        super().__init__()

        self.esp = esp
        self.alph = nn.Parameter(torch.ones(1)) # Multiplied
        self.bais = nn.Parameter(torch.zeros(1)) # Added

    def forward(self,x):
        mean = x.mean(dim = -1,keepdim=True)
        std = x.std(dim = -1,keepdim=True)
        return self.alph * ( x - mean) / (std + self.esp) + self.bais

class FeedForwardBlock(nn.Module):
    def __init__(self, d_model:int, d_ff:int, dropout:float ) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(d_model,d_ff) # w_1 and b_1
        self.dropout = nn.Dropout(dropout)

        self.linear_2 = nn.Linear(d_ff,d_model) # w_2 and b_2

    def forward(self,x):
        # (batch , seq_len, d_model) -> (batch , seq_len, d_ff) ->  (batch , seq_len, d_model)
        x = self.linear_1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear_2(x)

        return x


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, dropout:float ) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # number of head

        # d_model contain embding size , each head will try to undesratnd word with and diffrent meanng
        # for a word 1 head learns it as a nounce other word lears it as verb
        # each word will get all sequnce and some amount of embedding whihc tell realtion with other words
        assert d_model % h == 0, "d_model not divisble by h"

        self.d_k = d_model // h # spliting the input so each head can process some number of inputs

        self.w_q = nn.Linear(d_model,d_model)
        self.w_k = nn.Linear(d_model,d_model)
        self.w_v = nn.Linear(d_model,d_model)

        self.w_o = nn.Linear(d_model,d_model)

        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # @ = matrics multiplication in pytroch
        # transpose(-2,-1) = (batch , h,seq_len, d_model) ->  (batch , h, seq_len, seq_len)
        attention_score = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        if mask is not None:
            # replace anything whihc is 0 to -1e9
            # as softmax of 0 is 0.5
            attention_score.masked_fill_(mask == 0, -1e9)
        # will repalce of -1e9 is 0
        attention_score = attention_score.softmax(dim = -1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        # attention_score is used for visulization
        return (attention_score @ value) ,attention_score


    def forward(self,q,k,v,mask):
        # (batch , seq_len, d_model) ->  (batch , seq_len, d_model)
        qurey = self.w_q(q) # is mathematically Q=q⋅W^Q.
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch , seq_len, d_model[512]) ->  (batch , seq_len, d_model, h, d_k)
        # Don't see a single line of 512 features; see 8 groups of 64 features. [Batch, Seq_Len, 8, 64]
        qurey = qurey.view(qurey.shape[0], qurey.shape[1], self.h, self.d_k)
        # (batch, seq_len, h, d_k) ->  (batch , h, seq_len, d_k)
        # each head h in batch will process seq_len*d_k embedding
        qurey = qurey.transpose(1,2)

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_score = MultiHeadAttentionBlock.attention(qurey,key,value,mask,self.dropout)

        # (batch , h, seq_len, d_k) -> (batch, seq_len, d_model, h, d_k) -> (batch , seq_len, d_model)
        # 1. Swap Seq_Len and Heads back
        x = x.transpose(1, 2) # [Batch, Seq_Len, 8, 64]
        # 2. Flatten the last two dimensions (8 * 64 = 512)
        x = x.contiguous().view(x.shape[0], -1, self.h * self.d_k) # [Batch, Seq_Len, 512]

        return self.w_o(x)

# Skip connection
class ResidualConnection(nn.Module):
    def __init__(self, dropout:float ) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalizatin()

    def forward(self, x, sublayer):
        #sublayer is MultiHeadAttentionBlock or FeedForwardBlock
        y = self.norm(x)
        y = sublayer(y)
        y = self.dropout(y)

        # add and norm
        return x + y

# === Encoder ===
class EncoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([
                ResidualConnection(dropout) for _ in range(2)
            ])

    def forward(self,x, src_mask):
        # src_mask we dont want the padding words to ittracat with accutle words

        # calcaulte muli head attention then add and norm the input
        x = self.residual_connections[0](x,
                                      lambda x: self.self_attention_block(x,x,x,src_mask)
                                      )

        # feed_forward
        x = self.residual_connections[1](x, self.feed_forward)
        return x;
# Encoder is made of n_x encoder block
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalizatin()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

# === Decoder ===

class DecoderBlock(nn.Module):

    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock,feed_forward: FeedForwardBlock, dropout: float ) -> None:
        super().__init__()

        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block

        self.feed_forward = feed_forward
        self.dropout = nn.Dropout(dropout)
        self.residual_connections = nn.ModuleList([
                ResidualConnection(dropout) for _ in range(3)
            ])

    def forward(self, x, encoder_output, src_mask, tgt_mask):

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x,x,x,tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output,encoder_output,src_mask))

        x = self.residual_connections[2](x, self.feed_forward)

        return x



# Decoder is made of n_x Decoder block
class Decoder(nn.Module):

    def __init__(self, layers: nn.ModuleList):
        super().__init__()

        self.layers = nn.ModuleList(layers)
        self.norm = LayerNormalizatin()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x,encoder_output,src_mask,tgt_mask)
        return self.norm(x)

# === Linear layer ===

class ProjectionLayer(nn.Module):

    def __init__(self, d_model:int, vocab_size:int ):
        super().__init__()
        self.proj = nn.Linear(d_model,vocab_size)

    def forward(self, x):
        # (batch, seq_len, d_model) -> (batch, seq_len, vocab_size)
        x = torch.log_softmax(self.proj(x), dim= -1)

        return x
class Transformer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_emd: InputEmbeddings, tgt_emd: InputEmbeddings, src_pos : PostionalEncoding, tgt_pos: PostionalEncoding, projection_layer: ProjectionLayer):
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_emd = src_emd
        self.tgt_emd = tgt_emd
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self,x,src_mask):
        src = self.src_emd(x)
        src = self.src_pos(src)
        return self.encoder(src,src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_emd(tgt)
        tgt = self.tgt_pos(tgt)

        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x):
        return self.projection_layer(x)

# === Build transfomer
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: int, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048):
    # create the embedding
    src_emd = InputEmbeddings(d_model,src_vocab_size)
    tgt_emd = InputEmbeddings(d_model,tgt_vocab_size)

    # create postional encoding
    src_pos = PostionalEncoding(d_model,src_seq_len,dropout)
    tgt_pos = PostionalEncoding(d_model,tgt_seq_len,dropout)

    encoder_blocks = []
    for _ in range(N):
        encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    decoder_blocks = []
    for _ in range(N):
        decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
        feed_forward_block = FeedForwardBlock(d_model,d_ff,dropout)
        decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
        decoder_blocks.append(decoder_block)

    encoder = Encoder(encoder_blocks)
    decoder = Decoder(decoder_blocks)
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

    transformer = Transformer(encoder, decoder, src_emd, tgt_emd, src_pos, tgt_pos, projection_layer)

    #initlize Parameter
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_normal(p)

    return transformer



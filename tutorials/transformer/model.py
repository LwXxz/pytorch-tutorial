import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def make_batch():
  input_batch = [[src_vocab[n] for n in sentences[0].split()]]
  output_batch = [[tgt_vocab[n] for n in sentences[1].split()]]
  target_batch = [[tgt_vocab[n] for n in sentences[2].split()]]
  return torch.LongTensor(input_batch), torch.LongTensor(output_batch), torch.LongTensor(target_batch)

def get_sinusoid_encoding_table(n_position, d_model):
  def cal_angle(position, hid_idx):
      return position / np.power(10000, 2 * (hid_idx // 2) / d_model)
  def get_posi_angle_vec(position):
      return [cal_angle(position, hid_j) for hid_j in range(d_model)]

  sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])
  sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
  sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
  return torch.FloatTensor(sinusoid_table)

def get_attn_pad_mask(seq_q, seq_k):
  # print(seq_q)
  batch_size, len_q = seq_q.size()
  batch_size, len_k = seq_k.size()
  # eq(zero) is PAD token
  pad_attn_mask = seq_k.data.eq(0).unsqueeze(1)  # batch_size x 1 x len_k(=len_q), one is masking
  return pad_attn_mask.expand(batch_size, len_q, len_k)  # batch_size x len_q x len_k

def get_attn_subsequent_mask(seq):
  attn_shape = [seq.size(0), seq.size(1), seq.size(1)]
  subsequent_mask = np.triu(np.ones(attn_shape), k=1)
  subsequent_mask = torch.from_numpy(subsequent_mask).byte()
  return subsequent_mask

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, hidden_channels):
        super(MLP, self).__init__()
        self.num_layers = num_layers
        layer_hidden = [hidden_channels] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip([in_channels] + layer_hidden, 
                                                                    layer_hidden + [out_channels]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
        
class ScaleDotProductAttention(nn.Module):
    def __init__(self) -> None:
        super(ScaleDotProductAttention, self).__init__()

    def forward(self, q, k, v, atten_mask):
        score = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(d_k)
        score.mask_fill_(atten_mask, -1e9)
        attn = nn.Softmax(dim=-1)(score)

class MultiHeadAttention(nn.Module):
    def __init__(self, ) -> None:
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads) 
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, atten_mask):
        residual, batch_size = Q, Q.size(0)

        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # q_s: [batch_size x n_heads x len_q x d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # k_s: [batch_size x n_heads x len_k x d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # v_s: [batch_size x n_heads x len_k x d_v]

        atten_mask = atten_mask.unsqueeze(1).repeat(1, n_heads, 1, 1) # attn_mask : [batch_size x n_heads x len_q x len_k]

        context, attn = ScaleDotProductAttention()(q_s, k_s, v_s, atten_mask) # 注意力的计算
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size x len_q x n_heads * d_v] concat的作用
        output = self.linear(context) # 线性映射
        return self.layer_norm(output + residual), attn # output: [batch_size x len_q x d_model]

class PositionFeedForwardNet(nn.Module):
    def __init__(self) -> None:
        super(PositionFeedForwardNet, self).__init__()
        self.fc1 = nn.Conv1d(d_model, d_ff, 1)
        self.fc2 = nn.Conv1d(d_ff, d_model, 1)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        output = nn.ReLU()(self.fc1(input.transpose(1, 2)))
        output  = self.fc2(output).transpose(1, 2)
        return self.layer_norm(output + residual)



class EncoderLayer(nn.Module):
    def __init__(self) -> None:
        super(EncoderLayer, self).__init__()
        self.encoder_self_attn = MultiHeadAttention()
        self.pos_fnn = PositionFeedForwardNet()

    def forward(self, x, x_self_attn_mask):
        encoder_output, encoder_self_attn = self.encoder_self_attn(x, x, x, x_self_attn_mask)
        encoder_output = self.pos_fnn(encoder_output)
        return encoder_output, encoder_self_attn

class DecoderLayer(nn.Module):
    def __init__(self) -> None:
        super(DecoderLayer, self).__init__()
        self.decoder_self_attn = MultiHeadAttention()
        self.decoder_encoder_attn = MultiHeadAttention()
        self.pos_fnn = PositionFeedForwardNet()

    def forward(self, x_decoder, x_encoder, x_decoder_attn_mask, x_encoder_decoder_attn_mask):
        decoder_output, decoder_self_attn = self.decoder_self_attn(x_decoder, x_decoder, x_decoder, x_decoder_attn_mask)
        decoder_output, decoder_encoder_attn = self.decoder_encoder_attn(x_encoder, x_encoder, decoder_output, x_encoder_decoder_attn_mask)
        decoder_output = self.pos_fnn(decoder_output)
        return decoder_output, decoder_self_attn, decoder_encoder_attn

class TransformerEncoder(nn.Module):
    def __init__(self) -> None:
        super(TransformerEncoder, self).__init__()
        self.src_emb = nn.Embedding(src_len, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(src_len+1, d_model), freeze=True)
        self.layer = nn.Module([EncoderLayer for _ in range(n_layers)])

    def forward(self, encoder_input):
        encoder_output = self.src_emb(encoder_input) + self.pos_emb(torch.LongTensor([[1,2,3,4,0]]))
        encoder_self_attn_mask = get_attn_pad_mask(encoder_input, encoder_input)
        encoder_self_attention = []

        for layer in self.layer:
            encoder_output, encoder_self_attn = layer(encoder_output, encoder_self_attn_mask)
            encoder_self_attention.append(encoder_self_attn)
        # encoder_output是输入经过多个encoder层之后的输出，encoder_self_attention是对每一层的自注意力进行计算
        return encoder_output, encoder_self_attention

class TransformerDecoder(nn.Module):
    def __init__(self) -> None:
        super(TransformerDecoder, self).__init__()
        self.tgt_emb = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_emb = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(tgt_len+1, d_model), freeze=True)
        self.layer = nn.ModuleList([DecoderLayer() for _ in range(n_layers)])

    def forward(self, decoder_input, encoder_input, encoder_output):
        decoder_output = self.tgt_emb(decoder_input) + self.pos_emb(torch.LongTensor([[5,1,2,3,4]]))
        decoder_self_attn_pad_mask = get_attn_pad_mask(decoder_input, decoder_input)
        decoder_self_attn_subsequent_mask = get_attn_subsequent_mask(decoder_input)
        decoder_self_atten_mask = torch.gt((decoder_self_attn_pad_mask + decoder_self_attn_subsequent_mask), 0)

        decoder_encoder_attn_mask = get_attn_pad_mask(decoder_input, encoder_input)

        decoder_self_attention, decoder_encoder_attention = [], []
        for layer in self.layer:
            decoder_output, decoder_self_attn, decoder_encoder_attn = layer(decoder_output, encoder_output, decoder_self_atten_mask, decoder_encoder_attn_mask)
            decoder_self_attention.append(decoder_self_attn)
            decoder_encoder_attention.append(decoder_encoder_attn)
        # return deocder的输出，两种不同的attention权重
        return decoder_output, decoder_self_attention, decoder_encoder_attention

class Transformer(nn.Module):
    def __init__(self) -> None:
        super(Transformer, self).__init__()
        self.encoder = TransformerEncoder()
        self.decoder = TransformerDecoder()
        self.linear = nn.Linear(d_model, tgt_vocab_size, bias=True)

    def forward(self, encoder_input, decoder_input):
        encoder_output, encoder_self_attn = self.encoder(encoder_input)
        decoder_output, decoder_self_attention, decoder_encoder_attention = self.decoder(decoder_input, encoder_input, encoder_output)
        decoder_logit = self.linear(decoder_output)
        return decoder_logit.view(-1, decoder_logit.size(-1)), encoder_self_attn, decoder_self_attention, decoder_encoder_attention

if __name__ == '__main__':
  sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']
  # Transformer Parameters
  # Padding Should be Zero index
  src_vocab = {'P': 0, 'ich': 1, 'mochte': 2, 'ein': 3, 'bier': 4}
  src_vocab_size = len(src_vocab)

  tgt_vocab = {'P': 0, 'i': 1, 'want': 2, 'a': 3, 'beer': 4, 'S': 5, 'E': 6}
  number_dict = {i: w for i, w in enumerate(tgt_vocab)}
  tgt_vocab_size = len(tgt_vocab)

  src_len = 5 # length of source
  tgt_len = 5 # length of target

  d_model = 512  # Embedding Size
  d_ff = 2048  # FeedForward dimension
  d_k = d_v = 64  # dimension of K(=Q), V
  n_layers = 6  # number of Encoder of Decoder Layer
  n_heads = 8  # number of heads in Multi-Head Attention

  model = Transformer()

  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  enc_inputs, dec_inputs, target_batch = make_batch()

  for epoch in range(20):
      optimizer.zero_grad()
      outputs, enc_self_attns, dec_self_attns, dec_enc_attns = model(enc_inputs, dec_inputs)
      loss = criterion(outputs, target_batch.contiguous().view(-1))
      print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))
      loss.backward()
      optimizer.step()
import torch.nn as nn
import torch
from nn import InputEmbeddings, PositionalEncoding, Projection, MultiHeadAttention, FeedForward
from encoder import Encoder, EncoderBlock
from decoder import Decoder, DecoderBlock

class Transformer(nn.Module):
	def __init__(self, encoder: Encoder, decoder: Decoder, src_embedding: InputEmbeddings, src_position: PositionalEncoding,
							target_embedding: InputEmbeddings, target_position: PositionalEncoding, projection_layer: Projection) -> None:
		super().__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.src_embedding = src_embedding
		self.target_embedding = target_embedding
		self.src_position = src_position
		self.target_position = target_position
		self.projection_layer = projection_layer
			
	def encode(self, src, src_mask):
		src = self.src_embedding(src)
		src = self.src_position(src)
		return self.encoder(src, src_mask)

	def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor, target_mask: torch.Tensor, target: torch.Tensor):
		target = self.target_embedding(target)
		target = self.target_position(target)
		return self.decoder(target, encoder_output, src_mask, target_mask)
	
	def project(self, x):
		return self.projection_layer(x)


def build_transformer(src_vocab_size: int, target_vocab_size: int, src_seq_len: int, target_seq_len: int,
                      d_model: int=512, N: int=6, h: int=8, dropout: float=0.1, d_ff: int=2048) -> Transformer:
  # embedding layers
  src_embedding = InputEmbeddings(d_model, src_vocab_size)
  target_embedding = InputEmbeddings(d_model, target_vocab_size)
  
  # positional encoding layers
  src_position = PositionalEncoding(d_model, src_seq_len, dropout)
  target_position = PositionalEncoding(d_model, target_position, dropout)
  
  # encoder blocks
  encoder_blocks = []
  for _ in range(N):
    encoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)	
    feed_forward_block = FeedForward(d_model, d_ff, dropout)
    encoder_block = EncoderBlock(d_model, encoder_self_attention_block, feed_forward_block, dropout)
    encoder_blocks.append(encoder_block)
  
  # decoder blocks
  decoder_blocks = []
  for _ in range(N):
    decoder_self_attention_block = MultiHeadAttention(d_model, h, dropout)
    decoder_cross_attention_block = MultiHeadAttention(d_model, h, dropout)
    feed_forward_block = FeedForward(d_model, d_ff, dropout)
    decoder_block = DecoderBlock(d_model, decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
    decoder_blocks.append(decoder_block)
  
  # encoder and decoder
  encoder = Encoder(d_model, nn.ModuleList(encoder_blocks))
  decoder = Decoder(d_model, nn.ModuleList(decoder_blocks))
  
  # projection layer
  projection_layer = Projection(d_model, target_vocab_size)
  
  # transformer
  transformer = Transformer(encoder, decoder, src_embedding, src_position,
                            target_embedding, target_position, projection_layer)
  
  # initialize the parameters
  for p in transformer.parameters():
    if p.dim() > 1:
      nn.init.xavier_uniform_(p)
      
  return transformer
  
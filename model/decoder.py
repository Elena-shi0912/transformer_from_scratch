import torch.nn as nn
from nn import MultiHeadAttention, FeedForward, ResidualConnection, LayerNormalization

class DecoderBlock(nn.Module):
	def __init__(self, features: int, self_attention_block: MultiHeadAttention, cross_attention_block: MultiHeadAttention, feed_forward_block: FeedForward, dropout: int) -> None:
		super().__init__()
		self.self_attention_block = self_attention_block
		self.cross_attention_block = cross_attention_block
		self.feed_forward_block = feed_forward_block
		self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(3)])

	def forward(self, x, encoder_output, src_mask, target_mask):
		x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, target_mask))
		x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
		x = self.residual_connections[2](x, self.feed_forward_block)
		return x
    

class Decoder(nn.Module):
	def __init__(self, features: int, layers: nn.ModuleDict) -> None:
		super().__init__()
		self.layers = layers
		self.norm = LayerNormalization(features)
			
	def forward(self, x, encoder_output, src_mask, target_mask):
		for layer in self.layers:
					x = layer(x, encoder_output, src_mask, target_mask)
		return self.norm(x)
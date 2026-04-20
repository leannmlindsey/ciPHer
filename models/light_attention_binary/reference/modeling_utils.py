import torch
from torch import nn
from torch.nn.utils.weight_norm import weight_norm


class SimpleMLP(nn.Module):

	def __init__(
		self,
		in_dim: int,
		hid_dim: int,
		out_dim: int,
		dropout: float = 0.1,
	):
		super().__init__()
		self.mlp = nn.Sequential(
			weight_norm(nn.Linear(in_dim, hid_dim), dim=None),
			nn.ReLU(),
			nn.Dropout(dropout),
			weight_norm(nn.Linear(hid_dim, out_dim), dim=None)
		)

	def forward(self, x):
		return self.mlp(x)


class MeanPooling(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, embeddings, masks=None):
		if masks is None:
			extended_masks = torch.ones_like(embeddings)
		else:
			extended_masks = masks.unsqueeze(-1).expand_as(embeddings)
		embeddings = embeddings * extended_masks

		return torch.sum(embeddings, dim=1) / extended_masks.sum(dim=1)


class ConvolutionalAttention(nn.Module):

	def __init__(
		self,
		embed_dim: int = 1280,
		width: int = 9,
	):
		super().__init__()

		self._ca_w1 = nn.Conv1d(embed_dim, embed_dim, width, padding=width//2)
		self._ca_w2 = nn.Conv1d(embed_dim, embed_dim, width, padding=width//2)
		self._ca_mlp = nn.Linear(embed_dim*2, embed_dim)

	def forward(self, embeddings, masks=None, return_attns=False):
		"""
		embeddings: Tensor of shape [B, L, D]
		masks: Tensor of shape [B, L] with 1 for real residues, 0 for padded positions
		"""
		# Change shape to [B, D, L] for Conv1d
		_temp = embeddings.permute(0, 2, 1)    # [B, D, L]
		a = self._ca_w1(_temp)                 # [B, D, L]
		v = self._ca_w2(_temp)                 # [B, D, L]

		if masks is not None:
			# masks: [B, L] -> [B, 1, L] for broadcasting
			attn_mask = masks.unsqueeze(1).to(dtype=a.dtype)
			# Set masked positions to a large negative value before softmax
			a = a.masked_fill(attn_mask == 0, float('-inf'))
		a = a.softmax(dim=-1)  # attention weights over sequence length

		# Compute attention-weighted sum
		v_sum = (a * v).sum(dim=-1)  # [B, D]

		# Max pooling, with masking
		if masks is not None:
			# masked v: set padding positions to a very small value
			v_masked = v.masked_fill(attn_mask == 0, float('-1e4'))
			v_max = v_masked.max(dim=-1).values  # [B, D]
		else:
			v_max = v.max(dim=-1).values         # [B, D]

		# Concatenate and apply Linear
		pooled_output = self._ca_mlp(torch.cat([v_max, v_sum], dim=1))  # [B, D]

		if return_attns:
			return pooled_output, a
		return pooled_output
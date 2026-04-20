from typing import Union
from pathlib import Path

from torch import nn
from torch.nn.utils.weight_norm import weight_norm
from transformers import EsmModel, EsmTokenizer

from .utils.modeling_utils import SimpleMLP, MeanPooling, ConvolutionalAttention


class AbstractModel(nn.Module):

	def __init__(self):
		super().__init__()

	def forward(self, x, masks=None):
		""" forward pass """

		raise NotImplementedError

	def __str__(self):
		""" print model summary (model, trainable parameters) """

		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		return super().__str__() + f"\nTrainable parameters: {params}"


class SequenceClassificationModel(AbstractModel):

	def __init__(
		self,
		embed_dim: int,
		num_labels: int,
		pooler: str = 'ConvAttn',
		pooler_cnn_width: int = 9,
		**kwargs,
	):
		super().__init__()

		if num_labels < 2:
			raise ValueError("Regression model is not supported yet.")
		elif num_labels == 2:
			num_labels = 1

		self.num_labels = num_labels
		self.pooler = MeanPooling() if pooler == 'MeanPooling' else ConvolutionalAttention(embed_dim, width=pooler_cnn_width)
		self.classifier = SimpleMLP(embed_dim, int(embed_dim/2), num_labels)

	def forward(self, embed_tokens, input_masks=None, return_embeds=False, return_attns=False):
		pooled_output = self.pooler(embed_tokens, input_masks, return_attns=return_attns)
		if return_attns:
			pooled_output, attns = pooled_output

		output = self.classifier(pooled_output)
		ret = (output, )
		if return_embeds:
			ret = ret + (pooled_output, )
		if return_attns:
			ret = ret + (attns, )

		return ret
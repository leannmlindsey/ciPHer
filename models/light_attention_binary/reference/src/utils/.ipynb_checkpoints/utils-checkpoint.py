from typing import Sequence, Tuple, List, Optional
from loguru import logger
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence


def embed_seqs(seqs, tokenizer, esm_model, cpu=False, print_example=False):
	try:
		batch_tokens = tokenizer(
			seqs,
			padding=True,
			truncation=False,
			return_tensors="pt",
		)  # [B, L]
		if print_example:
			logger.debug(f"tokens: {batch_tokens[:2]['input_ids']}")

		# Check validity
		if not batch_tokens or 'input_ids' not in batch_tokens or batch_tokens.input_ids.numel() == 0:
			logger.warning(f"Tokenizer produced empty batch_tokens for batch {batch_idx}. Skipping.")
			return (None, None)

		# Embed tokens and get batch_masks
		batch_masks = torch.tensor([
			tokenizer.get_special_tokens_mask(
				ids.tolist(),
				already_has_special_tokens=True
			)
			for ids in batch_tokens.input_ids
		], dtype=torch.long)  # [B, L]
		batch_masks = (batch_masks == 0).long()

		if not cpu and torch.cuda.is_available():
			batch_tokens = {k: v.cuda() for k, v in batch_tokens.items()}

		embed_tokens = esm_model(**batch_tokens)  # [B, L, D]
		embed_tokens = embed_tokens.last_hidden_state
		if print_example:
			logger.debug(f"embed_tokens.dtype: {embed_tokens.dtype}")

		# remove <cls> token
		embed_tokens = embed_tokens[:, 1:, :]
		batch_masks = batch_masks[:, 1:]
	except torch.cuda.OutOfMemoryError as e:
		logger.error(f"CUDA out of memory: {e}")
		torch.cuda.empty_cache()
		raise

	return embed_tokens, batch_masks


def shuffle_seq(seq):
    seq_var = list(seq.replace('J', 'L'))
    np.random.shuffle(seq_var)

    return ''.join(seq_var)


def reverse_seq(seq):
	reverse_seq = seq.replace('J', 'L')[::-1]

	return reverse_seq


class FastaBatchConverter(object):
	"""
	Callable to convert an unprocessed (labels + sequences) batch to
	a processed (labels + tensor) batch
	"""

	def __init__(self, tokenizer, truncation_token_length: Optional[int] = None):
		self.tokenizer = tokenizer
		self.truncation_token_length = truncation_token_length

	def __call__(self, raw_batch: Sequence[Tuple[str, np.ndarray]]):
		# raw_batch: [(label, np.array(T_i, D)), ...]

		# labels / embeds
		batch_labels, seqs = zip(*raw_batch)
		batch_tokens = self.tokenizer(
			seqs,
			padding=True,
			truncation=True,
			max_length=self.truncation_token_length,
			return_tensors="pt"
		)
		batch_labels = list(batch_labels)

		return batch_labels, batch_tokens

class EmbeddedBatchConverter(object):
	"""
	Callable to convert an unprocessed (labels + tokens) batch to
	a processed (labels + tensor) batch.
	"""

	def __init__(self, truncation_token_length: Optional[int] = None):
		self.truncation_token_length = truncation_token_length

	def __call__(self, raw_batch: Sequence[Tuple[str, np.ndarray]]):
		# raw_batch: [(label, np.array(T_i, D)), ...]

		# labels / embeds
		batch_labels, embed_token_list = zip(*raw_batch)

		# truncation [optional]
		if self.truncation_token_length is not None:
			embed_token_list = [
				tokens[:self.truncation_token_length, :]
				for tokens in embed_token_list
			]

		# np.ndarray -> torch.Tensor
		embed_tensors: List[torch.Tensor] = [
			torch.from_numpy(arr).float() for arr in embed_token_list
		]

		# padding into [B, T_max, D]
		batch_tokens = pad_sequence(embed_tensors, batch_first=True)
		batch_labels = list(batch_labels)

		return batch_labels, batch_tokens
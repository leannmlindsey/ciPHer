from typing import Union, Callable, Dict, Optional
from pathlib import Path
from loguru import logger
import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import SequentialSampler, RandomSampler, WeightedRandomSampler
from transformers import EsmModel, EsmTokenizer
import pytorch_lightning as pl

from .utils.utils import embed_seqs
from .utils.constants import MODEL_PATH, ESM_PATH, ESM_MODELS
from .models import SequenceClassificationModel
from src import datasets as D


class TrainModel(pl.LightningModule):

	def __init__(
		self,
		data_type: str = 'fasta',
		lr: float = 5e-4,
		reduction: str = 'mean',
		sync_dist: bool = False,
		pretrained: Optional[str] = None,
		cache_path: Optional[Union[str, Path]] = None,
		embed_dim: int = 1280,
		**kwargs,
	):
		super().__init__()
		if data_type == 'fasta':
			model_name = ESM_MODELS[pretrained]
			self.tokenizer = EsmTokenizer.from_pretrained(model_name, cache_dir=cache_path)
			self.esm_model = EsmModel.from_pretrained(model_name, cache_dir=cache_path)
			self.esm_model.requires_grad_(False)
		self.classifier = SequenceClassificationModel(
			embed_dim=self.esm_model.config.hidden_size if data_type == 'fasta' else embed_dim,
			**kwargs
		)
		self.data_type = data_type
		self.lr = lr
		self.reduction = reduction
		self.sync_dist = sync_dist

	def forward(self, batch_tokens, batch_masks=None):
		if self.data_type == 'fasta':
			embed_tokens, batch_masks = embed_seqs(
				batch_tokens,
				self.tokenizer,
				self.esm_model
			)
		else:
			embed_tokens = batch_tokens
		outputs = self.classifier(embed_tokens, input_masks=batch_masks)

		return outputs[0]

	def __shared_step(self, batch, metrics: Dict[str, Callable] = None):
		labels, batch_tokens = batch
		if self.data_type == 'fasta':
			batch_masks = None
		else:
			batch_masks = (batch_tokens.abs().sum(dim=-1) != 0).int()

		logits = self(batch_tokens, batch_masks)
		loss_fn = nn.BCEWithLogitsLoss(reduction='none')
		if self.classifier.num_labels == 1:
			labels = labels.unsqeeze(1).type(torch.float32)
			loss = loss_fn(logits, labels)
			loss = loss.mean() if self.reduction == 'mean' else loss.sum()
		else:
			is_decoy = (labels == -1)
			is_real = ~is_decoy

			real_labels = labels[is_real]
			real_y = F.one_hot(real_labels, num_classes=self.classifier.num_labels).float()

			decoy_y = torch.zeros((is_decoy.sum(), self.classifier.num_labels), device=logits.device)

			y = torch.zeros((labels.size(0), self.classifier.num_labels), device=logits.device)
			y[is_real] = real_y
			y[is_decoy] = decoy_y

			loss = loss_fn(logits, y).sum(dim=1).mean()

		if metrics is not None:
			scores = {}
			for metric, metric_fn in metrics.items():
				value = metric_fn(logits, labels)
				scores[metric] = value

			return loss, scores

		return loss

	def training_step(self, batch, batch_idx):
		loss = self.__shared_step(batch)

		batch_size = batch[0].shape[0]
		self.log(
			'train_loss', loss, batch_size=batch_size,
			on_step=True, on_epoch=True, prog_bar=True,
			logger=True, sync_dist=self.sync_dist
		)

		return loss

	def validation_step(self, batch, batch_idx):
		loss = self.__shared_step(batch)

		batch_size = batch[0].shape[0]
		self.log(
			'valid_loss', loss, batch_size=batch_size,
			on_epoch=True, prog_bar=True,
			logger=True, sync_dist=self.sync_dist,
		)

		return loss

	def test_step(self, batch, batch_idx):
		loss = self.__shared_step(batch)

		batch_size = batch[0].shape[0]
		self.log(
			'test_loss', loss, batch_size=batch_size,
			prog_bar=True,
			logger=True, sync_dist=self.sync_dist,
		)

		return loss

	def configure_optimizers(self):
		optimizer = AdamW(
			self.parameters(),
			lr=5e-4,
			weight_decay=0.01,
			eps=1e-6,
			betas=(0.9, 0.999),
		)
		return optimizer


def get_data_loader(
	dataset: Dataset,
	batch_size: int,
	num_workers: int,
	weighted: bool = False,
	shuffle: bool = True,
):
	if weighted:
		labels = dataset.get_labels()
		class_sample_count = np.array(
			[len(np.where(labels == t)[0]) for t in np.unique(labels)]
		)
		weight = 1. / class_sample_count

		samples_weight = np.array([weight[t] for t in labels])
		samples_weight = torch.from_numpy(samples_weight)
		sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
	elif shuffle:
		sampler = RandomSampler(dataset)
	else:
		sampler = SequentialSampler(dataset)

	data_loader = DataLoader(
		dataset=dataset,
		batch_size=batch_size,
		sampler=sampler,
		num_workers=num_workers,
		collate_fn=dataset.collate_fn,
	)

	return data_loader


class DataModule(pl.LightningDataModule):
	""" 
	Data module for training classifier
	Args:
		data_dir (Union[str, Path]): Path to the data directory
		prefix (str): Prefix of the data files
		batch_size (int): Batch size
	"""

	def __init__(
		self,
		data_file: Union[str, Path],
		label_path: Union[str, Path],
		prefix: str,
		data_type: str = 'fasta',
		batch_size: int = 1,
		num_workers: int = 1,
		in_memory: bool = False,
	):
		super().__init__()
		self.data_type = data_type
		self.dataset_cls = getattr(D, 'PHROGDataSet')
		self.data_file = Path(data_file)
		self.label_path = Path(label_path)
		self.prefix = prefix
		self.batch_size = batch_size
		self.num_workers = num_workers
		self.in_memory = in_memory

		self.raw_data = None
		self.trainset = None
		self.validset = None
		self.testset = None

	def setup(self, stage: str):
		prefix = self.label_path / self.prefix

		if self.data_type not in ['fasta', 'embedded']:
			raise ValueError(
				f"Invalid data_type value: {self.data_type} ('fasta', 'embedded')"
			)

		data_cls = (
			getattr(D, "FastaDataset")
			if self.data_type == 'fasta' else
			getattr(D, "EmbeddedDataset")
		)
		self.raw_data = data_cls(self.data_file, in_memory=self.in_memory)

		self.trainset = self.dataset_cls(
			dataset=self.raw_data,
			label_file=f"{prefix}.train.tsv",
			data_type=self.data_type,
		)
		self.validset = self.dataset_cls(
			dataset=self.raw_data,
			label_file=f"{prefix}.valid.tsv",
			data_type=self.data_type,
		)
		self.testset = self.dataset_cls(
			dataset=self.raw_data,
			label_file=f"{prefix}.test.tsv",
			data_type=self.data_type,
		)

		logger.info(f"Number of train sequences: {len(self.trainset)}")
		logger.info(f"Number of validation sequences: {len(self.validset)}")
		logger.info(f"Number of test sequences: {len(self.testset)}")

	def train_dataloader(self):
		return get_data_loader(
			self.trainset,
			self.batch_size,
			num_workers=self.num_workers,
			weighted=True,
		)

	def val_dataloader(self):
		return get_data_loader(
			self.validset,
			self.batch_size,
			num_workers=self.num_workers,
			shuffle=False,
		)

	def test_dataloader(self):
		return get_data_loader(
			self.testset,
			self.batch_size,
			num_workers=self.num_workers,
			shuffle=False,
		)
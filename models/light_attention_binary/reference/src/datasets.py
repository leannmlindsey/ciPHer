import time
import h5py
import numpy as np
import pandas as pd
from typing import Union
from pathlib import Path
from loguru import logger

import torch
from torch.utils.data import Dataset, DataLoader

from .utils.constants import PHROG_CLASSES, CLASS_IDS
from .utils.utils import FastaBatchConverter, EmbeddedBatchConverter


class FastaDataset(Dataset):
	"""
	Read sequences from a fasta file
	Args:
		fasta_file (str): path to fasta file
		in_memory (bool): load all sequences into memory
			Default: False
	"""

	def __init__(
		self,
		fasta_file: Union[str, Path],
		in_memory: bool = False,
	):
		""" initialize dataset """

		fasta_file = Path(fasta_file)
		if not fasta_file.exists():
			raise FileNotFoundError(f"File not found: {fasta_file}")

		self.file_name = fasta_file
		self._cache = None
		self._data = None

		if in_memory:
			__file_stream = open(fasta_file, 'r')
			self._cache = []
			self._data = {}

			record = self.__read_one_seq(__file_stream)
			while record is not None:
				self._cache.append(record['id'])
				self._data[record['id']] = record
				record = self.__read_one_seq(__file_stream)
			__file_stream.close()
		else:
			self.__indexing()

		self._in_memory = in_memory
		self._num_examples = len(self._cache)

	def __len__(self):
		""" return the number of sequences """
		return self._num_examples

	def __getitem__(self, key: Union[int, str]):
		if isinstance(key, int):
			""" return sequence at index """
			if not 0 <= key < self._num_examples:
				raise IndexError(f"Index out of range: {key}")

			if self._in_memory:
				return self._data[self._cache[key]]
			else:
				__file_stream = open(self.file_name, 'r')
				__file_stream.seek(self._data[self._cache[key]])
				return self.__read_one_seq(__file_stream)
		elif isinstance(key, str):
			""" return sequence at sid """
			if key not in self._data:
				raise KeyError(f"KeyError: '{key}'")

			if self._in_memory:
				return self._data[key]
			else:
				__file_stream = open(self.file_name, 'r')
				__file_stream.seek(self._data[key])
				return self.__read_one_seq(__file_stream)

	def __read_one_seq(self, __file_stream):
		header = __file_stream.readline().strip()
		if not header:
			return None

		sid = header.split(' ')[0][1:]
		seq = ""
		while True:
			prev = __file_stream.tell()
			line = __file_stream.readline().strip()
			if not line or line.startswith('>'):
				__file_stream.seek(prev)
				break
			seq += line

		return {
			"id": sid,
			"header": header[1:],
			"seq": seq,
		}

	def __indexing(self):
		__file_stream = open(self.file_name, 'r')
		self._cache = []
		self._data = {}

		prev = __file_stream.tell()
		while True:
			line = __file_stream.readline()
			if not line:
				break

			if line.startswith('>'):
				key = line.strip()[1:].split(' ')[0]
				self._cache.append(key)
				self._data[key] = prev

			prev = __file_stream.tell()
		__file_stream.close()

	def collate_fn(self, batch):
		"""
		batch: [{'id': ..., 'header': ..., 'seq': ...}, ...]
		"""
		headers = [b["header"] for b in batch]
		seqs = [b["seq"] for b in batch]

		return headers, seqs


class EmbeddedDataset(Dataset):
	"""
	Read h5py format file
	Args:
		h5_file (str): path to embedded file
		in_memory (bool): load all data into memory
			Default: False
	"""

	def __init__(
		self,
		h5_file: Union[str, Path],
		in_memory: bool = False,
	):
		""" initialize dataset """

		h5_file = Path(h5_file)
		if not h5_file.exists():
			raise FileNotFoundError(f"File not found: {h5_file}")
		self.file_name = h5_file
		self.h5_stream = None

		self._cache = None
		self._data = None

		stime = time.time()
		if in_memory:
			print("Start loading data on memory...")
			self.__load_on_memory()
		else:
			print("Start caching keys...")
			with h5py.File(h5_file, 'r') as f:
				self._cache = list(f.keys())
		etime = time.time()
		logger.info(f"Loading/Caching DONE ({etime - stime:.2f} secs)")

		self._in_memory = in_memory
		self._num_examples = len(self._cache)
		self.batch_converter = EmbeddedBatchConverter()

	def __del__(self):
		if self.h5_stream:
			self.h5_stream.close()

	def __len__(self):
		""" return the number of vectors """
		return self._num_examples

	def __getitem__(self, key: Union[int, str]):
		if isinstance(key, int):
			""" return vector at index """
			if not 0 <= key < self._num_examples:
				raise IndexError(f"Index out of range: {key}")

			pid = self._cache[key]
			return self.__get_data(pid)
		elif isinstance(key, str):
			""" return vector at key """

			return self.__get_data(key)

	def __get_data(self, pid):
		if self._in_memory:
			return self._data[pid]
		else:
			if self.h5_stream is None:
				self.h5_stream = h5py.File(self.file_name, 'r')

			return {
				"id": pid,
				"embed_tokens": self.h5_stream[pid][::]
			}

	def __load_on_memory(self):
		with h5py.File(self.file_name, 'r') as h5_stream:
			self._cache = []
			self._data = {}

			for k in h5_stream:
				self._cache.append(k)
				self._data[k] = {
					"id": k,
					"embed_tokens": h5_stream[k][::]
				}

	def collate_fn(self, batch):
		"""
		batch: [{'id': ..., 'embed_tokens': ...}, ...]
		"""
		batch = [(item['id'], item['embed_tokens']) for item in batch]
		ids, embed_tokens = self.batch_converter(batch)

		return ids, embed_tokens


class SubDataset(Dataset):
	"""
	Dataset class to parse subset of the original dataset
	Args:
		dataset (Dataset): the original dataset object
		subset (Union[str, Path]): subset list containing a list of ids
	"""
	def __init__(
		self,
		dataset: Union[FastaDataset, EmbeddedDataset],
		subset: Union[str, Path],
	):
		""" initialize dataset """

		self._data = dataset
		self._subset = pd.read_csv(subset, sep='\t')['pid'].to_numpy()

	def __len__(self):
		return self._subset.shape[0]

	def __getitem__(self, idx: int):
		if not 0 <= idx < self._subset.shape[0]:
			raise IndexError(f"Index out of range: {idx}")

		pid = self._subset[idx]
		return self._data[pid]

	def collate_fn(self, batch):
		return self._data.collate_fn(batch)


class PHROGDataSet(Dataset):
	"""
	Read dataset for PHROG functional category classfier
	Args:
		dataset (Union[FastaDataset, EmbeddedDataset]): the ofiginal dataset object
		label_file (Union[str, Path]): path to label file
		data_type (str): type of dataset [fasta, embedded]
			Default: fasta
		in_memory (bool): load all data into memory
			Default: False
	"""
	def __init__(
		self,
		dataset: Union[FastaDataset, EmbeddedDataset],
		label_file: Union[str, Path],
		data_type: str = "fasta",
		tokenizer = None,
	):
		""" initialize dataset """

		if data_type not in ['fasta', 'embedded']:
			raise ValueError(
				f"Invalid data_type value: {data_type} ('fasta', 'embedded')"
			)
		elif data_type == 'fasta' and tokenizer is None:
			raise ValueError(
				f"Tokenizer object should be passed for the fasta datasets"
			)

		label_file = Path(label_file)
		if not label_file.exists():
			raise FileNotFoundError(f"File not found: {label_file}")

		self._data = dataset
		self.data_type = data_type
		self._labels = pd.read_csv(label_file, sep='\t')
		self.tokenizer = tokenizer

		if data_type == "fasta":
			self.batch_converter = FastaBatchConverter(tokenizer)
		else:
			self.batch_converter = EmbeddedBatchConverter()

	def __len__(self):
		return self._labels.shape[0]

	def __getitem__(self, idx: int):
		_row = self._labels.iloc[idx]

		pid = _row['pid']
		record = self._data[pid]
		if self.data_type == "fasta":
			data = record['seq']
		elif self.data_type == "embedded":
			data = record['embed_tokens']

		label = _row['label']
		label = CLASS_IDS[label] if label != 'decoy' else -1

		return label, data

	def collate_fn(self, batch):
		labels, batch_tokens = self.batch_converter(batch)
		return torch.LongTensor(labels), batch_tokens

	def get_labels(self):
		return np.array([CLASS_IDS[label] if label in CLASS_IDS else -1 for label in self._labels['label']])

	@classmethod
	def get_num_classes(cls):
		return len(PHROG_CLASSES)

	@classmethod
	def convert_id_to_label(cls, idx: int, labels: dict = None):
		if not 0 <= idx < len(PHROG_CLASSES):
			raise IndexError(f"Index out of range: {idx}")

		label = PHROG_CLASSES[idx]
		label = labels[label] if labels is not None else label
		return label
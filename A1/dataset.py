from typing import List, Dict

from torch.utils.data import Dataset
import torch

from utils import Vocab, pad_to_len
from zmq import NULL


class SeqClsDataset(Dataset):
    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {idx: intent for intent, idx in self.label_mapping.items()}
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO: implement collate_fn
        # aggregate all single dict to form a general dict and embed all text to code with "function defined"
        all_text = []
        all_intent = []
        all_id = []
        for dic in samples:
          all_text.append(dic["text"].split( ))
          if("intent" in dic.keys()):
            all_intent.append(self.label_mapping[dic["intent"]])
          all_id.append(dic["id"])
        return {
          "text": torch.Tensor(self.vocab.encode_batch(all_text, self.max_len)), 
          "intent": torch.LongTensor(all_intent), 
          "id": all_id
        }



    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]

class SeqTaggingClsDataset(SeqClsDataset):
    def collate_fn(self, samples):
        all_tokens = []
        all_tags = []
        all_id = []
        all_length = []
        for dic in samples:
            all_tokens.append(dic["tokens"])
            length = len(dic["tokens"])
            if("tags" in dic.keys()):
                tags = []
                for tag in dic["tags"]:
                    tags.append(self.label_mapping[tag])
                # tags = pad_to_len(tags, self.max_len, 0)
                # for i in range(length, self.max_len):
                #     tags.append(self.label_mapping["O"])
                all_tags.append(tags)
            all_length.append(length)
            all_id.append(dic["id"])
        self.max_len = max(all_length)
        return {
            "tokens": torch.Tensor(self.vocab.encode_batch(all_tokens, self.max_len)),
            "tags": torch.LongTensor(pad_to_len(all_tags, self.max_len, self.label_mapping["O"])),
            "id": all_id, 
            "length": all_length
        }
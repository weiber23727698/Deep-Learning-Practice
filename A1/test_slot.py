import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from random import randrange
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    testLoader = DataLoader(dataset, args.batch_size, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # load weights into model
    model = torch.load(args.ckpt_path)
    
    model.eval()

    # TODO: predict dataset
    ID = []
    labels = []
    lengths = []

    for i, batch in enumerate(testLoader):
        batch["tokens"] = batch["tokens"].to(args.device, dtype=torch.long)
        ID += batch["id"]
        lengths += batch["length"]
        outputs = model(batch)
        outputs = torch.stack(tuple(outputs.values()), dim = 0).float()
        labels += torch.argmax(outputs, dim=-1).squeeze().tolist()

    # TODO: write prediction to file (args.pred_file)

    with open(args.pred_file, 'w') as f:
        f.write('id,tags\n')
        for i, label, length in zip(ID, labels, lengths):
            f.write("%s," %(i))
            for j in range(length):
                if(j != length-1):
                    f.write("%s " %(dataset.idx2label(label[j])))
                else:
                    f.write("%s\n" %(dataset.idx2label(label[j])))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
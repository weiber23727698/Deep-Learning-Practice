import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader #todo

def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    testLoader = DataLoader(dataset, args.batch_size, collate_fn=dataset.collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model = ckpt
    model.eval()

    # TODO: predict dataset
    ID = []
    labels = []

    for i, batch in enumerate(testLoader):
        batch["text"] = batch["text"].to(args.device, dtype=torch.long)
        ID += batch["id"]
        outputs = model(batch)
        outputs = torch.stack(tuple(outputs.values()), dim = 0).float()
        labels += torch.argmax(outputs, dim=-1).squeeze().tolist()

    # TODO: write prediction to file (args.pred_file)

    with open(args.pred_file, 'w') as f:
        f.write('id,intent\n')
        for i, label in zip(ID, labels):
            f.write("%s,%s\n" %(i, dataset.idx2label(label)))


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

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

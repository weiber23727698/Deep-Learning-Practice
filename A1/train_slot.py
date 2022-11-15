import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm, trange
from seqeval.metrics import classification_report, f1_score
from seqeval.scheme import IOB2
import seqeval

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader #todo

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def evaluation(outputs, label):
    count = 0
    outputs = torch.argmax(outputs, dim = -1)
    for x in range(len(outputs)):
      a = torch.sum(torch.ne(outputs[x], label[x])).tolist()
      if(a == 0):
        count += 1
    return count

def evaluation2(lengths, outputs, label):
    all = 0 # all tags
    count = 0 # correct tags
    outputs = torch.argmax(outputs, dim = -1)
    for i in range(len(outputs)):
        for j in range(lengths[i]):
            all += 1
            if(outputs[i][j] == label[i][j]):
                count += 1
    return all, count


def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    trainLoader = DataLoader(datasets[TRAIN], args.batch_size, collate_fn=datasets[TRAIN].collate_fn)
    devLoader = DataLoader(datasets[DEV], args.batch_size, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes
    ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.8)

    model.train()
    t_batch = len(trainLoader) 
    v_batch = len(devLoader)
    all_loss, all_acc= 0, 0
    best_acc = 0.0
    loss_fn = torch.nn.BCELoss()
    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    y_true, y_pred = [], []
    final_token_acc = 0.0
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        all_loss, all_acc = 0, 0
        for i, batch in enumerate(trainLoader):
            batch["tokens"] = batch["tokens"].to(args.device, dtype=torch.long)
            optimizer.zero_grad() # clear gradient
            outputs = model(batch)
            outputs = torch.stack(tuple(outputs.values()))
            outputs = outputs.squeeze()
            batch["tags"] = batch["tags"].to(args.device)
            para = torch.nn.functional.one_hot(batch["tags"], datasets[TRAIN].num_classes).detach().float()
            loss1 = loss_fn(outputs, para) # calculate training loss
            loss2 = loss_fn(outputs*para, para) * datasets[TRAIN].num_classes
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, batch["tags"])
            all_acc += (correct / args.batch_size)
            all_loss += loss.item()
        scheduler.step()
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(all_loss/t_batch, all_acc/t_batch*100))
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        fuck = 0
        with torch.no_grad():
            all_loss, all_acc = 0, 0
            all_tokens, correct_tokens = 0, 0
            tokens_acc = 0.0
            tmp_true, tmp_pred = [], []
            for i, batch in enumerate(devLoader):
                batch["tokens"] = batch["tokens"].to(args.device, dtype=torch.long)
                outputs = model(batch)
                outputs = torch.stack(tuple(outputs.values()))
                outputs = outputs.squeeze()
                batch["tags"] = batch["tags"].to(args.device)
                para = torch.nn.functional.one_hot(batch["tags"], datasets[TRAIN].num_classes).detach().float()
                # joint accuracy
                correct = evaluation(outputs, batch["tags"])
                all_acc += (correct / args.batch_size)
                # tokens accuracy
                all_tokens, correct_tokens = evaluation2(batch["length"], outputs, batch["tags"])
                tokens_acc += correct_tokens / all_tokens
                # seqeval
                each_true, each_pred = [], []
                outputs = torch.argmax(outputs, dim = -1).tolist()
                for i in range(len(outputs)):
                    each_pred = []
                    for j in range(batch["length"][i]):
                        each_pred.append(datasets[DEV].idx2label(outputs[i][j]))
                    tmp_pred.append(each_pred)
                for i in range(len(batch["tags"])):
                    each_true = []
                    for j in range(batch["length"][i]):
                        each_true.append(datasets[DEV].idx2label(batch["tags"][i].tolist()[j]))
                    tmp_true.append(each_true)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(all_loss/v_batch, all_acc/v_batch*100))
            if all_acc > best_acc:
                best_acc = all_acc
                final_token_acc = tokens_acc
                torch.save(model, "{}/ckpt.model".format(args.ckpt_dir))
                print('saving model with acc {:.3f}'.format(all_acc/v_batch*100))
                y_true = tmp_true
                y_pred = tmp_pred
        print('-----------------------------------------------')
        model.train()
    
    print("hidden_size: %d" %(args.hidden_size))
    print("num_layers: %d" %(args.num_layers))
    print("dropout: %.2lf" %(args.dropout))
    print("learning rate: %.4lf" %(args.lr))
    print("batch_size: %d" %(args.batch_size))
    print("max_len: %d" %(args.max_len))
    print("best joint acc: %.4f" %(best_acc/v_batch*100))
    print("best tokens acc: %.4f" %(final_token_acc/v_batch*100))
    print(v_batch)
    print("f1 score: %.4f" %(f1_score(y_true, y_pred)))
    print(classification_report(y_true, y_pred, scheme=IOB2, mode="strict"))


def parse_args() -> Namespace:
    parser = ArgumentParser()
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

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=100)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
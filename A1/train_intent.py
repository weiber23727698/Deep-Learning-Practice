import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader #todo
from torch.optim import lr_scheduler
from tqdm import trange

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]

def evaluation(outputs, labels):
    # print(labels.shape)
    #print(torch.argmax(outputs, dim = -1).shape)
    return torch.sum(torch.eq(torch.argmax(outputs, dim = -1), labels).float()).item()

def main(args):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    trainLoader = DataLoader(datasets[TRAIN], args.batch_size, shuffle = True, collate_fn=datasets[TRAIN].collate_fn)
    devLoader = DataLoader(datasets[DEV], args.batch_size, shuffle = False, collate_fn=datasets[DEV].collate_fn)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes
    ).to(args.device)

    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)

    model.train()
    t_batch = len(trainLoader) 
    v_batch = len(devLoader)
    all_loss, all_acc= 0, 0
    best_acc = 0.0
    loss_fn = torch.nn.BCELoss()

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        all_loss, all_acc = 0, 0
        for i, labels in enumerate(trainLoader):
            labels["text"] = labels["text"].to(args.device, dtype=torch.long)
            optimizer.zero_grad() # clear gradient
            outputs = model(labels)
            outputs = torch.stack(tuple(outputs.values()))
            outputs = outputs.squeeze()
            labels["intent"] = labels["intent"].to(args.device)
            para = torch.nn.functional.one_hot(labels["intent"], datasets[TRAIN].num_classes).detach().float()
            loss1 = loss_fn(outputs, para)
            loss2 = loss_fn(outputs*para, para) * datasets[TRAIN].num_classes
            loss = loss1 + loss2
            loss.backward()
            optimizer.step()
            correct = evaluation(outputs, labels["intent"])
            all_acc += (correct / args.batch_size)
            all_loss += loss.item()
        print('\nTrain | Loss:{:.5f} Acc: {:.3f}'.format(all_loss/t_batch, all_acc/t_batch*100))
        # TODO: Evaluation loop - calculate accuracy and save model weights
        scheduler.step()
        model.eval()
        with torch.no_grad():
            all_loss, all_acc = 0, 0
            for i, labels in enumerate(devLoader):
                labels["text"] = labels["text"].to(args.device, dtype=torch.long)
                outputs = model(labels)
                outputs = torch.stack(tuple(outputs.values()))
                outputs = outputs.squeeze()
                labels["intent"] = labels["intent"].to(args.device)
                para = torch.nn.functional.one_hot(labels["intent"], datasets[TRAIN].num_classes).detach().float()         
                correct = evaluation(outputs, labels["intent"])
                all_acc += (correct / args.batch_size)
            print("Valid | Loss:{:.5f} Acc: {:.3f} ".format(all_loss/v_batch, all_acc/v_batch*100))
            if all_acc/v_batch*100 > best_acc:
                best_acc = all_acc/v_batch*100
                torch.save(model, "{}/ckpt.model".format(args.ckpt_dir))
                print('saving model with acc {:.3f}'.format(all_acc/v_batch*100))
        print('-----------------------------------------------')
        model.train()
    
    print("hidden_size: %d" %(args.hidden_size))
    print("num_layers: %d" %(args.num_layers))
    print("dropout: %.2lf" %(args.dropout))
    print("learning rate: %.4lf" %(args.lr))
    print("batch_size: %d" %(args.batch_size))
    print("max_len: %d" %(args.max_len))
    print(best_acc)

    


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )

    # data
    parser.add_argument("--max_len", type=int, default=32)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
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

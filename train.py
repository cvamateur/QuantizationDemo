import os

import torch
import torch.optim

from tqdm import tqdm

from common.cli import get_parser
from common.net import MNIST_Net
from common.dataloader import get_mnist_dataloader

MODULE_PLAN = (
    16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128,
)

USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")

best_loss: float = float("inf")
best_acc: float = -0.0
best_ckpt: str = "./ckpt/best.pth"
best_ckpt_info: str = "./ckpt/best.info"
if os.path.exists(best_ckpt_info):
    os.system(f"rm {best_ckpt_info}")

def train_step(epoch, model, dataset, criterion, optimizer, args):
    model.train()

    total_loss: float = 0.0
    total_correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Training  : {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        logits = model(images)
        loss = criterion(logits, labels)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # calc loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)
    return total_loss, acc


@torch.no_grad()
def eval_step(epoch, model, dataset, criterion, args):
    model.eval()

    total_loss: float = 0.0
    total_correct: int = 0
    acc: float = 0.0
    progress_bar = tqdm(dataset, desc=f"Validation: {epoch}/{args.num_epochs}")
    for i, (images, labels) in enumerate(dataset):
        if USE_GPU:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

        # forward
        logits = model(images)
        loss = criterion(logits, labels)

        # calc loss
        total_loss += loss.item() / len(dataset)
        avg_loss = total_loss * len(dataset) / (i + 1)

        # metric
        preds = logits.argmax(dim=1)
        total_correct += torch.eq(preds, labels).sum().item()
        acc = total_correct / ((i + 1) * args.batch_size) * 100

        # Log info
        info_str = "loss: {loss:.4f}, acc: {acc:.1f}%".format(loss=avg_loss, acc=acc)
        progress_bar.set_postfix_str(info_str)
        if (i + 1) % args.log_freq == 0:
            progress_bar.update(args.log_freq)

    progress_bar.update(len(dataset) - progress_bar.n)
    return total_loss, acc


def save_best_ckpt(model, epoch, loss, acc):
    global best_loss, best_acc
    if loss < best_loss or acc > best_acc:
        best_loss = loss
        best_acc = acc
        with open(best_ckpt_info, 'w') as f:
            f.write(f"Epoch: {epoch}, Loss: {loss:.4f}, Acc: {acc: .2f}%\n")
        torch.save(model.state_dict(), best_ckpt)
        print("info: saved best ckpt:", best_ckpt)


def main(args):
    ds_train, ds_valid = get_mnist_dataloader(args)

    model = MNIST_Net(args.factor).to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(1, args.num_epochs + 1):
        train_step(epoch, model, ds_train, criterion, optimizer, args)
        if epoch >= args.eval_epoch and epoch % args.eval_freq == 0:
            loss, acc = eval_step(epoch, model, ds_valid, criterion, args)
            save_best_ckpt(model, epoch, loss, acc)


if __name__ == '__main__':
    args = get_parser().parse_args()
    main(args)

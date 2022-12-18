from comet_ml import Experiment
import os
import torch
import torch.optim as optim
import torch.nn as nn
import argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(10, 20, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = output.view(output.shape[0], -1)
        output = self.fc(output)
        return output


def main(args):
    if not os.path.exists(args.train_data_dir):
        path: Path = Path(args.train_data_dir)
        path.mkdir(exist_ok=False, parents=True)
    if not os.path.exists(args.val_data_dir):
        path: Path = Path(args.val_data_dir)
        path.mkdir(exist_ok=False, parents=True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(args.train_data_dir, train=True, download=True, transform=transform)
    val_data = datasets.MNIST(args.val_data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.val_batch_size, shuffle=True)

    model = Net()
    model = model.to(device=args.device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    if args.use_comet == True:

        experiment = Experiment(
            api_key=args.api_key,
            project_name=args.project_name,
            workspace=args.workspace
        )

        parameters = {
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "train_batch_size": args.train_batch_size,
            "val_batch_size": args.val_batch_size
        }
        experiment.log_parameters(parameters)

        for epoch in range(1, args.epochs + 1):
            pbar = tqdm(total=len(train_data), desc='Epoch:{}/{}'.format(epoch, args.epochs))
            model.train()
            mean_train_loss = 0
            step = 0
            for steps, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device=args.device)
                labels = labels.to(device=args.device)
                output = model(images)
                train_loss = loss_function(output, labels)
                mean_train_loss += train_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                step = steps
                pbar.update(len(images))
            mean_train_loss = mean_train_loss / step

            model.eval()
            mean_val_loss = 0
            step = 0
            with torch.no_grad():
                for steps, (images, labels) in enumerate(val_loader, start=1):
                    images = images.to(device=args.device)
                    labels = labels.to(device=args.device)
                    output = model(images)
                    val_loss = loss_function(output, labels)
                    mean_val_loss += val_loss.item()
                    step = steps
                mean_val_loss = mean_val_loss / step
                pbar.set_postfix_str('train_loss:{:.6f},val_loss:{:.6f}'.format(mean_train_loss, mean_val_loss))
            metrics = {'train_loss': mean_train_loss,
                       'val_loss': mean_val_loss
                       }
            experiment.log_metrics(metrics, epoch=epoch)
            pbar.close()
    else:
        for epoch in range(1, args.epochs + 1):
            pbar = tqdm(total=len(train_data), desc='Epoch:{}/{}'.format(epoch, args.epochs))
            model.train()
            mean_train_loss = 0
            step = 0
            for steps, (images, labels) in enumerate(train_loader, start=1):
                images = images.to(device=args.device)
                labels = labels.to(device=args.device)
                output = model(images)
                train_loss = loss_function(output, labels)
                mean_train_loss += train_loss.item()
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                step = steps
                pbar.update(len(images))
            mean_train_loss = mean_train_loss / step

            model.eval()
            mean_val_loss = 0
            step = 0
            with torch.no_grad():
                for steps, (images, labels) in enumerate(val_loader, start=1):
                    images = images.to(device=args.device)
                    labels = labels.to(device=args.device)
                    output = model(images)
                    val_loss = loss_function(output, labels)
                    mean_val_loss += val_loss.item()
                    step = steps
                mean_val_loss = mean_val_loss / step
                pbar.set_postfix_str('train_loss:{:.6f},val_loss:{:.6f}'.format(mean_train_loss, mean_val_loss))
            metrics = {'train_loss': mean_train_loss,
                       'val_loss': mean_val_loss
                       }
            pbar.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=2e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--train_batch_size', type=int, default=2048)
    parser.add_argument('--val_batch_size', type=int, default=3000)
    parser.add_argument('--train_data_dir', type=str, default='./data/train/')
    parser.add_argument('--val_data_dir', type=str, default='./data/val/')
    parser.add_argument('--api_key', type=str, default='',help='your_comet_account_api_key')
    parser.add_argument('--workspace', type=str, default='',help='your_workspace')
    parser.add_argument('--project_name', type=str, default='',help='your_project_name')
    parser.add_argument('--use_comet', type=bool, default=False)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)

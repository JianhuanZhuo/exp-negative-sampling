import os
import random
import sys

import torch

from dataset.SRNS_ml1m import SRNSML1MDataset
from evaluator import Evaluator
from tools.config import load_specific_config
from tools.resolvers import model_resolver
from tools.tee import StdoutTee, StderrTee
from torch.optim.adagrad import Adagrad
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from torch.nn import functional as F
from setproctitle import setproctitle
import numpy as np


def wrap(config):
    pid = os.getpid()
    config['pid'] = pid
    print(f"pid is {pid}")

    if 'writer_path' not in config:
        folder = config['log_tag']
        if config["git/state"] == "Good":
            folder += '-%s' % (config['git']['hexsha'][:5])

        config['writer_path'] = os.path.join(config['log_folder'],
                                             folder,
                                             config.postfix()
                                             )
    if not os.path.exists(config['writer_path']):
        os.makedirs(config['writer_path'])

    setproctitle(config['writer_path'])

    if 'logfile' not in config or config['logfile']:
        logfile_std = os.path.join(config['writer_path'], "std.log")
        logfile_err = os.path.join(config['writer_path'], "err.log")
        with StdoutTee(logfile_std, buff=1), StderrTee(logfile_err, buff=1):
            try:
                main_run(config)
            except Exception as e:
                sys.stderr.write(str(e) + "\n")
    else:
        main_run(config)
    exit(0)


def main_run(config):
    # set random seed
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = config['cuda']
    torch.manual_seed(config['seed'])
    random.seed(config['seed'])

    summary = SummaryWriter(config['writer_path'])
    summary.add_text('config', config.__str__())
    print(f"output to {config['writer_path']}")

    dataset = SRNSML1MDataset(config)
    dataloader = DataLoader(
        dataset=dataset,
        **config['DataLoader']
    )

    # 模型定义
    model_cls = model_resolver.lookup(config['model'])
    # model = model_cls(**config.get_or_default("model_args", default={}))
    model = model_cls(user_num=dataset.num_user, item_num=dataset.num_item, config=config)
    print("loading model and assign GPU memory...")
    model = model.cuda()
    print("loaded over.")

    evaluator = Evaluator(config, summary, dataset)
    # 优化器
    optimizer = Adagrad(model.parameters(), **config['optimizer'])
    if config.get_or_default("train/softw_enable", False):
        softw = torch.nn.Parameter(torch.ones([dataset.num_user, 1, 1], requires_grad=True, device="cuda"))
        optimizer = Adagrad([*model.parameters(), softw], **config['optimizer'])

    epoch_loop = range(config['epochs'])
    if config.get_or_default("train/epoch_tqdm", False):
        epoch_loop = tqdm(epoch_loop,
                          desc="train",
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          )
    for epoch in epoch_loop:
        # 我们 propose 的模型训练
        epoch_loss = []
        loader = dataloader
        if config.get_or_default("train/batch_tqdm", True):
            loader = tqdm(loader,
                          desc=f'train  \tepoch: {epoch}/{config["epochs"]}',
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          )
        for packs in loader:
            optimizer.zero_grad()
            model.train()
            user, positive, negative = [p.cuda() for p in packs]
            batch_size = user.shape[0]
            user = user.unsqueeze(dim=1)
            assert user.shape == torch.Size([batch_size, 1])
            assert positive.shape == torch.Size([batch_size, config["sample_group_size"]])
            assert negative.shape == torch.Size([batch_size, config["sample_group_size"]])
            pos_score = model(user, positive).squeeze(dim=2)
            neg_score = model(user, negative).squeeze(dim=2)
            assert pos_score.shape == torch.Size([batch_size, config["sample_group_size"]])
            assert neg_score.shape == torch.Size([batch_size, config["sample_group_size"]])

            pos_score = pos_score.unsqueeze(dim=1)
            neg_score = neg_score.unsqueeze(dim=2)
            assert pos_score.shape == torch.Size([batch_size, 1, config["sample_group_size"]])
            assert neg_score.shape == torch.Size([batch_size, config["sample_group_size"], 1])

            neg_score = neg_score.topk(dim=1, k=config['sample_top_size'])[0]

            assert pos_score.shape == torch.Size([batch_size, 1, config["sample_group_size"]])
            assert neg_score.shape == torch.Size([batch_size, config['sample_top_size'], 1])

            if config.get_or_default("train/softw_enable", False):
                margin = neg_score - pos_score + softw
            else:
                margin = neg_score - pos_score + 1

            if 'loss/function' in config and config['loss/function'] == 'logistic':
                loss = torch.log(1 + torch.exp(margin))
            else:
                loss = F.relu(margin)

            if config.get_or_default("train/softw_enable", False):
                loss += torch.exp(-softw)

            loss.sum().backward()
            optimizer.step()
            epoch_loss.append(loss.mean().item())
        summary.add_scalar('Epoch/Loss', np.mean(epoch_loss), global_step=epoch)

        # 数据记录和精度验证
        if (epoch + 1) % config['evaluator_time'] == 0:
            torch.save(
                {
                    "model_static_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "softw": softw,
                    "epoch": epoch,
                    "config": config
                },
                os.path.join('%s' % config['writer_path'], f"checkpoint-{epoch}.tar")
            )
            evaluator.evaluate(model, epoch)
            evaluator.record_softw(softw)
            if evaluator.should_stop():
                print("early stop...")
                break
    summary.close()


if __name__ == '__main__':
    cfg = load_specific_config("config.yaml")
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            print("additional argument: " + arg)
            if "=" in arg and len(arg.split("=")) == 2:
                k, v = arg.strip().split("=")
                cfg[k] = v
                continue
            print("arg warning : " + arg)
            exit(0)
    wrap(cfg)

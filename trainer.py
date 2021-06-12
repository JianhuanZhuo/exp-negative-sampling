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
from model.GNS import GNS
from setproctitle import setproctitle


def wrap(config):
    pid = os.getpid()
    config['pid'] = pid
    print(f"pid is {pid}")

    if 'writer_path' not in config:
        subfolder = config['log_tag']
        if config["git/state"] == "Good":
            subfolder = '%s-%s' % (config['log_tag'], config['git']['hexsha'][:5])

        config['writer_path'] = os.path.join(config['log_folder'],
                                             subfolder,
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

    for epoch in range(config['epochs']):
        # 我们 propose 的模型训练
        epoch_loss = []
        for packs in tqdm(dataloader,
                          desc=f'train  \tepoch: {epoch}/{config["epochs"]}',
                          bar_format="{desc}{percentage:3.0f}%|{bar:10}{r_bar}",
                          ):
            optimizer.zero_grad()
            model.train()
            user, positive, negative = [p.cuda() for p in packs]
            pos_score = model(user, positive)
            neg_score = model(user, negative)
            loss = F.relu(pos_score - neg_score + 1)

            loss.sum().backward()
            optimizer.step()
            epoch_loss.append(loss)
        summary.add_scalar('Epoch/Loss', torch.stack(epoch_loss).mean().item(), global_step=epoch)

        # 数据记录和精度验证
        if (epoch + 1) % config['evaluator_time'] == 0:
            torch.save(
                {
                    "model_static_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join('%s' % config['writer_path'], f"checkpoint-{epoch}.tar")
            )
            evaluator.evaluate(model, epoch)


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

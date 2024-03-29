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
    grid_spec = ""
    if "grid_spec" in config:
        total = config.get_or_default("grid_spec/total", -1)
        current = config.get_or_default("grid_spec/current", -1)
        print(f"grid spec: {current:02}/{total:02} on cuda:{config['cuda']}")
        grid_spec = f"{current:02}/{total:02}/{config['cuda']}#"

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

    setproctitle(grid_spec + config['writer_path'])

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

    return


def distance(config, model, user, positive, negative, weight):
    batch_size = user.shape[0]
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

    margin = neg_score - pos_score + weight

    if 'loss/function' in config and config['loss/function'] == 'logistic':
        loss = torch.log(1 + torch.exp(margin))
    else:
        loss = F.relu(margin)
    return loss

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
            user_raw, positive, negative = [p.cuda() for p in packs]
            batch_size = user_raw.shape[0]
            user = user_raw.unsqueeze(dim=1)
            weight = 1
            if config.get_or_default("train/softw_enable", False):
                weight = softw[user_raw]

            loss = dist = distance(config, model, user, positive, negative, weight)

            if config.get_or_default("train/softw_enable", False):
                loss = dist + torch.exp(-softw[user_raw])

            loss.sum().backward()
            optimizer.step()
            epoch_loss.append(loss.mean().item())

            if config.get_or_default("sample_ig/enable", False):
                un_loss = dist.cpu().detach().mean(dim=2)

                if config.get_or_default("sample_ig/post_un_loss", False):
                    with torch.no_grad():
                        model.eval()
                        un_loss = distance(config, model, user, positive, negative, weight).cpu().detach().mean(dim=2)

                assert un_loss.shape == torch.Size([batch_size, config['sample_top_size']])
                dataset.update_un(user_raw.cpu().detach(), negative.cpu().detach(), un_loss)

        summary.add_scalar('Epoch/Loss', np.mean(epoch_loss), global_step=epoch)

        # 数据记录和精度验证
        # if (epoch + 1) % config['evaluator_time'] == 0:
        if epoch % config['evaluator_time'] == 0:
            save_dict = {
                "model_static_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "config": config
            }
            if config.get_or_default("train/softw_enable", False):
                save_dict["softw"] = softw
            torch.save(
                save_dict,
                os.path.join('%s' % config['writer_path'], f"checkpoint-{epoch}.tar")
            )

            evaluator.evaluate(model, epoch)
            if config.get_or_default("train/softw_enable", False):
                evaluator.record_softw(softw, epoch)
            if config.get_or_default("sample_ig/enable", False):
                evaluator.record_ig(dataset, epoch)
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

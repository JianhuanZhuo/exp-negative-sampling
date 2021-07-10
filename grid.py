from multiprocessing import Pool

from tools import config
from trainer import wrap
from itertools import product

if __name__ == '__main__':
    gpus = [0, 3, 1, 2] * 5 + [0, 3] * 12
    process_pool = Pool(len(gpus))
    exp_config = config.load_specific_config("config.yaml")

    grid = {
        # "loss/function": ["relu", "logistic"],
        # "DataLoader/batch_size": [2048],  # nbatch 先用大批次加速训练，找到好的参数后再用小的吧
        # "sample_group_size": [32],
        # "sample_top_size": [32],
        # "drop": [0.1],
        # "train/softw_enable": [True],
        "sample_ig/alpha": [0.0],
        "dataset/noise_p": [0, 0.05, 0.1, 0.15, 0.20, 0.25, 0.30],
        # "dataset/noise": [True, False],
        "optimizer/weight_decay": [0.6, 1.2, 2.4, 4.8, 9.6],  # learning_rate
        "optimizer/lr": [0.09, 0.12, 0.3],  # learning_rate
    }

    repeat = 3
    exp_config['log_folder'] = 'grid'
    task = 0
    exp_config['grid_spec/total'] = repeat * len(list(product(*list(grid.values()))))
    for r in range(repeat):
        for i, setting in enumerate(product(*list(grid.values()))):
            print(setting)
            for idx, k in enumerate(grid.keys()):
                exp_config[k] = setting[idx]
            exp_config['cuda'] = str(gpus[(r * repeat + i) % len(gpus)])
            task += 1
            exp_config['grid_spec/current'] = task
            process_pool.apply_async(wrap, args=(exp_config.clone(),))
        exp_config.random_again()

    process_pool.close()
    process_pool.join()

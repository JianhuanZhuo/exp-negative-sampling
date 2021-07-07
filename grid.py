from multiprocessing import Pool

from tools import config
from trainer import wrap
from itertools import product

if __name__ == '__main__':
    gpus = [0, 3, 1, 2] * 6 + [0, 3] * 6
    process_pool = Pool(len(gpus))
    exp_config = config.load_specific_config("config.yaml")

    grid = {
        # "DataLoader/batch_size": [2048],  # nbatch 先用大批次加速训练，找到好的参数后再用小的吧
        "optimizer/lr": [0.9, 0.12, 0.3],  # learning_rate
        "optimizer/weight_decay": [4.8, 9.6, 12.9],  # learning_rate
        # "sample_group_size": [32],
        # "sample_top_size": [32],
        # "drop": [0.1],
        # "dataset/noise_p": [0.05, 0.1, 0.15],
        "dataset/noise": [True, False],
        "train/softw_enable": [True, False],
        "loss/function": ["relu", "logistic"],
    }

    repeat = 2
    exp_config['log_folder'] = 'grid'
    for r in range(repeat):
        for i, setting in enumerate(product(*list(grid.values()))):
            print(setting)
            for idx, k in enumerate(grid.keys()):
                exp_config[k] = setting[idx]
            exp_config['cuda'] = str(gpus[(r * repeat + i) % len(gpus)])
            process_pool.apply_async(wrap, args=(exp_config.clone(),))
        exp_config.random_again()

    process_pool.close()
    process_pool.join()

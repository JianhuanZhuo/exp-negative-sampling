from multiprocessing import Pool

from tools import config
from trainer import wrap
from itertools import product

if __name__ == '__main__':
    # gpus = [0, 3, 1, 2]
    gpus = [0, 3, 1, 2]
    process_pool = Pool(3 * len(gpus))
    exp_config = config.load_specific_config("config.yaml")

    grid = {
        "DataLoader/batch_size": [1024],  # nbatch 先用大批次加速训练，找到好的参数后再用小的吧
        "optimizer/lr": [0.3],  # learning_rate
        # "optimizer/weight_decay": [0, 0.1],  # learning_rate
        "sample_group_size": [512, 256, 128, 64]
    }

    exp_config['log_folder'] = 'grid'
    for i, setting in enumerate(product(*list(grid.values()))):
        print(setting)
        for idx, k in enumerate(grid.keys()):
            exp_config[k] = setting[idx]
        exp_config['cuda'] = str(gpus[i % len(gpus)])
        print(f"{i:4} {exp_config.postfix()}")
        process_pool.apply_async(wrap, args=(exp_config.clone(),))

    process_pool.close()
    process_pool.join()
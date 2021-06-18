import json
import os
import random
import time

import yaml
from git import Repo
from git import cmd as gitCMD
import copy

class Config(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if '_key_' not in self:
            raise NotImplementedError("The _key_ not in the Config")

        if 'seed' not in self:
            self['origin_seed'] = "undefined"
            self['seed'] = random.randint(0, 100)
        else:
            self['origin_seed'] = self['seed']
            if not isinstance(self['seed'], int):
                self['seed'] = random.randint(0, 100)
        self['random_seed'] = self['seed']

        # set timestamp mask
        self['timestamp_mask'] = time.strftime("%m%d-%H%M%S", time.localtime())
        self['pid'] = os.getpid()
        self['git'] = current_git_info()

    def postfix(self, *args):
        posts = []
        for k, short in sorted(self['_key_'].items()):
            if short is None:
                short = k.split('/')[-1]
            v = self[k]
            if isinstance(v, bool):
                v = "Y" if v else "N"
            s = f"{short}{v}"
            posts.append(s)
        if args is not None:
            posts = [str(x) for x in args] + posts
        post = "-".join(posts)
        if self["git/state"] == "Good":
            post = "-".join([self["git"]["active_branch"], self["git"]["hexsha"][:5], post])

        post = "-".join([self["timestamp_mask"], post, "pid" + str(self["pid"])])

        if 'folder' in self:
            post = os.path.join(self['folder'], post)

        return post

    def json_str(self):
        return json.dumps(self, indent=2)

    def __contains__(self, item):
        if super(Config, self).__contains__(item):
            return True
        elif '/' not in item:
            return False
        else:
            v = self
            for p in item.split('/'):
                if p not in v:
                    return False
                else:
                    v = v[p]
            return True

    def __getitem__(self, item):
        if super(Config, self).__contains__(item):
            return super(Config, self).__getitem__(item)
        elif '/' in item:
            v = self
            for p in item.split('/'):
                assert p in v
                v = v[p]
            return v

    def __setitem__(self, key, value):
        if super(Config, self).__contains__(key) or '/' not in key:
            super(Config, self).__setitem__(key, value)
        else:
            d = self
            for p in key.split("/")[:-1]:
                v = {}
                if p in d:
                    v = d[p]
                    assert isinstance(v, dict)
                d[p] = v
                d = v
            d[key.split("/")[-1]] = value

    def get_or_default(self, key, default=None):
        if key in self:
            return self[key]
        else:
            return default

    def clone(self):
        return copy.deepcopy(self)

def load_config(path):
    """
    Get and parse the configeration file in `path`
    """
    if path.endswith("yaml") or path.endswith("yml"):
        with open(path, "r") as f:
            return Config(**yaml.load(f, Loader=yaml.FullLoader))
    raise Exception(f"config path exception: {path}")


def load_specific_config(path, default_file_path="./config.yaml"):
    if path.endswith("yaml") or path.endswith("yml"):
        if not os.path.exists(default_file_path):
            raise Exception(f"file is not exist : {os.path.abspath(default_file_path)}")
        with open(default_file_path, "r") as cm, open(path, "r") as sf:
            sf_config = yaml.load(sf, Loader=yaml.FullLoader)
            cm_config = yaml.load(cm, Loader=yaml.FullLoader)
            res = {}
            res.update(cm_config)
            res.update(sf_config)
            return Config(**res)
    raise Exception(f"config path exception: {path}")


def current_git_info():
    try:
        g = gitCMD.Git(".")
        print("git pull...")
        state = g.pull()
        print("git info:" + state)
        assert state == 'Already up to date.', f"the result of git state: {state}"
        with Repo(".") as repo:
            t = repo.active_branch.commit.committed_date
            return {
                "state": "Good",
                "active_branch": repo.active_branch.name,
                "working_tree_dir": repo.working_tree_dir,
                "timestamp": t,
                "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)),
                "summary": repo.active_branch.commit.summary,
                "hexsha": repo.active_branch.commit.hexsha,
            }
    except Exception as e:
        return {
            "state": "Invalid",
            "message": str(e.__str__()),
        }


if __name__ == '__main__':
    # print(current_git_info())
    config = load_config('test.yaml')
    # print(config.json_str())
    # config['pipeline/dataset'] = 'TransE'
    # print(config.postfix(1))
    # print(config.get_or_default('model_args', default="Nothing"))
    # config['pipeline/dataset'] = 'TransE'
    #
    # print(config['pipeline/dataset'])
    # a = config.clone()
    # config['pipeline/dataset'] = 'TransD'
    # print(config['pipeline/dataset'])
    # print(a['pipeline/dataset'])
    # def a(*, sd):
    #     print(sd)
    #
    # a(123, 543, sd=3123)

    print('pipeline' in config)
    print('pipeline/No' in config)
    print('pipeline/dataset' in config)
    print('No/No' in config)
    # print(config['pipeline/dataset'])
    # print(config['pipeline/No'])

import yaml
import argparse
from tqdm import tqdm
from ptsemseg.loader import get_loaders
from collections import defaultdict

def transform(cfg):
	loaders, n_classes = get_loaders(cfg["data"]["dataset"], cfg)
	for (_, _) in tqdm(loaders['train']):
		i = 0
	# for (_, _) in tqdm(loaders['val']):
	# 	i = 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument(
        "--config",
        nargs="?",
        type=str,
        default="configs/train/rgbd_BayesianSegnet_0.5_T000.yml",
        help="Configuration file to use",
    )

    parser.add_argument(
        "--tag",
        nargs="?",
        type=str,
        default="",
        help="Unique identifier for different runs",
    )

    parser.add_argument(
        "--run",
        nargs="?",
        type=str,
        default="",
        help="Directory to rerun",
    )

    parser.add_argument(
        "--seed",
        nargs="?",
        type=int,
        default=-1,
        help="Directory to rerun",
    )

    args = parser.parse_args()

    # cfg is a  with two-level dictionary ['training','data','model']['batch_size']
    if args.run != "":

        # find and load config
        for root, dirs, files in os.walk(args.run):
            for f in files:
                if '.yml' in f:
                    path = root + f
                    args.config = path

        with open(path) as fp:
            cfg = defaultdict(lambda: None, yaml.load(fp))

        # find and load saved best models
        for m in cfg['models'].keys():
            for root, dirs, files in os.walk(args.run):
                for f in files:
                    if m in f and '.pkl' in f:
                        cfg['models'][m]['resume'] = root + f

        logdir = args.run

    else:
        with open(args.config) as fp:
            cfg = defaultdict(lambda: None, yaml.load(fp))

        logdir = "/".join(["runs"] + args.config.split("/")[1:])[:-4]+'/'+cfg['id']

        # append tag 
        if args.tag:
            logdir += "/" + args.tag
        
        # set seed if flag set
        if args.seed != -1:
            cfg['seed'] = args.seed
            logdir += "/" + str(args.seed)

    transform(cfg)
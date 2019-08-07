import matplotlib

matplotlib.use('Agg')

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os
import argparse
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--include', action='append', nargs='+')
parser.add_argument('--match', action='append', nargs='+')
parser.add_argument('--baselines', action='store_true')
parser.add_argument('--best', action='store_true')
args = parser.parse_args()
include = args.include[0] if args.include else []
baselines = args.baselines
match = args.match[0] if args.match else []
best = args.best

runs = {}
for i, file in enumerate(glob.glob("./runs/**/*tfevents*", recursive=True)):

    directory = "/".join(file.split("/")[:-1])
    yml = glob.glob("{}/*.yml".format(directory))
    if len(yml) == 0:
        continue

    with open(yml[0], "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if not any([i in directory for i in include] + [m == directory.split("/")[-1] for m in match]):
        continue

    name = configs['id']
    # if any([e==name for e in exclude]):
    #     continue

    # print("Reading: {}".format(name))

    for event in tf.train.summary_iterator(file):
        for value in event.summary.value:

            if not directory in runs:
                runs[directory] = {}
                runs[directory]['raw_config'] = configs.copy()
                runs[directory]['raw_config']['file'] = file
                runs[directory]['raw_config']['file_only'] = file.split("/")[-2]

            if not value.tag in runs[directory]:
                runs[directory][value.tag] = {}
                runs[directory][value.tag]['step'] = []
                runs[directory][value.tag]['time'] = []
                runs[directory][value.tag]['value'] = []

            if value.HasField('simple_value'):
                # if len(runs[directory][value.tag]['step'])>0 and event.step<runs[directory][value.tag]['step'][-1]:
                runs[directory][value.tag]['step'].append(event.step)
                runs[directory][value.tag]['time'].append(event.wall_time)
                runs[directory][value.tag]['value'].append(value.simple_value)
if baselines:
    for i, file in enumerate(glob.glob("./runs/baselines/**/*tfevents*", recursive=True)):

        directory = "/".join(file.split("/")[:-1])

        # yaml_file = "{}/pspnet_airsim.yml".format(directory)
        # yaml_file = "{}/segnet_airsim_normal.yml".format(directory)
        yaml_file = glob.glob("{}/*.yml".format(directory))[0]

        if not os.path.isfile(yaml_file):
            continue

        with open(yaml_file, "r") as f:
            configs = yaml.load(f, Loader=yaml.FullLoader)

        name = configs['id']
        # if any([e==name for e in exclude]):
        #     continue

        # print("Reading: {}".format(name))

        for event in tf.train.summary_iterator(file):

            for value in event.summary.value:

                if not directory in runs:
                    runs[directory] = {}
                    runs[directory]['raw_config'] = configs.copy()
                    runs[directory]['raw_config']['file'] = file
                    runs[directory]['raw_config']['file_only'] = file.split("/")[-2]

                if not value.tag in runs[directory]:
                    runs[directory][value.tag] = {}
                    runs[directory][value.tag]['step'] = []
                    runs[directory][value.tag]['time'] = []
                    runs[directory][value.tag]['value'] = []

                if value.HasField('simple_value'):
                    runs[directory][value.tag]['step'].append(event.step)
                    runs[directory][value.tag]['time'].append(event.wall_time)
                    runs[directory][value.tag]['value'].append(value.simple_value)

# standardize config
for k, v in runs.items():
    c = v['raw_config']

    import random
    v['std_config'] = {}
    v['std_config']['size'] = "{}x{}".format(c['data']['img_rows'], c['data']['img_cols'])
    v['std_config']['id'] = v['raw_config']['id']
    v['std_config']['pretty'] = v['raw_config']['id'] + "_" + str(random.random())

scopes = ["Mean_Acc____"] \
    # + ["Mean_IoU____"] \
# + ["cls_{}".format(i) for i in range(9)]

figures = {}
axes = {}
data = []

for run in runs.keys():

    conditions = runs[run]['std_config']

    name = ", ".join(["{}{}".format(k, v) for k, v in conditions.items()])

    # find iteration with the best overall validation accuracies
    best_iter = -1
    best_value = -1
    if best:
        scores = defaultdict(float)
        for full in [k for k in runs[run].keys() if "config" not in k]:


            if 'loss' in full:
                continue

            tv, test, scope = full.split("/")

            if scope not in scopes:
                continue

            for i, v in enumerate(runs[run][full]['value']):
                scores[i] += v

        for i, v in scores.items():
            if best_value < v:
                best_iter = i
                best_value = v

    # iterate and append result to dataframe
    for full in [k for k in runs[run].keys() if "config" not in k]:

        if 'loss' in full:
            continue

        tv, test, scope = full.split("/")

        if scope not in scopes:
            continue

        # figures[test]
        x = runs[run][full]['step']
        y = runs[run][full]['value']

        t = runs[run][full]['time']
        a_i = scopes.index(scope) // 4
        a_j = scopes.index(scope) % 4

        # RESULTS
        i = best_iter

        avg = y[i]
        std = 0


        test_pretty = filter(None, test.split("_"))
        test_pretty = [s for s in test_pretty if s not in ["8camera", "dense"]]
        test_pretty = "\n".join(test_pretty)

        data.append({**conditions.copy(),
                     **{"raw": run,
                        "cls": scope,
                        "test": test_pretty,
                        "mean": avg,
                        "std": std},
                     "iter": x[i]})

df = pd.DataFrame(data)

if len(data) == 0:
    print("No Runs Found")
    exit()
data_fields = ['test', 'mean', 'std', 'raw', 'iter'] + list(set(runs[list(runs)[0]]['std_config']))
id_fields = ['test'] + list(set(runs[list(runs)[0]]['std_config']))

df = df[data_fields]

# uniqe identifier
df['unique_id'] = (df.groupby(id_fields).cumcount())
df['full'] = df['pretty']
df = df.sort_values(by=id_fields)

df = df.set_index(["test"])
df = df.pivot_table(index=df.index, values='mean', columns='full', aggfunc='first')
df.loc['combined'] = df.mean()


def rename(x):
    s = x.split()
    if 'value' in x:
        return s[-5] + "\n" + s[-3]
    return x


df = df.rename(rename, axis='index')

print(df)

# df = df.groupby(["block","mcdo_passes","mcdo_start_iter"]).mean()
# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.figure(figsize=(12, 8))
ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
# df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,0.99))
df.plot(kind='bar', ax=ax, yticks=[0.1 * i for i in range(10)]).legend(bbox_to_anchor=(1.0, -0.5))  # ,prop={'size':5})
ax.yaxis.grid(True)
plt.xticks(rotation=00)
# plt.xticks(rotation=45)
plt.ylabel("Mean Accuracy")
plt.savefig("plot.png")

df.to_csv('out.csv', index=False)

plt.show()

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

parser = argparse.ArgumentParser()
parser.add_argument('--include', action='append', nargs='+')
parser.add_argument('--match', action='append', nargs='+')
args = parser.parse_args()
include = args.include[0] if args.include else []
match = args.match[0] if args.match else []

print(include, match)
runs = {}
for i,file in enumerate(glob.glob("./runs/**/**/*tfevents*",recursive=True)):


    directory = "/".join(file.split("/")[:-1])

    # yaml_file = "{}/pspnet_airsim.yml".format(directory)
    # yaml_file = "{}/segnet_airsim_normal.yml".format(directory)
    yaml_file = glob.glob("{}/*.yml".format(directory))[0]

    if not os.path.isfile(yaml_file):
        continue

    with open(yaml_file,"r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)

    if not any([i in directory for i in include] + [m == directory.split("/")[-1] for m in match]):
        continue
        
    name = configs['id']
    # if any([e==name for e in exclude]):
    #     continue

    #print("Reading: {}".format(name))

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



#standardize config
#del_runs = []
#for k,v in runs.items():
#    if not any([v['raw_config']['file_only'] in [vvv for vvv,_ in vv['names']] for kk,vv in run_comments.items()]):
#        del_runs.append(k)

#for k in del_runs:
#    del runs[k]

#standardize config
for k,v in runs.items():

    c = v['raw_config']
    #print(k,v)

    v['std_config'] = {}
    v['std_config']['size'] = "{}x{}".format(c['data']['img_rows'],c['data']['img_cols'])
    v['std_config']['id'] = v['raw_config']['id']
    v['std_config']['pretty'] = v['raw_config']['id']
    # Extract comments for run
    v['std_config']['comments'] = "comments"
    v['std_config']['run_group'] = 'rungroup'

    #print(v)

    # # if c['start_layers'] is None or len(list(c['models']))==1:
    # if len(list(c['models']))==1:
    #     model = list(c['models'].keys())[0]
    #     v['std_config']['block'] = model
    # else:
    #     model = "rgb"
    #     v['std_config']['block'] = "-".join(c['start_layers'])

    model = list(c['models'].keys())[0]


    v['std_config']['reduction'] = c['models'][model]['reduction']    
    # v['std_config']['start_layers'] = c['start_layers']
    v['std_config']['mcdo_passes'] = c['models'][model]['mcdo_passes']
    v['std_config']['fuse_mech'] = "ModeSummed" if "fuse" in c['models'].keys() and c['models']['fuse']['in_channels']==-1 else "ModeStacked"
    #v['std_config']['mcdo_start_iter'] = c['models'][model]['mcdo_start_iter']
    #v['std_config']['multipass_backprop'] = c['models'][model]['mcdo_backprop']
    #v['std_config']['learned_uncertainty'] = True if c['models'][model]['learned_uncertainty']=='yes' else False
    v['std_config']['dropoutP'] = c['models'][model]['dropoutP']
    v['std_config']['pretrained'] = str(c['models'][model]['resume']) != "None"

scopes = ["Mean_Acc____"] \
        +["Mean_IoU____"] \
        +["cls_{}".format(i) for i in range(9)]

figures = {}
axes = {}
data = []


# print([v['std_config']['block'] for k,v in runs.items()])
# exit()

for run in runs.keys():
    #print(run)


    conditions = runs[run]['std_config']

    name = ", ".join(["{}{}".format(k,v) for k,v in conditions.items()])




    for full in [k for k in runs[run].keys() if "config" not in k]:
        #print(full)
        if 'loss' in full:
            continue
       

        tv, test, scope = full.split("/")



        if not scope in scopes:
            continue
        


        # if not test in figures:
        #     figures[test], axes[test] = plt.subplots(4,4) 
        #     figures[test].suptitle(test)

        # print(scope)
        # print(scopes.index(scope))
        # exit()

        # figures[test]
        x = runs[run][full]['step']
        y = runs[run][full]['value']
        
        t = runs[run][full]['time']
        a_i = scopes.index(scope) // 4
        a_j = scopes.index(scope) % 4
        # axes[test][a_i,a_j].plot(x,y,label=name)
        # axes[test][a_i,a_j].set_title(scope)
        # [axes[test][-1,i].set_axis_off() for i in range(4)]
        # axes[test][-2,0].legend(bbox_to_anchor=(4.0,-0.1))


        # print(runs[run]['raw_config']['file'])

        # if 500*(x[-1] // 500) <= 5000:
        #     continue

        # RESULTS
        # avg + std of last 50k iterations
        i = -1 #x.index(int(500*(x[-1] // 500))-5000)
        avg = np.mean(y)
        std = np.std(y)

        test_pretty = filter(None,test.split("_"))
        test_pretty = [s for s in test_pretty if s not in ["8camera","dense"]]
        test_pretty = "\n".join(test_pretty)



        data.append({**conditions.copy(),
                     **{"raw":run,
                        "cls":scope,
                        "test":test_pretty,
                        "mean":avg,
                        "std":std},
                        "iter":x[-1]})





df = pd.DataFrame(data)


df = df[(df.cls=="Mean_IoU____")]
data_fields = ['test','mean','std','raw','iter']+list(set(runs[list(runs)[0]]['std_config']))
id_fields = ['test']+list(set(runs[list(runs)[0]]['std_config']))


df = df[data_fields]

# uniqe identifier
df['unique_id'] = (df.groupby(id_fields).cumcount()) 
df['full'] = df['pretty']
df = df.sort_values(by=id_fields)

df = df[ 
        (
            (df['size'] != "")
        ) | ( 
            (df['size'] == "")

        )]


df = df.set_index(["test"])
df = df.pivot_table(index=df.index, values='mean', columns='full', aggfunc='first')
df.loc['combined'] = df.mean()

print(df)

# df = df.groupby(["block","mcdo_passes","mcdo_start_iter"]).mean()
# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.figure()
ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
# df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,0.99))
df.plot(kind='bar', ax=ax, yticks=[0.1*i for i in range(10)], xticks=[0]).legend(bbox_to_anchor=(1.0,-0.5)) #,prop={'size':5})
ax.yaxis.grid(True)
plt.xlabel("Test")
plt.xticks(rotation=00)
# plt.xticks(rotation=45)
plt.ylabel("Mean Accuracy")
plt.savefig("plot.png")

df.to_csv('out.csv',index=False)

plt.show()



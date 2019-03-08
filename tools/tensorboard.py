# from tensorflow.python.summary import event_accumulator
# from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
# from tensorflow.python.summary import event_accumulator as ea

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

desired_dirs = [
               "FullDropoutLeg_PreDropoutCorrections",
               "FullDropoutLeg_PreDropoutCorrections",
               "BurnIn_PreDropoutCorrections",
               "IsolatedLayer_PreDropoutCorrections",
               "layer_test_baseline_128x128_0.5reduction__train_fog_020___test_all__01-16-2019",
              ]

file_filter = [
               "correctedDropout",
               "layer_test",
               "burn-in",
               "IsolatedLayer",
              ]

runs = {}

for desired_dir in desired_dirs:
    for file in glob.glob("{}/**/*tfevents*".format(desired_dir),recursive=True):
        if any([f in file for f in file_filter]):
            # print("\n".join(filter(None,file.split("_"))))

            print(file)

            for event in tf.train.summary_iterator(file):
                for value in event.summary.value:

                    if not file in runs:
                        runs[file] = {}

                    if not value.tag in runs[file]:
                        runs[file][value.tag] = {}
                        runs[file][value.tag]['step'] = []
                        runs[file][value.tag]['time'] = []
                        runs[file][value.tag]['value'] = []

                    if value.HasField('simple_value'):
                        runs[file][value.tag]['step'].append(event.step)
                        runs[file][value.tag]['time'].append(event.wall_time)
                        runs[file][value.tag]['value'].append(value.simple_value)


out_file = open("results.txt","w")

scopes = ["Mean_Acc____"] \
        +["Mean_IoU____"] \
        +["cls_{}".format(i) for i in range(9)]

figures = {}
axes = {}
data = []

for run in runs.keys():
    line = run.split("/")[1]

    l = list(filter(None,line.split("__")))
    print(l)

    if len(l)==1:
        continue

    conditions = {"reduction":"N/A",
                  "size":"N/A",
                  "block":"N/A (Input Fusion Baseline)",
                  "passes":"N/A",
                  "mcdostart":"N/A"}

    if not "baseline" in line:
        conditions["reduction"] = l[0].split("_")[-1].replace("reduction","") if "reduction" in line else ""
        conditions["size"] = l[0].split("_")[-2]
        conditions["block"] = l[1]
        conditions["passes"] = l[-4].split("_")[0].replace("passes","") if "passes" in line else ""
        conditions["mcdostart"] = l[-4].split("_")[1].replace("mcdostart","") if "mcdostart" in line else ""
    else:
        print("\n"*10)

    [print("{}: {}".format(k,v)) for k,v in conditions.items()]


    for full in runs[run].keys():
        if 'loss' in full:
            continue
        
        tv, test, scope = full.split("/")

        if not scope in scopes:
            continue

        if not test in figures:
            figures[test], axes[test] = plt.subplots(4,4) 
            figures[test].suptitle(test)

        # print(scope)
        # print(scopes.index(scope))
        # exit()

        # figures[test]
        x = runs[run][full]['step']
        y = runs[run][full]['value']
        t = runs[run][full]['time']
        a_i = scopes.index(scope) // 4
        a_j = scopes.index(scope) % 4
        # axes[test][a_i,a_j].plot(x,y,label=line)
        # axes[test][a_i,a_j].set_title(scope)
        # [axes[test][-1,i].set_axis_off() for i in range(4)]
        # axes[test][-2,0].legend(bbox_to_anchor=(4.0,-0.1))


        # RESULTS
        # avg + std of last 50k iterations
        i = x.index(int(x[-1])-50000)
        avg = np.mean(y[i:])
        std = np.std(y[i:])


        data.append({**conditions,
                   **{"raw":line,
                      "cls":scope,
                      "test":test,
                      "mean":avg,
                      "std":std}})



df = pd.DataFrame(data)


df = df[(df.cls=="Mean_Acc____")]
# df = df[(df.test=="fog_000")]

df = df[['test','block','passes','mcdostart','mean','std']]

df['full'] = df['block']+", "+df['passes']+" passes, "+df['mcdostart']+" burn-in"


df = df.sort_values(by=["test","block","passes","mcdostart"])

df = df.set_index(["test"])
df.to_csv('out.csv',index=False)

df = df[((df["block"] == "N/A (Input Fusion Baseline)")) | 
         (df["block"] == "convbnrelu1_1-convbnrelu1_2-convbnrelu1_3") |
         (df["block"] == "convbnrelu1_1-convbnrelu1_3-res_block2") |
         (df["block"] == "convbnrelu1_1-res_block2-res_block3") |
         (df["block"] == "convbnrelu1_1-res_block2-res_block4") |
         (df["block"] == "convbnrelu1_1-res_block2-res_block5") |
         (df["block"] == "convbnrelu1_1-res_block2-pyramid_pooling") |
         (df["block"] == "convbnrelu1_1-res_block3-res_block4") |
         (df["block"] == "convbnrelu1_1-res_block3-res_block5") #) |

         # ((df["block"] == "convbnrelu1_1-convbnrelu1_3") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-res_block2") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-res_block3") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-res_block5") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-pyramid_pooling") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-cbr_final") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-classification") & (df['passes'] == "1")) |
         # ((df["block"] == "convbnrelu1_1-convbnrelu1_3") & (df['passes'] == "5")) |
         # ((df["block"] == "convbnrelu1_1-res_block2") & (df['passes'] == "5")) |
         # ((df["block"] == "convbnrelu1_1-res_block3") & (df['passes'] == "5")) |
         # ((df["block"] == "convbnrelu1_1-res_block5") & (df['passes'] == "5")) |
         # ((df["block"] == "convbnrelu1_1-pyramid_pooling") & (df['passes'] == "5")) |
         # ((df["block"] == "convbnrelu1_1-cbr_final") & (df['passes'] == "5")) |
         # ((df["block"] == "convbnrelu1_1-classification") & (df['passes'] == "5")) 

        # (df["block"] == "convbnrelu1_1-convbnrelu1_3") |
        # (df["block"] == "convbnrelu1_1-res_block2") |
        # (df["block"] == "convbnrelu1_1-res_block3") |
        # (df["block"] == "convbnrelu1_1-res_block5") |
        # (df["block"] == "convbnrelu1_1-pyramid_pooling") |
        # (df["block"] == "convbnrelu1_1-cbr_final") |
        # (df["block"] == "convbnrelu1_1-classification") 
        ]

# df = df.pivot_table(index=df.index, values='mean', columns='block', aggfunc='first')
df = df.pivot_table(index=df.index, values='mean', columns='full', aggfunc='first')

# df['combined'] = df.loc[:,"N/A (Input Fusion Baseline)":"convbnrelu1_1-res_block3-res_block4"].mean(axis=1)
df.loc['combined'] = df.mean()

print(df)

# df = df.groupby(["block","passes","mcdostart"]).mean()
# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.figure()
ax = plt.subplot2grid((4, 4), (0, 0), colspan=3, rowspan=4)
df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,0.99))
plt.xlabel("Test")
plt.xticks(rotation=15)
plt.ylabel("Mean Accuracy")
plt.show()

print(df)

df.to_csv('out.csv',index=False)



# plt.show()



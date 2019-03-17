import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os

include = [
           #'run1',
           #'run2',
           'run3',
           'run4',
           'run5',
           'run6',
           'run7',
           ]

run_comments = {
    "run1": {
        "names": [
            "MAGNUS_baseline_128x128__input_fusion___N-o-n-e__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain__train_8camera_fog_050_dense___test_all__01-16-2019",
            "MAGNUS_baseline_128x128__rgb_only___N-o-n-e__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain__train_8camera_fog_050_dense___test_all__01-16-2019",
            "MAGNUS_legFusion_128x128__convbnrelu1_1-classification___convbnrelu1_1-classification__4bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain__train_8camera_fog_050_dense___test_all__01-16-2019",
        ],
        "text":
            """No rgb/d/rgbd degradations; prior to adding recalibration split""",
    },
    "run2": {
        "names": [
            "MAGNUS_reweightedLoss_128x128__d_only__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain__train_8camera_fog_050_dense___test_all__01-16-2019",
            "MAGNUS_reweightedLoss_128x128__rgb_only__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain__train_8camera_fog_050_dense___test_all__01-16-2019",
            "MAGNUS_reweightedLoss_128x128__convbnrelu1_1-res_block3__8bs_0.5reduction_5passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_MultipliedFuse__train_8camera_fog_050_dense___test_all__01-16-2019",
            "MAGNUS_reweightedLoss_128x128__input_fusion__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain__train_8camera_fog_050_dense___test_all__01-16-2019",
        ],
        "text":
            """rgb/d degradation; ripl-w2; prior to adding recalibration split""",
    },
    "run3": {
        "names": [
            "MAGNUS_reweightedLoss_128x128__rgb_only__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain_01-16-2019",
            "MAGNUS_reweightedLoss_128x128__d_only__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain_01-16-2019",
            #"MAGNUS_reweightedLoss_128x128__convbnrelu1_1-classification__8bs_0.5reduction_5passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_StackedFuse_01-16-2019",
            "MAGNUS_reweightedLoss_128x128__input_fusion__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Nonemcdobackprop_pretrain_01-16-2019",
        ],
        "text":
            """rgb/d/rgbd degradation; ripl-w1; prior to adding recalibration split""",
    },    
    "run4": {
        "names": [
            "MAGNUS_attemptingVarianceFusion_128x128__convbnrelu1_1-classification__8bs_0.5reduction_5passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_MultipliedFuse_01-16-2019",
            "MAGNUS_attemptingVarianceFusion_128x128__convbnrelu1_1-res_block3__8bs_0.5reduction_5passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_MultipliedFuse_01-16-2019",
        ],
        "text":
            """
                rgb/d/rgbd degradation; ripl-w2; added recalibration split; adjusted color means to reflect entire dataset; 
                classification fusion seems to do well on ood data; however, it looks like it is simply fitting to one class / not learning anything useful; 
                mcdo on res_block3 seems better; it is not overfitting (shows true degradation in rgbd setting)
            """,
    },
    "run5": {
        "names": [
            "MAGNUS_attemptingVarianceFusionPretrain_128x128__convbnrelu1_1-classification__8bs_0.5reduction_5passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_MultipliedFuse_01-16-2019",
            "MAGNUS_attemptingVarianceFusionPretrain_128x128__convbnrelu1_1-res_block3__8bs_0.5reduction_5passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_MultipliedFuse_01-16-2019",
        ],
        "text":
            """
                rgb/d/rgbd degradation; ripl-w2; added recalibration split; adjusted color means to reflect entire dataset;                 
                trying pretraining for mcdo
            """,
    },    
    "run6": {
        "names": [
            "MAGNUS_attemptingVarianceFusionPretrain_128x128__convbnrelu1_1-classification__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_StackedFuse_01-16-2019",
            "MAGNUS_attemptingVarianceFusionPretrain_128x128__convbnrelu1_1-res_block3__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_StackedFuse_01-16-2019",
        ],
        "text":
            """
                rgb/d/rgbd degradation; ripl-w2; added recalibration split; adjusted color means to reflect entire dataset;                 
                trying pretraining for non-mcdo fusion (stacked outputs from each leg)
            """,
    },    
    "run7": {
        "names": [
            "MAGNUS_attemptingVarianceFusionPretrain1_128x128__convbnrelu1_1-classification__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_StackedFuse_01-16-2019",
            "MAGNUS_equalWeightingFusionBaseline_128x128__convbnrelu1_1-classification__8bs_0.5reduction_1passes_0.1dropoutP_nolearnedUncertainty_0mcdostart_Falsemcdobackprop_pretrain_MultipliedFuse_01-16-2019",
        ],
        "text":
            """
                trying pretraining for non-mcdo fusion; trying stacking outputs from each leg and equal weight sum of outputs from each leg
            """,
    },   


 

}



runs = {}
for i,file in enumerate(glob.glob("./**/*tfevents*",recursive=True)):

    directory = "/".join(file.split("/")[:-1])
    yaml_file = "{}/pspnet_airsim.yml".format(directory)

    if not os.path.isfile(yaml_file):
        continue

    with open(yaml_file,"r") as f:
        configs = yaml.load(f)

    if any([file.split("/")[-2] in vv['names'] and not kk in include for kk,vv in run_comments.items()]):
        continue
        
    name = configs['id']
    # if any([e==name for e in exclude]):
    #     continue

    print("Reading: {}".format(name))

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
del_runs = []
for k,v in runs.items():
    if not any([v['raw_config']['file_only'] in vv["names"] for kk,vv in run_comments.items()]):
        del_runs.append(k)

for k in del_runs:
    del runs[k]


#standardize config
for k,v in runs.items():

    c = v['raw_config']

    v['std_config'] = {}
    v['std_config']['size'] = "{}x{}".format(c['data']['img_rows'],c['data']['img_cols'])
    v['std_config']['id'] = v['raw_config']['id']
  
    if True or any([s==v['raw_config']['id'] for s in [
                                               "MAGNUS_baseline",
                                               "MAGNUS_legFusion",
                                               "MAGNUS_reweightedLoss",
                                               "MAGNUS_attemptingVarianceFusion",
                                              ]]):


        # Extract comments for run
        v['std_config']['comments'] = [vv["text"] for kk,vv in run_comments.items() if v['raw_config']['file_only'] in vv["names"]][0]
        v['std_config']['run_group'] = [kk for kk,vv in run_comments.items() if v['raw_config']['file_only'] in vv["names"]][0]

        if c['start_layers'] is None or len(list(c['models']))==1:
            model = list(c['models'].keys())[0]
            v['std_config']['block'] = model
        else:
            model = "rgb"
            v['std_config']['block'] = "-".join(c['start_layers'])


        v['std_config']['reduction'] = c['models'][model]['reduction']    
        # v['std_config']['start_layers'] = c['start_layers']
        v['std_config']['mcdo_passes'] = c['models'][model]['mcdo_passes']
        v['std_config']['fuse_mech'] = "ModeSummed" if "fuse" in c['models'].keys() and c['models']['fuse']['in_channels']==-1 else "ModeStacked"
        v['std_config']['mcdo_start_iter'] = c['models'][model]['mcdo_start_iter']
        v['std_config']['multipass_backprop'] = c['models'][model]['mcdo_backprop']
        v['std_config']['learned_uncertainty'] = True if c['models'][model]['learned_uncertainty']=='yes' else False
        v['std_config']['dropoutP'] = c['models'][model]['dropoutP']
        v['std_config']['pretrained'] = str(c['models'][model]['resume']) != "None"

        # print(v['raw_config']['id'])
        # print(v['std_config'])
        # print(c['models'][model]['resume'])
        # input()


    else:

        print(k)
        print()
        print(v['raw_config'])
        print()
        print(v['std_config'])
        exit()


out_file = open("results.txt","w")

scopes = ["Mean_Acc____"] \
        +["Mean_IoU____"] \
        +["cls_{}".format(i) for i in range(9)]

figures = {}
axes = {}
data = []


# print([v['std_config']['block'] for k,v in runs.items()])
# exit()

for run in runs.keys():


    conditions = runs[run]['std_config']

    name = ", ".join(["{}{}".format(k,v) for k,v in conditions.items()])



    for full in [k for k in runs[run].keys() if "config" not in k]:
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

        if 500*(x[-1] // 500) <= 5000:
            continue

        # RESULTS
        # avg + std of last 50k iterations
        i = x.index(int(500*(x[-1] // 500))-5000)
        avg = np.mean(y[i:])
        std = np.std(y[i:])

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
# df = df[(df.test=="fog_000")]

data_fields = ['test','mean','std','raw','iter']+list(set(runs[list(runs)[0]]['std_config']))
id_fields = ['test']+list(set(runs[list(runs)[0]]['std_config']))


df = df[data_fields]

# uniqe identifier
df['unique_id'] = (df.groupby(id_fields).cumcount()) 

# full string identifier
df['full'] = df['size']+", "+\
             df['block']+", "+\
             df['mcdo_passes'].map(str)+" mcdo_passes, "+\
             df['fuse_mech'].map(str)+" fuse_mech, "+\
             df['pretrained'].map(str)+" pretrained, "+\
             df['mcdo_start_iter'].map(str)+" burn-in, "+\
             df['multipass_backprop'].map(str)+" multipass_backprop, "+\
             df['learned_uncertainty'].map(str)+" learned_uncertainty, "+\
             df['dropoutP'].map(str)+" dropoutP ("+\
             df['unique_id'].map(str)+")"+\
             df['id'].map(str)

# sort by id fields
df = df.sort_values(by=id_fields)


# df.to_csv('out.csv',index=False)

df = df[ 
        (
            (df['size'] == "128x128") &
            # (df['multipass_backprop'] == True)
            (df['pretrained'] == True) &
            (df['fuse_mech'] == "ModeSummed") &
            (df["block"] == "convbnrelu1_1-classification") &    

            (df['size'] != "")
        
        
        # ) & (
        #     (df["block"] == "input_fusion") | 
        #     (df["block"] == "fused") | 
        #     (df["block"] == "rgb_only") | 
        #     (df["block"] == "d_only") |         
        #     (df['learned_uncertainty'] == True)

        # (
        #     (df['raw'].str.contains('baseline')) |
        #     (df['raw'].str.contains('correctedDropoutScalarLayerTest')) |
        #     (df['raw'].str.contains('layer_test_128x128'))
        # )

        ) | (
            (df["block"] == "input_fusion") | 
            (df["block"] == "fused") | 
            (df["block"] == "rgb_only") | 
            (df["block"] == "d_only") | 

        #     # (df["block"] == "convbnrelu1_1-convbnrelu1_2-convbnrelu1_3") |
        #     # (df["block"] == "convbnrelu1_1-convbnrelu1_3-res_block2") |
        #     # (df["block"] == "convbnrelu1_1-res_block2-res_block3") |
        #     # (df["block"] == "convbnrelu1_1-res_block2-res_block4") |
        #     # (df["block"] == "convbnrelu1_1-res_block2-res_block5") 

        # #     ((df["block"] == "convbnrelu1_1-convbnrelu1_3") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-res_block2") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-res_block3") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-res_block5") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-pyramid_pooling") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-cbr_final") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-classification") & (df['mcdo_passes'] == "1")) |
        # #     ((df["block"] == "convbnrelu1_1-convbnrelu1_3") & (df['mcdo_passes'] == "5")) |
        # #     ((df["block"] == "convbnrelu1_1-res_block2") & (df['mcdo_passes'] == "5")) |
        # #     ((df["block"] == "convbnrelu1_1-res_block3") & (df['mcdo_passes'] == "5")) |
        # #     ((df["block"] == "convbnrelu1_1-res_block5") & (df['mcdo_passes'] == "5")) |
        # #     ((df["block"] == "convbnrelu1_1-pyramid_pooling") & (df['mcdo_passes'] == "5")) |
        # #     ((df["block"] == "convbnrelu1_1-cbr_final") & (df['mcdo_passes'] == "5")) |
        # #     ((df["block"] == "convbnrelu1_1-classification") & (df['mcdo_passes'] == "5")) 

        #     # (df["block"] == "convbnrelu1_1-convbnrelu1_3") | 
        #     # (df["block"] == "convbnrelu1_1-res_block2") |
            # (df["block"] == "convbnrelu1_1-res_block3") |
        #     # (df["block"] == "convbnrelu1_1-res_block4") |
        #     # (df["block"] == "convbnrelu1_1-res_block5") |
        #     # (df["block"] == "convbnrelu1_1-pyramid_pooling") |
        #     # (df["block"] == "convbnrelu1_1-cbr_final") |
            # (df["block"] == "convbnrelu1_1-classification") |      
            (df['size'] == "")

        ###################
        # Test Conditions #
        ###################
        # ) & (
        #     (df['test'] == "fog_000") | 
        #     (df['test'] == "fog_025") | 
        #     (df['test'] == "fog_050") | 
        #     (df['test'] == "fog_100") |
        #     (df['test'] == "fog_100__depth_noise_mag20") |
        #     (df['test'] == "fog_100__rgb_noise_mag20") |
        #     (df['test'] == "combined")
        ###################

        )]

df.to_csv('out.csv',index=False)


df = df.set_index(["test"])
df = df.pivot_table(index=df.index, values='mean', columns='full', aggfunc='first')
df.loc['combined'] = df.mean()

print(df)

# df = df.groupby(["block","mcdo_passes","mcdo_start_iter"]).mean()
# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.figure()
ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
# df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,0.99))
df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,-0.5)) #,prop={'size':5})
plt.xlabel("Test")
plt.xticks(rotation=00)
# plt.xticks(rotation=45)
plt.ylabel("Mean Accuracy")
plt.show()

print(df)

df.to_csv('out.csv',index=False)



plt.show()



import matplotlib
matplotlib.use('TkAgg')

import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os

exclude = [
           'test1',
           'test',
           'viz',
           'viz_conf',
           'viz_weirdLoss',
           # 'isolated_layer_test',
           # 'burn-in_test',
           # '02-14-2019_layer_test',
           # 'FULL_Baseline',
           # 'FULL',
           # 'FULL_1BackpassMCDO',
           # 'FULL_No1BackpassMCDO',
           # 'FULL_1.0reduce_',
           # 'GAMUT',
           # 'HALF_GAMUT_learnedUncertainty',
           "legFusion_concreteDropoutSloppy",
           "MAGNUS_legFusion_concreteDropoutSloppy",
           "MAGNUS_baseline",
           "MAGNUS_legFusion",
           "MAGNUS_reweightedLoss",
           '',
           ]

runs = {}
for i,file in enumerate(glob.glob("./**/*tfevents*",recursive=True)):

    directory = "/".join(file.split("/")[:-1])
    yaml_file = "{}/pspnet_airsim.yml".format(directory)

    if not os.path.isfile(yaml_file):
        continue

    with open(yaml_file,"r") as f:
        configs = yaml.load(f)

    name = configs['id']
    if any([e==name for e in exclude]):
        continue

    print("Reading: {}".format(name))

    for event in tf.train.summary_iterator(file):
        for value in event.summary.value:

            if not directory in runs:
                runs[directory] = {}
                runs[directory]['raw_config'] = configs.copy()
                runs[directory]['raw_config']['file'] = file

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
for k,v in runs.items():
    c = v['raw_config']

    v['std_config'] = {}
    v['std_config']['size'] = "{}x{}".format(c['data']['img_rows'],c['data']['img_cols'])
    v['std_config']['id'] = v['raw_config']['id']


    if v['raw_config']['id']=="layer_test_baseline":
        v['std_config']['reduction'] = c['models']['fused']['reduction']
        v['std_config']['start_layers'] = 'input_fusion'
        v['std_config']['mcdo_passes'] = c['models']['fused']['mcdo_passes']
        v['std_config']['mcdo_start_iter'] = c['models']['fused']['mcdo_start_iter']
        v['std_config']['multipass_backprop'] = False
        v['std_config']['learned_uncertainty'] = False
        v['std_config']['dropoutP'] = 0.1
    elif any([s==v['raw_config']['id'] for s in ["layer_test",
                                                 "burn-in_test",
                                                 "02-14-2019_layer_test",
                                                ]]):
        v['std_config']['reduction'] = c['models']['fuse']['reduction']    
        v['std_config']['start_layers'] = [c['models']['rgb']['start_layer'],c['models']['fuse']['start_layer']]
        v['std_config']['mcdo_passes'] = c['models']['rgb']['mcdo_passes']
        v['std_config']['mcdo_start_iter'] = c['models']['rgb']['mcdo_start_iter']
        v['std_config']['multipass_backprop'] = c['models']['rgb']['mcdo_backprop']
        v['std_config']['learned_uncertainty'] = False
        v['std_config']['dropoutP'] = 0.1        
    elif any([s==v['raw_config']['id'] for s in ["isolated_layer_test",
                                                ]]):
        v['std_config']['reduction'] = c['models']['fuse']['reduction']    
        v['std_config']['start_layers'] = c['start_layers']
        v['std_config']['mcdo_passes'] = c['models']['rgb']['mcdo_passes']
        v['std_config']['mcdo_start_iter'] = c['models']['rgb']['mcdo_start_iter']
        v['std_config']['multipass_backprop'] = c['models']['rgb']['mcdo_backprop']
        v['std_config']['learned_uncertainty'] = False
        v['std_config']['dropoutP'] = 0.1        
    elif any([s==v['raw_config']['id'] for s in ["isolated_layer_test_aux",
                                                 "isolated_layer_test1",
                                                 "HALF_GAMUT_learnedUncertainty",
                                                 "learnedLoss",
                                                 "learnedLossNoPowerOnGradient",
                                                 "GAMUT_learnedLossNoPowerOnGradient",
                                                 "GAMUT",
                                                 "FULL",
                                                 "FULL_1BackpassMCDO",
                                                 "FULL_No1BackpassMCDO",
                                                 "FULL_1.0reduce_",
                                                 "correctedDropoutScalarLayerTest",
                                                 "onePassBackprop",
                                                 "output_fusion",
                                                 "legFusion",
                                                 "legFusion_pretrained",
                                                 "legFusion_pretrained_LR5",
                                                 "legFusion_pretrained_LR10",
                                                 "legFusion_dropoutP",
                                                 "isolated_layer_test_notpretrain",
                                                 "isolated_layer_test_new_notpretrain",
                                                 'isolated_layer_test_new',
                                                 "isolated_layer_test_evalDropoutOnly_notpretrain",

                                                ]]):
        v['std_config']['reduction'] = c['models']['fuse']['reduction']    
        v['std_config']['start_layers'] = c['start_layers']
        v['std_config']['mcdo_passes'] = c['models']['rgb']['mcdo_passes']
        v['std_config']['mcdo_start_iter'] = c['models']['rgb']['mcdo_start_iter']
        v['std_config']['multipass_backprop'] = c['models']['rgb']['mcdo_backprop']
        if not 'learned_uncertainty' in c['models']['rgb'].keys():
            v['std_config']['learned_uncertainty'] = False 
        else:
            v['std_config']['learned_uncertainty'] = True if c['models']['rgb']['learned_uncertainty']=='yes' else False

        if not 'dropoutP' in c['models']['rgb'].keys():
            v['std_config']['dropoutP'] = 0.1
        else:
            v['std_config']['dropoutP'] = c['models']['rgb']['dropoutP']

    elif any([s==v['raw_config']['id'] for s in ["GAMUT_baseline",
                                                 "FULL_Baseline",
                                                 "baseline_individual"
                                                ]]):
        v['std_config']['reduction'] = c['models'][list(c['models'].keys())[0]]['reduction']    
        v['std_config']['start_layers'] = c['start_layers']
        v['std_config']['mcdo_passes'] = c['models'][list(c['models'].keys())[0]]['mcdo_passes']
        v['std_config']['mcdo_start_iter'] = c['models'][list(c['models'].keys())[0]]['mcdo_start_iter']
        v['std_config']['multipass_backprop'] = c['models'][list(c['models'].keys())[0]]['mcdo_backprop']
        v['std_config']['learned_uncertainty'] = True if c['models'][list(c['models'].keys())[0]]['learned_uncertainty']=='yes' else False
        v['std_config']['dropoutP'] = 0.1

    else:

        print(k)
        print()
        print(v['raw_config'])
        print()
        print(v['std_config'])
        exit()

    if v['std_config']['start_layers'] is None or len(list(c['models']))==1:
        v['std_config']['block'] = list(c['models'].keys())[0]
    else:
        v['std_config']['block'] = "-".join(v['std_config']['start_layers'])

    print(v['std_config']['block'])
    print(v['raw_config']['file'])



out_file = open("results.txt","w")

scopes = ["Mean_Acc____"] \
        +["Mean_IoU____"] \
        +["cls_{}".format(i) for i in range(9)]

figures = {}
axes = {}
data = []

for run in runs.keys():


    conditions = runs[run]['std_config']

    name = ", ".join(["{}{}".format(k,v) for k,v in conditions.items() if k!="start_layers"])

    print(name)

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

        if 5000*(x[-1] // 5000) <= 50000:
            continue

        # RESULTS
        # avg + std of last 50k iterations
        i = x.index(int(5000*(x[-1] // 5000))-50000)
        avg = np.mean(y[i:])
        std = np.std(y[i:])

        print(conditions)

        data.append({**conditions,
                     **{"raw":run,
                        "cls":scope,
                        "test":test,
                        "mean":avg,
                        "std":std},
                        "iter":x[-1]})



df = pd.DataFrame(data)



df = df[(df.cls=="Mean_Acc____")]
# df = df[(df.test=="fog_000")]

df = df[['test','mean','std','raw','iter']+list(set(runs[list(runs)[0]]['std_config'])-set(['start_layers']))]

print(list(runs[list(runs)[0]]['std_config']))

df['unique_id'] = (df.groupby(['test']+list(set(runs[list(runs)[0]]['std_config'])-set(['start_layers']))).cumcount()) #,'size','block','mcdo_passes','mcdo_start_iter','multipass_backprop','learned_uncertainty']).cumcount())


df['full'] = df['size']+", "+\
             df['block']+", "+\
             df['mcdo_passes'].map(str)+" mcdo_passes, "+\
             df['mcdo_start_iter'].map(str)+" burn-in, "+\
             df['multipass_backprop'].map(str)+" multipass_backprop, "+\
             df['learned_uncertainty'].map(str)+" learned_uncertainty, "+\
             df['dropoutP'].map(str)+" dropoutP ("+\
             df['unique_id'].map(str)+")"+\
             df['id'].map(str)


# print(df['block'].unique())
# exit()

# df.to_csv('out.csv',index=False)

# exit()

df = df.sort_values(by=['test']+list(set(runs[list(runs)[0]]['std_config'])-set(['start_layers'])))
    #['test','size','block','mcdo_passes','mcdo_start_iter','multipass_backprop','learned_uncertainty'])



# df.to_csv('out.csv',index=False)

df = df[ 
        (
            (df['size'] == "128x128") &
            # (df['multipass_backprop'] == True)
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

        # ) & (
        #     (df["block"] == "input_fusion") | 
        #     (df["block"] == "fused") | 
        #     (df["block"] == "rgb_only") | 
        #     (df["block"] == "d_only") | 

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
        #     # (df["block"] == "convbnrelu1_1-res_block3") |
        #     # (df["block"] == "convbnrelu1_1-res_block4") |
        #     # (df["block"] == "convbnrelu1_1-res_block5") |
        #     # (df["block"] == "convbnrelu1_1-pyramid_pooling") |
        #     # (df["block"] == "convbnrelu1_1-cbr_final") |
        #     # (df["block"] == "convbnrelu1_1-classification") |      
        #     (df['size'] == "")

        ) & (
            (df['test'] == "fog_000") | 
            (df['test'] == "fog_025") | 
            (df['test'] == "fog_050") | 
            (df['test'] == "fog_100") |
            (df['test'] == "fog_100__depth_noise_mag20") |
            (df['test'] == "fog_100__rgb_noise_mag20") |
            (df['test'] == "combined")


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
df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,-0.2)) #,prop={'size':5})
plt.xlabel("Test")
plt.xticks(rotation=15)
plt.ylabel("Mean Accuracy")
plt.show()

print(df)

df.to_csv('out.csv',index=False)



plt.show()



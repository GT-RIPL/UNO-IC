import matplotlib
# matplotlib.use('TkAgg')
matplotlib.use('Agg')


import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
import os

include = [
           "redo0",
           ]

run_comments = {
    "redo0": {
        "names": [
            ("rgb_BayesianSegnet_0.5_NoTScale_T000_BS4_02","RGB Only (without Temperature Scaling)"),
            ("d_BayesianSegnet_0.5_TScale_T000_BS4_02","D Only (with Temperature Scaling)"),
        ],
        "text":
            """pretraining MCDO legs""",
    },  
}



runs = {}
for i,file in enumerate(glob.glob("./**/*tfevents*",recursive=True)):

    print(file)

    directory = "/".join(file.split("/")[:-1])
    # yaml_file = "{}/pspnet_airsim.yml".format(directory)
    # yaml_file = "{}/segnet_airsim_normal.yml".format(directory)
    yaml_file = glob.glob("{}/*.yml".format(directory))[0]

    if not os.path.isfile(yaml_file):
        continue

    with open(yaml_file,"r") as f:
        configs = yaml.load(f)

    if any([file.split("/")[-2] in [vvv for vvv,_ in vv['names']] and not kk in include for kk,vv in run_comments.items()]):
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
    if not any([v['raw_config']['file_only'] in [vvv for vvv,_ in vv['names']] for kk,vv in run_comments.items()]):
        del_runs.append(k)

for k in del_runs:
    del runs[k]

#standardize config
for k,v in runs.items():

    c = v['raw_config']

    v['std_config'] = {}
    v['std_config']['size'] = "{}x{}".format(c['data']['img_rows'],c['data']['img_cols'])
    v['std_config']['id'] = v['raw_config']['id']
    v['std_config']['pretty'] = [vvv1 if not vvv1 is None else v['raw_config']['id'] for vvv0,vvv1 in [vv["names"] for kk,vv in run_comments.items() if v['raw_config']['file_only'] in [vvv for vvv,_ in vv['names']]][0] if vvv0==v['raw_config']['file_only']][0]

    # Extract comments for run
    v['std_config']['comments'] = [vv["text"] for kk,vv in run_comments.items() if v['raw_config']['file_only'] in [vvv for vvv,_ in vv['names']]][0]
    v['std_config']['run_group'] = [kk for kk,vv in run_comments.items() if v['raw_config']['file_only'] in [vvv for vvv,_ in vv['names']]][0]

    print(v)

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
    v['std_config']['dropoutP'] = c['models'][model]['dropoutP']
    v['std_config']['pretrained'] = str(c['models'][model]['resume']) != "None"




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

        # if 500*(x[-1] // 500) <= 5000:
        #     continue

        # RESULTS
        # avg + std of last 50k iterations
        i = -1 #x.index(int(500*(x[-1] // 500))-5000)
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

print(df)

df = df[(df.cls=="Mean_IoU____")]
# df = df[(df.test=="fog_000")]

data_fields = ['test','mean','std','raw','iter']+list(set(runs[list(runs)[0]]['std_config']))
id_fields = ['test']+list(set(runs[list(runs)[0]]['std_config']))


df = df[data_fields]

# uniqe identifier
df['unique_id'] = (df.groupby(id_fields).cumcount()) 

# full string identifier
 # df['block']+", "+\
df['full'] = df['pretty'] #+", "+\
             # df['size']+", "+\
             # df['mcdo_passes'].map(str)+" mcdo_passes, "+\
             # df['fuse_mech'].map(str)+" fuse_mech, "+\
             # df['pretrained'].map(str)+" pretrained, "+\
             # df['mcdo_start_iter'].map(str)+" burn-in, "+\
             # df['multipass_backprop'].map(str)+" multipass_backprop, "+\
             # df['learned_uncertainty'].map(str)+" learned_uncertainty, "+\
             # df['dropoutP'].map(str)+" dropoutP ("+\
             # df['unique_id'].map(str)+")"+\
             # df['id'].map(str)

# sort by id fields
df = df.sort_values(by=id_fields)


# df.to_csv('out.csv',index=False)


df = df[ 
        (
            # (df['size'] == "128x128") &
            # (df['multipass_backprop'] == True)
            # (df['pretrained'] == True) &
            # (df['fuse_mech'] == "ModeSummed") &
            # (df["block"] == "convbnrelu1_1-classification") &    

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



        ################
        # Architecture #
        ################
        ) | (
        
        #     (df["block"] == "input_fusion") | 
        #     (df["block"] == "fused") | 
        #     (df["block"] == "rgb_only") | 
        #     (df["block"] == "d_only") | 
        #     (df["block"] == "N-o-n-e") | 

        # #     # (df["block"] == "convbnrelu1_1-convbnrelu1_2-convbnrelu1_3") |
        # #     # (df["block"] == "convbnrelu1_1-convbnrelu1_3-res_block2") |
        # #     # (df["block"] == "convbnrelu1_1-res_block2-res_block3") |
        # #     # (df["block"] == "convbnrelu1_1-res_block2-res_block4") |
        # #     # (df["block"] == "convbnrelu1_1-res_block2-res_block5") 

        # # #     ((df["block"] == "convbnrelu1_1-convbnrelu1_3") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-res_block2") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-res_block3") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-res_block5") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-pyramid_pooling") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-cbr_final") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-classification") & (df['mcdo_passes'] == "1")) |
        # # #     ((df["block"] == "convbnrelu1_1-convbnrelu1_3") & (df['mcdo_passes'] == "5")) |
        # # #     ((df["block"] == "convbnrelu1_1-res_block2") & (df['mcdo_passes'] == "5")) |
        # # #     ((df["block"] == "convbnrelu1_1-res_block3") & (df['mcdo_passes'] == "5")) |
        # # #     ((df["block"] == "convbnrelu1_1-res_block5") & (df['mcdo_passes'] == "5")) |
        # # #     ((df["block"] == "convbnrelu1_1-pyramid_pooling") & (df['mcdo_passes'] == "5")) |
        # # #     ((df["block"] == "convbnrelu1_1-cbr_final") & (df['mcdo_passes'] == "5")) |
        # # #     ((df["block"] == "convbnrelu1_1-classification") & (df['mcdo_passes'] == "5")) 

        # #     # (df["block"] == "convbnrelu1_1-convbnrelu1_3") | 
        # #     # (df["block"] == "convbnrelu1_1-res_block2") |
        #     # (df["block"] == "convbnrelu1_1-res_block3") |
        # #     # (df["block"] == "convbnrelu1_1-res_block4") |
        # #     # (df["block"] == "convbnrelu1_1-res_block5") |
        # #     # (df["block"] == "convbnrelu1_1-pyramid_pooling") |
        # #     # (df["block"] == "convbnrelu1_1-cbr_final") |
        #     # (df["block"] == "convbnrelu1_1-classification") |      
            (df['size'] == "")
        ################


        ###################
        # Test Conditions #
        ###################
        # ) & (
        #     (df['test'] == "train") | 
        #     (df['test'] == "async\nfog\n000\nclear") | 
        #     # (df['test'] == "fog_025") | 
        #     # (df['test'] == "fog_050") | 
        #     # (df['test'] == "fog_100") |
        #     # (df['test'] == "fog_100__depth_noise_mag20") |
        #     # (df['test'] == "fog_100__rgb_noise_mag20") |
        #     (df['test'] == "combined")
        ###################

        )]

df.to_csv('out.csv',index=False)


df = df.set_index(["test"])
df = df.pivot_table(index=df.index, values='mean', columns='full', aggfunc='first')

df = df[:3]
# exit()

df.loc['combined'] = df.mean()

print(df)

# df = df.groupby(["block","mcdo_passes","mcdo_start_iter"]).mean()
# grid = plt.GridSpec(2, 3, wspace=0.4, hspace=0.3)

plt.figure()
ax = plt.subplot2grid((4, 4), (0, 0), colspan=4, rowspan=2)
ax.grid()
# df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,0.99))
df.plot(kind='bar',ax=ax).legend(bbox_to_anchor=(1.0,-0.5)) #,prop={'size':5})
plt.xlabel("Test")
plt.xticks(rotation=00)

# plt.xticks(rotation=45)
plt.ylabel("Mean Accuracy")
plt.minorticks_on()
plt.grid(b=True, which='major', color='black', linestyle='-')
plt.grid(b=True, which='minor', color='black', linestyle='-')
plt.grid(True)
plt.show()

print(df)

df.to_csv('out.csv',index=False)


plt.savefig("histogram.png")
# plt.show()



# include = [
#            # 'run1',
#            # 'run2',
#            # 'run3',
#            # 'run4',
#            # 'run5',
#            # 'run6',
#            # 'run7',
#            # 'run8',
#            # 'run9',
#            # 'run10',
#            # "run11",
#            # "run12",
#            # "run13",
#            # "run14",
#            # "run15",
#            # "run16",
#            # "run17",
#            # "run18",
#            # "run19",
#            # "run20",
#            # "run21",
#            # "run22",
#            # "run23",
#            # "run24",
#            # "run25",
#            # "run26",
#            "run27",
#            "run28",
#            ]

# run_comments = {
#     "run1": {
#         "names": [
#             ("83811","RGB Only (No Dropout)"),
#         ],
#         "text":
#             """initial test on RGB only; no dropout; batch size 2""",
#     },
#     "run2": {
#         "names": [
#             ("rgb_dropout_between_layers_0.1",None),
#             ("rgb_dropout_between_layers_0.3",None),
#             ("rgb_dropout_between_layers_0.5",None),
#             ("rgb_dropout_between_layers_0.9",None),
#         ],
#         "text":
#             """testing for best dropout performance (0.1,0.3,0.5,0.9); dropout after each block; batch size 2""",
#     },
#     "run3": {
#         "names": [
#             ("rgb_baseline_0.1dropout_extensiveDropout",None),
#             ("rgb_baseline_0.5dropout_extensiveDropout",None),
#             ("d_baseline_0.1dropout_extensiveDropout",None),
#             ("d_baseline_0.5dropout_extensiveDropout",None),            
#         ],
#         "text":
#             """testing for best dropout performance (0.1,0.3,0.5,0.9); dropout after each convolutional layer; batch size 2""",
#     },    
#     "run4": {
#         "names": [
#             # ("d_BayesianSegnet_0.1","Depth Only (Bayesian Segnet, p = 0.1)"),
#             # ("rgb_BayesianSegnet_0.1","RGB Only (Bayesian Segnet, p = 0.1)"),
#             ("d_BayesianSegnet_0.5","Depth Only (Bayesian Segnet, p = 0.5)"),
#             ("rgb_BayesianSegnet_0.5","RGB Only (Bayesian Segnet, p = 0.5)"),
#         ],
#         "text":
#             """following architecture from BayesSegnet paper""",
#     },
#     "run5": {
#         "names": [
#             ("outputFusion_calibratedSoftmaxMultiply",   "[Train on 000, Recal on 000] RGB x D Hist"),
#             ("outputFusion_uncalibratedSoftmaxMultiply", "[Train on 000, No Recal] RGB x D Hist"),
#             # ("outputFusion_uncalibratedSoftmaxDonly",    "[Train on 000, No Recal] D"),
#             # ("outputFusion_uncalibratedSoftmaxRGBonly",  "[Train on 000, No Recal] RGB"),
#         ],
#         "text":
#             """calibrated softmaxes before adding values for fusion""",            
#     },
#     "run6": {
#         "names": [
#             # ("outputFusion_calibratedSoftmaxMultiply_recalibrateOn050",            "[Train on 000, Recal on 050] RGB x D"),
#             ("outputFusion_calibratedSoftmaxMultiply_recalibrateOn100",            "[Train on 000, Recal on 100] RGB x D"),
#             # ("outputFusion_calibratedSoftmaxMultiply_recalibrateOn100DBlackout",   "[Train on 000, Recal on 100 / No D] RGB x D"),
#             # ("outputFusion_calibratedSoftmaxMultiply_recalibrateOn100RGBBlackout", "[Train on 000, Recal on 100 / No RGB] RGB x D"),
#             # ("outputFusion_calibratedSoftmaxMultiply_recalibrateOnAllSplit",       "[Train on 000, Recal on All] RGB x D"),            
#         ],
#         "text":
#             """recalibrating on testing distribution""",            
#     },
#     "run7": {
#         "names": [
#             ("inputFusion_baseline", "[Train on 000, No Recal] RGBD Input Fusion"),
#         ],
#         "text":
#             """input fusion baseline""",            
#     },    
#     "run8": {
#         "names": [
#             ("outputFusion_FusionSoftmaxMultiply_Train000_Recal000", ".(p=0.5) [Train on 000, Recal on 000] RGB x D"),
#             ("outputFusion_FusionSoftmaxMultiply_Train000_Recal100", ".(p=0.5) [Train on 000, Recal on 100] RGB x D"),
#         ],
#         "text":
#             """refactored code check""",            
#     },    
#     "run9": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_T100_R000","[Train on 100, Recal on 000] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.5_T100_R050","[Train on 100, Recal on 050] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.5_T100_R100","[Train on 100, Recal on 100] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.5_T050_R000","[Train on 050, Recal on 000] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.5_T050_R050","[Train on 050, Recal on 050] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.5_T050_R100","[Train on 050, Recal on 100] RGB x D"),            
#         ],
#         "text":
#             """train on 050,100pct fog levels""",
#     },    
#     "run10": {
#         "names": [
#             # ("outputFusion_BayesianSegnet_0.5_Before6MCDO_T000_R000","(p=0.5) [Train on 000, Sample Recal on 000] RGB x D MCDO Recal"),
#             # ("outputFusion_BayesianSegnet_0.5_Before6MCDO_T000_R000.050.100","(p=0.5) [Train on 000, Sample Recal on 000,050,100] RGB x D MCDO Recal"),
#             # ("outputFusion_BayesianSegnet_0.5_Before6MCDO_T000_R050","(p=0.5) [Train on 000, Sample Recal on 050] RGB x D MCDO Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Before6MCDO_T000_R100","(p=0.5) [Train on 000, Sample Recal on 100] RGB x D MCDO Recal [Original Results]"),
#         ],
#         "text":
#             """recalibrating 50pct dropout before mean var calculations""",
#     },     
#     "run11": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.1_After6MCDO_T000_R000",".(p=0.1) [Train on 000, Sample Recal on 000] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.1_After6MCDO_T000_R100",".(p=0.1) [Train on 000, Sample Recal on 100] RGB x D"),
#             ("outputFusion_BayesianSegnet_0.1_Before6MCDO_T000_R000","(p=0.1) [Train on 000, Sample Recal on 000] RGB x D MCDO Recal"),
#             ("outputFusion_BayesianSegnet_0.1_Before6MCDO_T000_R100","(p=0.1) [Train on 000, Sample Recal on 100] RGB x D MCDO Recal"),            
#         ],
#         "text":
#             """trying recalibrating 10pct dropout before/after mean var calculations""",
#     },        
#     "run12": {
#         "names": [
#             ("outputFusion_LearnedRecalibrator_Polynomial4","[Train on 000, Recal on 000] RGB x D Poly4"),
#             ("outputFusion_LearnedRecalibrator_Polynomial8","[Train on 000, Recal on 000] RGB x D Poly8"),
#             ("outputFusion_LearnedRecalibrator_Polynomial16","[Train on 000, Recal on 000] RGB x D Poly16"),
#         ],
#         "text":
#             """trying learned parameterized recalibration curves""",
#     },     
#     "run13": {
#         "names": [
#             ("d_BayesianSegnet_0.5_dropout2d","D Only (Bayesian Segnet 2D, p = 0.5)"),
#             ("rgb_BayesianSegnet_0.5_dropout2d","RGB Only (Bayesian Segnet 2D, p = 0.5)"),
#         ],
#         "text":
#             """trying dropout2d instead of dropout, better suited for convnets""",
#     },     
#     "run14": {
#         "names": [
#             ("outputFusion_LearnedRecalibrator_Polynomial2","[Train on 000, Recal on 000] RGB x D Poly2"),
#             ("outputFusion_LearnedRecalibrator_Polynomial4","[Train on 000, Recal on 000] RGB x D Poly4"),
#             ("outputFusion_LearnedRecalibrator_Polynomial16","[Train on 000, Recal on 000] RGB x D Poly16"),
#             ("outputFusion_LearnedRecalibrator_Polynomial64","[Train on 000, Recal on 000] RGB x D Poly64"),
#         ],
#         "text":
#             """polynomial reparameterized recalibration""",
#     },     
#     "run15": {
#         "names": [
#             ("outputFusion_HistogramLinear_3","[Train on 000, Recal on 000] RGB x D Hist3"),
#             ("outputFusion_HistogramLinear_5","[Train on 000, Recal on 000] RGB x D Hist5"),
#             ("outputFusion_HistogramLinear_7","[Train on 000, Recal on 000] RGB x D Hist7"),
#             ("outputFusion_HistogramLinear_9","[Train on 000, Recal on 000] RGB x D Hist9"),
#             ("outputFusion_HistogramLinear_10","[Train on 000, Recal on 000] RGB x D Hist10"),
#             ("outputFusion_HistogramLinear_20","[Train on 000, Recal on 000] RGB x D Hist20"),
#             ("outputFusion_HistogramLinear_30","[Train on 000, Recal on 000] RGB x D Hist30"),
#         ],
#         "text":
#             """testing number of histogram bins""",
#     },     
#     "run16": {
#         "names": [
#             ("outputFusion_HistogramLinear_4_R100","[Train on 000, Recal on 100] RGB x D Hist4"),
#             ("outputFusion_HistogramLinear_8_R100","[Train on 000, Recal on 100] RGB x D Hist8"),
#             ("outputFusion_HistogramLinear_16_R100","[Train on 000, Recal on 100] RGB x D Hist16"),
#             ("outputFusion_HistogramLinear_32_R100","[Train on 000, Recal on 100] RGB x D Hist32"),            
#         ],
#         "text":
#             """testing number of histogram bins""",
#     },     
#     # SOMETHING WONKY AROUND HERE; wherever new dropout implenetation went
#     "run17": {
#         "names": [
#             ("rgb_BayesianSegnet_0.5_Masks0-5_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 0-5"),
#             ("rgb_BayesianSegnet_0.5_Masks6-11_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 6-11"),
#             ("rgb_BayesianSegnet_0.5_Masks12-17_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 12-17"),
#         ],
#         "text":
#             """trying different mask splits""",
#     },   
#     "run18": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_Masks0-5_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 00-05"),
#             ("outputFusion_BayesianSegnet_0.5_Masks12-17_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 12-17"),
#             ("outputFusion_BayesianSegnet_0.5_Masks18-23_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 18-23"),
#             ("outputFusion_BayesianSegnet_0.5_Masks24-29_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 24-29"),
#             ("outputFusion_BayesianSegnet_0.5_Masks30-35_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 30-35"),
#             ("outputFusion_BayesianSegnet_0.5_Masks36-41_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 36-41"),
#             ("outputFusion_BayesianSegnet_0.5_Masks42-47_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 42-47"),
#             ("outputFusion_BayesianSegnet_0.5_Masks48-53_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 48-53"),
#             ("outputFusion_BayesianSegnet_0.5_Masks54-59_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 54-59"),
#             ("outputFusion_BayesianSegnet_0.5_Masks6-11_T000_RNone_PLOT","[Train on 000, Recal on None] RGB x D Mask 06-11"),
#         ],
#         "text":
#             """trying even more mask splits; seems to indicate that masks are pretty reasonably sampled; recal size 0.2x""",
#     },   
#     "run19": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramFlat_10","[Train on 000, Recal on 100] RGB x D HistFlat 10"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramFlat_20","[Train on 000, Recal on 100] RGB x D HistFlat 20"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramFlat_3","[Train on 000, Recal on 100] RGB x D HistFlat 03"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramFlat_5","[Train on 000, Recal on 100] RGB x D HistFlat 05"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramFlat_50","[Train on 000, Recal on 100] RGB x D HistFlat 50"),
#         ],
#         "text":
#             """testing flat histogram parameters; recal size 0.2x; using weight averaging""",
#     },  
#     "run20": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramLinear_10","[Train on 000, Recal on 100] RGB x D HistLinear 10"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramLinear_20","[Train on 000, Recal on 100] RGB x D HistLinear 20"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramLinear_3","[Train on 000, Recal on 100] RGB x D HistLinear 03"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramLinear_5","[Train on 000, Recal on 100] RGB x D HistLinear 05"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_HistogramLinear_50","[Train on 000, Recal on 100] RGB x D HistLinear 50"),
#         ],
#         "text":
#             """testing linear interpolated histogram parameters; recal size 0.2x; using weight averaging""",
#     },  
#     "run21": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_Polynomial_10","[Train on 000, Recal on 100] RGB x D Polynomial 10"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_Polynomial_20","[Train on 000, Recal on 100] RGB x D Polynomial 20"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_Polynomial_3","[Train on 000, Recal on 100] RGB x D Polynomial 03"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_Polynomial_5","[Train on 000, Recal on 100] RGB x D Polynomial 05"),
#             ("outputFusion_BayesianSegnet_0.5_T000_R100_WAvg_RecalFunctionCrawl_Polynomial_50","[Train on 000, Recal on 100] RGB x D Polynomial 50"),
#         ],
#         "text":
#             """testing polynomial parameters; recal size 0.2x; using weight averaging""",
#     },    
#     "run22": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_WAvg_T000_R100_Reduction0.1_NoExtraTrain","[Train on 000, Recal on 100] RGB x D HistogramLinear 50 [My Implementation of Dropout]"),
#             ("outputFusion_BayesianSegnet_0.5_WAvg_T000_R100_Reduction0.1_NoExtraTrain_OldDropout","[Train on 000, Recal on 100] RGB x D HistogramLinear 50 [Back to nn.Dropout()]"),
#             # ("outputFusion_BayesianSegnet_0.5_WAvg_T000_R100_Reduction0.2_NoExtraTrain","[Train on 000, (0.2x) Recal on 100] RGB x D HistogramLinear 50"),
#             # ("outputFusion_BayesianSegnet_0.5_WAvg_T000_R100_Reduction0.5_NoExtraTrain","[Train on 000, (0.5x) Recal on 100] RGB x D HistogramLinear 50"),
#             # ("outputFusion_BayesianSegnet_0.5_WAvg_T000_R100_Reduction1.0_NoExtraTrain","[Train on 000, (1.0x) Recal on 100] RGB x D HistogramLinear 50"),
#         ],
#         "text":
#             """trying to figure out discrepancy between previous recalibration and current recalibration; current recalibration has worse performance; maybe because I changed the size of the recalibration dataset?; hm, there is something missing between the versions (possibly rescaling during forward evaluation pass?); found it, something off with dropout masks""",
#     },   
#     "run23": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_Masks0-5_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks00-05 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks12-17_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks12-17 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks18-23_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks18-23 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks24-29_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks24-29 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks30-35_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks30-35 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks36-41_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks36-41 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks42-47_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks42-47 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks48-53_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks48-53 PerPass Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks6-11_T000_R100_HistogramLinear50_BeforeMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks06-11 PerPass Recal"),
#         ],
#         "text":
#             """testing recalibration of different masks; per pass recalibration""",
#     },  
#     "run24": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_Masks0-5_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks00-05 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks12-17_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks12-17 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks18-23_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks18-23 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks24-29_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks24-29 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks30-35_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks30-35 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks36-41_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks36-41 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks42-47_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks42-47 WAvg Recal"),
#             ("outputFusion_BayesianSegnet_0.5_Masks6-11_T000_R100_HistogramLinear50_AfterMCDO_PLOT","[Train on 000, Recal on 100] RGB x D Masks06-11 WAvg Recal"),
#         ],
#         "text":
#             """testing recalibration of different masks; after weight averaging""",
#     },  
#     "run25": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_Masks0-5_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks00-05 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks12-17_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks12-17 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks18-23_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks18-23 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks24-29_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks24-29 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks30-35_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks30-35 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks36-41_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks36-41 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks42-47_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks42-47 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks48-53_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks48-53 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks6-11_T000_R000_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 000] RGB x D Masks06-11 PerPass Recal UndoneScaling"),
#         ],
#         "text":
#             """testing recalibration of different masks; per pass recalibration; rescaling dropout mask""",
#     },
#     "run26": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_Masks0-5_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks00-05 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks12-17_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks12-17 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks18-23_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks18-23 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks24-29_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks24-29 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks30-35_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks30-35 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks36-41_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks36-41 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks42-47_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks42-47 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks48-53_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks48-53 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks6-11_T000_R050_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 050] RGB x D Masks06-11 PerPass Recal UndoneScaling"),
#         ],
#         "text":
#             """testing recalibration of different masks; per pass recalibration; rescaling dropout mask""",
#     },
#     "run27": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5_Masks0-5_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks00-05 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks12-17_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks12-17 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks18-23_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks18-23 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks24-29_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks24-29 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks30-35_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks30-35 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks36-41_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks36-41 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks42-47_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks42-47 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks48-53_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks48-53 PerPass Recal UndoneScaling"),
#             ("outputFusion_BayesianSegnet_0.5_Masks6-11_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks06-11 PerPass Recal UndoneScaling"),
#         ],
#         "text":
#             """testing recalibration of different masks; per pass recalibration; rescaling dropout mask""",
#     },
#     "run28": {
#         "names": [
#             ("outputFusion_BayesianSegnet_0.5Channel_Masks0-5_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks00-05 PerPass Recal Channel Dropout"),
#             ("outputFusion_BayesianSegnet_0.5Channel_Masks12-17_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks12-17 PerPass Recal Channel Dropout"),
#             ("outputFusion_BayesianSegnet_0.5Channel_Masks18-23_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks18-23 PerPass Recal Channel Dropout"),
#             ("outputFusion_BayesianSegnet_0.5Channel_Masks24-29_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks24-29 PerPass Recal Channel Dropout"),
#             ("outputFusion_BayesianSegnet_0.5Channel_Masks30-35_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks30-35 PerPass Recal Channel Dropout"),
#             ("outputFusion_BayesianSegnet_0.5Channel_Masks6-11_T000_R100_HistogramLinear50_PPass_UndoneScaling","[Train on 000, Recal on 100] RGB x D Masks06-11 PerPass Recal Channel Dropout"),
#         ],
#         "text":
#             """testing recalibration of different masks; per pass recalibration; rescaling dropout mask; dropout channels""",
#     }    
# }

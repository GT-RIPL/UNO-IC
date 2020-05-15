# Adapted from score written by wkentaro
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py

import numpy as np


class runningScore(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))
        # self.classes = {'Void':4.24,'Sky':14.35,'Building':30.05,'Road':21.06,'Sidewalk':5.00,'Fence':4.31,'Vegetation':3.24,'Pole':1.50,'Car':9.19
        #                 ,'Sign':0.43,'Pedestrian':0.56,'Bicycle':0.41,'Lanemarking':5.24,'Reserved':0,'Reserved':0,'Traffic_Light':0.409
        #                 ,'Reserved':0,'Reserved':0}
        self.classes = ['Void','Sky','Building','Road','Sidewalk','Fence','Vegetation','Pole','Car'
                        ,'Sign','Pedestrian','Bicycle','Lanemarking','Reserved1','Reserved2','Traffic_Light'
                        ,'Reserved3','Reserved4']

    def _fast_hist(self, label_true, label_pred, n_class):
        #import ipdb;ipdb.set_trace()
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(
                lt.flatten(), lp.flatten(), self.n_classes
            )

    def get_scores(self):
        """Returns accuracy score evaluation result.
            - overall accuracy
            - mean accuracy
            - mean IU
            - fwavacc
        """
        hist = self.confusion_matrix
        overall_acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        mean_acc = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
        cls_iu = dict(zip(self.classes, iu))
        cls_acc = dict(zip(self.classes, acc_cls))
        count = hist.sum(axis=0)/hist.sum()
        # print(count*100)
        return (
            {
                "Overall Acc: \t": overall_acc,
                "Mean Acc : \t": mean_acc,
                "FreqW Acc : \t": fwavacc,
                "Mean IoU : \t": mean_iu,
            },
            cls_iu,
            cls_acc,
            count
        )



    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))



class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


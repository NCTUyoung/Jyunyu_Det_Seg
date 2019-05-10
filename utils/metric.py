from __future__ import print_function, division
import numpy as np
from functools import reduce

class runningScore_binary_classification(object):
    def __init__(self, threshold):
        self.threshold = threshold
        self.count = 1e-5
        self.equal = 0
    def update(self, pred, target):
        # binary threshold
        pred[pred < self.threshold] = 0
        pred[pred >= self.threshold] = 1
        # compare with gt
        num_equal = np.sum(np.equal(pred, target))
        # number of data 
        total_data = reduce(lambda i,j:i*j, [shape for shape in pred.shape])
        # update
        self.equal += num_equal
        self.count += total_data
    def get_scores(self):
        return self.equal/self.count
    def reset(self):
        self.count = 1e-5
        self.equal = 0

class runningScore_one_channel(object):
    """
    Running score for instance network.
    Network output are "one" channel only. 
    Compute accuracy all only dont need accuracy class.
    """
    def __init__(self, threshold = 0.51, visualize = False):
        self._accuracy_all_sum = 0.0
        # self._accuracy_class_sum = 0.0
        self.count = 0
        self.threshold = threshold
        self.visualize = visualize
    def update(self, pred, target, count):
        # accuracy all
        pred[pred >= self.threshold] = 1.0
        pred[pred < self.threshold] = 0.0
        correct_pred = np.sum(np.equal(pred,target))
        total_pixel = reduce(lambda a,b: a*b ,[shape for shape in pred.shape])
        _accuracy_all = correct_pred/float(total_pixel)
        # print("_accuracy_all", _accuracy_all)
        # accuracy class
        # class_ = class_[:,0].astype(np.int32) - 24
        # class_[class_ >=5] = class_[class_ >=5] -  2
        # # print(class_)
        # correct_pred = 0
        # total_pixel = 0
        # for i in range(pred.shape[0]):
        #     correct_pred += np.sum(np.equal(pred[i,class_[i],:,:], target[i,class_[i],:,:]))
        #     total_pixel  += reduce(lambda a,b: a*b ,[shape for shape in pred[i,class_[i],:,:].shape])
        # _accuracy_class = correct_pred/float(total_pixel)
        # print(_accuracy_class)
        # debug show mask
        # if self.visualize: self.show_mask(pred, target, class_)
        # update 
        self.count += count
        self._accuracy_all_sum += _accuracy_all
        # self._accuracy_class_sum += _accuracy_class

    def get_scores(self):
        return self._accuracy_all_sum/float(self.count)

    def reset(self):
        self._accuracy_all_sum = 0
        # self._accuracy_class_sum = 0
        self.count = 0
    def show_mask(self, pred, target, class_):
        # only show the first element
        print(pred.shape, target.shape, class_.shape, class_[0]+24)
        # get mask
        pred = pred[0]
        pred = pred.transpose((1,2,0))
        pred = pred[:,:,class_[0]]  
        # get target in normal format 
        target = target[0]
        target = target.transpose((1,2,0))
        target = target[:,:,class_[0]]  
        cv2.imshow("mask", pred)
        cv2.imshow("traget", target)
        cv2.waitKey(0)

class runningScore(object):
    """
    Running score for instance network.
    Network output are multi channel. 
    Compute accuracy all and accuracy class.
    """
    def __init__(self, threshold = 0.5, visualize = False):
        self._accuracy_all_sum = 0.0
        self._accuracy_class_sum = 0.0
        self.count = 0
        self.threshold = threshold
        self.visualize = visualize
    def update(self, pred, target, class_, count):
        # accuracy all
        pred[pred >= self.threshold] = 1.0
        pred[pred < self.threshold] = 0.0
        correct_pred = np.sum(np.equal(pred,target))
        total_pixel = reduce(lambda a,b: a*b ,[shape for shape in pred.shape])
        _accuracy_all = correct_pred/float(total_pixel)
        # print("_accuracy_all", _accuracy_all)
        # accuracy class
        class_ = class_[:,0].astype(np.int32) - 24
        class_[class_ >=5] = class_[class_ >=5] -  2
        # print(class_)
        correct_pred = 0
        total_pixel = 0
        for i in range(pred.shape[0]):
            correct_pred += np.sum(np.equal(pred[i,class_[i],:,:], target[i,class_[i],:,:]))
            total_pixel  += reduce(lambda a,b: a*b ,[shape for shape in pred[i,class_[i],:,:].shape])
        _accuracy_class = correct_pred/float(total_pixel)
        # print(_accuracy_class)
        # debug show mask
        if self.visualize: self.show_mask(pred, target, class_)
        # update 
        self.count += count
        self._accuracy_all_sum += _accuracy_all
        self._accuracy_class_sum += _accuracy_class

    def get_scores(self):
        return self._accuracy_all_sum/float(self.count), self._accuracy_class_sum/float(self.count) 

    def reset(self):
        self._accuracy_all_sum = 0
        self._accuracy_class_sum = 0
        self.count = 0
    def show_mask(self, pred, target, class_):
        # only show the first element
        print(pred.shape, target.shape, class_.shape, class_[0]+24)
        # get mask
        pred = pred[0]
        pred = pred.transpose((1,2,0))
        pred = pred[:,:,class_[0]]  
        # get target in normal format 
        target = target[0]
        target = target.transpose((1,2,0))
        target = target[:,:,class_[0]]  
        cv2.imshow("mask", pred)
        cv2.imshow("traget", target)
        cv2.waitKey(0)


class runningScore_instance(object):
    """
    Running score for each instance object
    This is for demo script only.
    """
    def __init__(self, num_class):
        self.num_class = num_class
        self.running_score = [{"count":1e-5, "acc_sum":0.0} for i in range(self.num_class)]
    def update(self, accuracy, class_):
        self.running_score[class_]["count"] += 1
        self.running_score[class_]["acc_sum"] += accuracy
    def get_scores(self):
        return self.running_score
    def reset(self):
        self.running_score = [{"count":1e-5, "acc_sum":0.0} for i in range(self.num_class)]

###########################################
# Running score for semantic segmentation #
###########################################
class runningScore_Segmentation(object):
    def __init__(self, n_classes):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def _fast_hist(self, label_true, label_pred, n_class):
        mask = (label_true >= 0) & (label_true < n_class)
        hist = np.bincount(
            n_class * label_true[mask].astype(int) + label_pred[mask],
            minlength=n_class ** 2,
        ).reshape(n_class, n_class)
        # print(hist.sum(axis=1))
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
        acc = np.diag(hist).sum() / hist.sum()
        acc_cls = np.diag(hist) / hist.sum(axis=1)
        acc_cls = np.nanmean(acc_cls)
        iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
        mean_iu = np.nanmean(iu)
        freq = hist.sum(axis=1) / hist.sum()
        fwavacc = (freq[freq > 0] * iu[freq > 0]).sum()
    
        cls_iu = dict(zip(range(self.n_classes), iu))

        return (
            {
                "OverallAcc": acc,
                "MeanAcc": acc_cls,
                "FreqWAcc": fwavacc,
                "mIoU": mean_iu,
            },
            cls_iu,
        )

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


class averageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.__val = 0
        self.__avg = 0
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.__val = val
        self.__sum += val * n
        self.__count += n
        self.__avg = self.__sum / self.__count

    @property
    def avg(self):
        return self.__avg
    @property
    def sum(self):
        return self.__sum
    
    
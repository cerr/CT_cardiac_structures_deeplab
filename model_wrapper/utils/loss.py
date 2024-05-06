import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
#from utils.layers import Dice, Jaccard
# class DiceLoss(nn.Module):
#     def __init__(self, reduce_axes=[1, 2, 3, 4], weight=1.0):
#         super(DiceLoss, self).__init__()
#         self.dice = Dice(reduce_axes=reduce_axes)
#         self.weight = weight

#     def forward(self, labels, predictions, **kwargs):
#         inverse_dice = 1.0 - self.dice(labels, predictions)

#         return self.weight * inverse_dice.mean()


# class JaccardLoss(nn.Module):
#     def __init__(self, reduce_axes=[1, 2, 3, 4], weight=1.0):
#         super(JaccardLoss, self).__init__()
#         self.jaccard = Jaccard(reduce_axes=reduce_axes)
#         self.weight = weight

#     def forward(self, labels, predictions, **kwargs):
#         negative_jaccard = self.jaccard(labels, predictions) * (-1.0)

#         return self.weight * negative_jaccard.mean()

class DICELossMultiClass(nn.Module):

    def __init__(self):
        super(DICELossMultiClass, self).__init__()

    def forward(self, output, mask):
        num_classes = output.size(1)
        n = output.size(0)
        dice_eso = 0
        # R.H. 29th March, 2019
        # converting mask for each image in batch from 3D to 4D to match output
        total_mask = []
        for j in range(n):
            tempM = mask[j,:,:]
            m = []

            mlabels = np.unique(tempM)
            for label in range(num_classes):
                m1 = (tempM==label)
                if self.cuda:
                  m1 = m1.type(torch.cuda.FloatTensor)
                else:
                  m1 = m1.type(torch.FloatTensor)
                m.append(m1)
            m = torch.stack(m)
            #reshape and append to total_mask
            #m = m.view(1,m.size(0), m.size(1), m.size(2))
            total_mask.append(m)
        total_mask = torch.stack(total_mask)
        total_mask = total_mask.view(mask.size(0), num_classes, mask.size(1), mask.size(2))
        
        if (total_mask.size(0) != mask.size(0)):
            print('4D total_mask size mismatch!')
            print(total_mask.size())

        for i in range(num_classes):
            probs = torch.squeeze(output[:, i, :, :], 1)
            sub_mask = torch.squeeze(total_mask[:, i, :, :], 1)
            num = probs * sub_mask
            num = torch.sum(num, 2)
            num = torch.sum(num, 1)

            # print( num )

            den1 = probs * probs
            # print(den1.size())
            den1 = torch.sum(den1, 2)
            den1 = torch.sum(den1, 1)

            # print(den1.size())

            den2 = sub_mask * sub_mask
            # print(den2.size())
            den2 = torch.sum(den2, 2)
            den2 = torch.sum(den2, 1)

            # print(den2.size())
            eps = 0.0000001
            dice = 2 * ((num + eps) / (den1 + den2 + eps))
            # dice_eso = dice[:, 1:]
            dice_eso += dice

        loss = 1 - (torch.sum(dice_eso) / dice_eso.size(0))/n
        return loss


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='dice'):
        """Choices: ['ce' or 'focal' or 'dice']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        elif mode== 'dice': 
            return self.DiceLoss
        elif mode== 'combined': 
            return self.combinedLoss
        elif mode== 'gendice': 
            return self.generalizedDiceLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss

    def combinedLoss(self, logit, target):
        n, c, h, w = logit.size()
        DSCloss = DICELossMultiClass()(output=logit, mask=target)
        CEloss = self.CrossEntropyLoss(logit=logit,target=target)
        return (DSCloss+CEloss)

    def DiceLoss(self, logit, target):
        n, c, h, w = logit.size()
        
        #loss = self._Dice(y_pred=logit, y_true=total_mask.float())
        #if self.batch_average:
        #    loss /= n
        loss = DICELossMultiClass()(output=logit, mask=target)
        return loss

    def generalizedDiceLoss(self, logit, target):
        n, c, h, w = logit.size()
        
        loss = self._Dice(y_pred=logit, y_true=target)
        #if self.batch_average:
        #    loss /= n
        return loss

    def _Dice(self, y_pred, y_true):
        epsilon=1e-7
        reduce_axes = tuple(range(1, len(y_pred.shape)-1))

        # converting predicted labels to 4D
        num_classes = y_pred.size(1)
        n = y_pred.size(0)
        # R.H. 29th March, 2019
        # converting mask for each image in batch from 3D to 4D to match output
        total_mask = []
        for j in range(n):
            tempM = y_true[j,:,:]
            m = []

            mlabels = np.unique(tempM)
            for label in range(num_classes):
                m1 = (tempM==label)
                if self.cuda:
                  m1 = m1.type(torch.cuda.FloatTensor)
                else:
                  m1 = m1.type(torch.FloatTensor)
                m.append(m1)
            m = torch.stack(m)
            total_mask.append(m)
        total_mask = torch.stack(total_mask)
        total_mask = total_mask.view(y_true.size(0), num_classes, y_true.size(1), y_true.size(2))

        intersection = (y_pred * total_mask)
        y_pred_square = y_pred.pow(2)
        y_true_square = total_mask.pow(2)

        for axis in reduce_axes:
            intersection = intersection.sum(dim=axis, keepdim=True)
            y_true_square = y_true_square.sum(dim=axis, keepdim=True)
            y_pred_square = y_pred_square.sum(dim=axis, keepdim=True)

        dice = (2.0 * intersection) / (y_true_square + y_pred_square + epsilon)
        return (1.0 - dice).mean() # average over classes and batch

        # skip the batch and class axis for calculating Dice score
        #axes = tuple(range(1, len(y_pred.shape)-1)) 
        #numerator = 2. * torch.sum(y_pred * y_true, axes)
        #denominator = torch.sum(y_pred_square + y_true_square, axes)
        #return (1 - (numerator / (denominator + epsilon))).mean() # average over classes and batch


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
    print(DICELossMultiClass()(a,b).item())
    print(loss.generalizedDiceLoss(a,b).item())





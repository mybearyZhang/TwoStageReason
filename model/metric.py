import torch
import torch.nn as nn

def MSE(output, target):
    return nn.MSELoss()(output, target)

def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)

def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)

def CFScore(output, target):
    loss_kpts = MSE(output[..., :2], target[..., :2])
    loss_coef = MSE(output[..., 2:7], target[..., 2:7])
    loss_state = MSE(output[..., :7], target[..., :7])

    loss_encoding = MSE(output, target)
    loss = MSE(output, target) * 1e3 + loss_encoding * 1e3
    score = (loss_kpts+loss_coef+ loss_state+loss_encoding+loss)/5
    return score

def VQAaccuracy(output, target):
    label = target[0]
    multi_choice = target[1]
    _, pred_exp1 = torch.max(output, 1)  # [batch_size]
    # _, pred_exp2 = torch.max(output, 1)  # [batch_size]
    # pred_exp2[pred_exp2 == ans_unk_idx] = -9999
    running_corr_exp1 = 0
    running_corr_exp1 += torch.stack([(multi_choice[:, i] == pred_exp1) for i in range(multi_choice.shape[1])]).any(dim=0).sum()
    # running_corr_exp2 += torch.stack([(ans == pred_exp2.cpu()) for ans in multi_choice]).any(dim=0).sum()
    epoch_acc_exp1 = running_corr_exp1.double() / len(label)      # multiple choice
    # epoch_acc_exp2 = running_corr_exp2.double() / len(label)      # multiple choice
    return epoch_acc_exp1
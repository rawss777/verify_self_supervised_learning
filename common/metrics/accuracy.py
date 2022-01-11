import torch


def accuracy(label, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = label.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


"""
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        batch_size = target.size(0)

        print(output)
        pred = output >= 0.5
        correct = pred.eq(target.view(1, -1))
        print(correct)
        res = correct.sum() / batch_size
        print(res)
        raise

        return res
"""

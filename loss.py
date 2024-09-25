import torch
import torch.nn.functional as F 
import numpy as np
class my_MSELoss(torch.nn.Module):
    
    def __init__(self):
        super(my_MSELoss, self).__init__()
        self.name = 'MSELoss'

    def forward(self, y_pred, y_target):
        
        L = (y_pred - y_target) ** 2
        return torch.mean(L)

class my_KLDLoss(torch.nn.Module):

    def __init__(self):
        super(my_KLDLoss, self).__init__()
        self.name = 'KLDLoss'

    def forward(self, y_pred, y_target):
        
        L_func = torch.nn.KLDivLoss(reduction='sum')
        y_target += 1e-16
        L = L_func(y_pred, y_target) / y_target.shape[0]
        return L

class my_weighted_MSELoss(torch.nn.Module):

    def __init__(self):
        super(my_weighted_MSELoss, self).__init__()
        self.name = 'weighted_MSELoss'

    def forward(self, y_pred, y_target, weights):
        loss = (y_pred - y_target) ** 2
        # print(weights.shape)
        weights = torch.tensor(weights).view(y_target.size(0), 1).cuda()
        loss *= weights
        loss = torch.mean(loss)

        return loss

# not implemented
# class my_weighted_FocalLoss(torch.nn.Module):

#     def __init__(self, weights=None):
#         super(my_weighted_MSELoss, self).__init__()
#         self.name = 'weighted_FocalLoss'
#         self.weights = weights 
#     def forward(self, y_pred, y_target, activate='sigmoid', beta=.2, gamma=1):
#         loss = (y_pred - y_target) ** 2
#         # loss *= (torch.tanh(beta * torch.abs(inputs - targets))) ** gamma if activate == 'tanh' else ((2 * torch.sigmoid(beta * torch.abs(inputs - targets)) - 1)** )
#         loss *= self.weights.expand_as(loss)
#         loss = torch.mean(loss)
#         return loss

if __name__=="__main__":
    from utils import num2vect
    import numpy as np

    my_mse = my_MSELoss()
    my_kld = my_KLDLoss()

    # print(my_mse(num2vect(5, 0), (num2vect(5, 0))))
    # print(my_mse(num2vect(5, 1), (num2vect(5, 1))))
    # print(my_kld(num2vect(5, 0), (num2vect(5, 0))))
    # print(my_kld(num2vect(5, 1), (num2vect(5, 1))))


    print("The network output is in the shape of (batch_size * 30)")
    print("If using hard label")
    
    network_output = torch.rand((2, 30))
    label = [5, 6]

    hard_output = torch.argmax(network_output, dim=1)
    print("hard output is : ", hard_output)

    soft_label = num2vect(label , sigma=1)
    print("soft label is : ", soft_label)
    print("hard label is : ", label)

    centers = torch.range(start=0.5, end=29.5)
    print("soft output is : ", torch.sum(centers*network_output, dim=1))
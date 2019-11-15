import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import math
import torch.nn.functional as F
import pdb


def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy 


def grl_hook(coeff):
    def fun1(grad):
        return -coeff*grad.clone()
    return fun1


def cal_weight_err(weight, pos_sample, output, label):
    w = torch.index_select(weight.view(-1, 1), 0, torch.tensor(pos_sample).to(weight.device))
    weight_err = torch.sum(w * nn.BCELoss(reduction='none')(output, label)) / torch.sum(w).detach().item()
    return weight_err


def CDAN(input_list, ad_net, entropy=None, coeff=None, random_layer=None):
    softmax_output = input_list[1].detach()
    feature = input_list[0]
    if random_layer is None:
        op_out = torch.bmm(softmax_output.unsqueeze(2), feature.unsqueeze(1))
        ad_out = ad_net(op_out.view(-1, softmax_output.size(1) * feature.size(1)))
    else:
        random_out = random_layer.forward([feature, softmax_output])
        ad_out = ad_net(random_out.view(-1, random_out.size(1)))       
    batch_size = softmax_output.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[feature.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:feature.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + \
                 target_weight / torch.sum(target_weight).detach().item()
        return torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, dc_target)) / torch.sum(weight).detach().item()
    else:
        return nn.BCELoss()(ad_out, dc_target) 


def DANN(features, ad_net):
    ad_out = ad_net(features)
    batch_size = ad_out.size(0) // 2
    dc_target = torch.from_numpy(np.array([[1]] * batch_size + [[0]] * batch_size)).float().cuda()
    return nn.BCELoss()(ad_out, dc_target)

def MADA(features, softmax_out, ad_net):
    pass
    loss = 0
    return loss


# input_list = [features, outputs]
def proposed(input_list, labels_source, ad_net, ad_net_group, entropy, coeff, iter_num, random_layer, trade_off=0.03):
    features = input_list[0]
    outputs = input_list[1].detach()
    [features_source, features_target] = torch.chunk(features, 2)
    [outputs_source, outputs_target] = torch.chunk(outputs, 2)

    features_source = random_layer.forward([features_source, nn.Softmax(dim=1)(outputs_source).detach()])
    features_target = random_layer.forward([features_target, nn.Softmax(dim=1)(outputs_target).detach()])
    features_source = features_source.view(-1, features_source.size(1))
    features_target = features_target.view(-1, features_target.size(1))
    LOG_INTERVAL = 10
    batch_size = features_source.size(0)
    # 计算领域对抗损失
    domain_label_src = torch.zeros(batch_size).view(-1, 1).float().cuda()
    domain_output_src = ad_net(features_source)

    domain_label_tar = torch.ones(batch_size).view(-1, 1).float().cuda()
    domain_output_tar = ad_net(features_target)

    ad_out = torch.cat((domain_output_src, domain_output_tar))
    ad_out_label = torch.cat((domain_label_src, domain_label_tar))
    # 计算领域对抗损失
    if entropy is not None:
        entropy.register_hook(grl_hook(coeff))
        entropy = 1.0+torch.exp(-entropy)
        source_mask = torch.ones_like(entropy)
        source_mask[features.size(0)//2:] = 0
        source_weight = entropy*source_mask
        target_mask = torch.ones_like(entropy)
        target_mask[0:features.size(0)//2] = 0
        target_weight = entropy*target_mask
        weight = source_weight / torch.sum(source_weight).detach().item() + target_weight / torch.sum(target_weight).detach().item()
        err_domain = torch.sum(weight.view(-1, 1) * nn.BCELoss(reduction='none')(ad_out, ad_out_label)) / torch.sum(weight).detach().item()
    else:
        err_domain = nn.BCELoss()(domain_output_src, domain_label_src)

    # Training model using source data
    category_domain_src, category_centre_src = ad_net_group(features_source, outputs_source, 'source',
                                                            label=labels_source)
    if iter_num % LOG_INTERVAL == 0:
        # 计算源域类别对抗损失(分为中心样本对抗损失与非中心样本对抗损失)
        err_c_s_domain = torch.tensor(0.).cuda()

        for key in category_domain_src['centre']['output'].keys():
            c_domain_label_src = torch.zeros(len(category_domain_src['centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_c_s_domain += cal_weight_err(weight, category_domain_src['centre']['index'][key], category_domain_src['centre']['output'][key], c_domain_label_src)
            else:
                err_c_s_domain += nn.BCELoss()(category_domain_src['centre']['output'][key], c_domain_label_src)

        for key in category_domain_src['no_centre']['output'].keys():
            c_domain_label_src = torch.zeros(len(category_domain_src['no_centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_c_s_domain += cal_weight_err(weight, category_domain_src['no_centre']['index'][key], category_domain_src['no_centre']['output'][key], c_domain_label_src)
            else:
                err_c_s_domain += nn.BCELoss()(category_domain_src['no_centre']['output'][key], c_domain_label_src)


        # 计算源域 中心/非中心样本对抗损失
        err_centre_src = torch.tensor(0.).cuda()
        for key in category_centre_src['centre']['output'].keys():
            centre_src_label = torch.ones(len(category_centre_src['centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_centre_src += cal_weight_err(weight, category_centre_src['centre']['index'][key], category_centre_src['centre']['output'][key], centre_src_label)
            else:
                err_centre_src += nn.BCELoss()(category_centre_src['centre']['output'][key], centre_src_label)

        for key in category_centre_src['no_centre']['output'].keys():
            no_centre_src_label = torch.zeros(len(category_centre_src['no_centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_centre_src += cal_weight_err(weight, category_centre_src['no_centre']['index'][key], category_centre_src['no_centre']['output'][key], no_centre_src_label)
            else:
                err_centre_src += nn.BCELoss()(category_centre_src['no_centre']['output'][key], no_centre_src_label)

    # Training model using target data
    category_domain_tar, category_centre_tar = ad_net_group(features_target, outputs_target, 'target')

    if iter_num % LOG_INTERVAL == 0:
        # 计算目标域类别对抗损失
        err_c_t_domain = torch.tensor(0.).cuda()

        for key in category_domain_tar['centre']['output'].keys():
            c_domain_label_tar = torch.ones(len(category_domain_tar['centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_c_t_domain += cal_weight_err(weight, category_domain_tar['centre']['index'][key], category_domain_tar['centre']['output'][key], c_domain_label_tar)
            else:
                err_c_t_domain += nn.BCELoss()(category_domain_tar['centre']['output'][key], c_domain_label_tar)

        for key in category_domain_tar['no_centre']['output'].keys():
            c_domain_label_tar = torch.ones(len(category_domain_tar['no_centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_c_t_domain += cal_weight_err(weight, category_domain_tar['no_centre']['index'][key], category_domain_tar['no_centre']['output'][key], c_domain_label_tar)
            else:
                err_c_t_domain += nn.BCELoss()(category_domain_tar['no_centre']['output'][key], c_domain_label_tar)

        # 计算目标域 中心/非中心样本对抗损失
        err_centre_tar = torch.tensor(0.).cuda()
        for key in category_centre_tar['centre']['output'].keys():
            centre_tar_label = torch.ones(len(category_centre_tar['centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_centre_tar += cal_weight_err(weight, category_centre_tar['centre']['index'][key], category_centre_tar['centre']['output'][key], centre_tar_label)
            else:
                err_centre_tar += nn.BCELoss()(category_centre_tar['centre']['output'][key], centre_tar_label)

        for key in category_centre_tar['no_centre']['output'].keys():
            no_centre_tar_label = torch.zeros(len(category_centre_tar['no_centre']['output'][key])).view(-1, 1).float().cuda()
            if entropy is not None:
                err_centre_tar += cal_weight_err(weight, category_centre_tar['no_centre']['index'][key], category_centre_tar['no_centre']['output'][key], no_centre_tar_label)
            else:
                err_centre_tar += nn.BCELoss()(category_centre_tar['no_centre']['output'][key], no_centre_tar_label)

    # 全部损失
    if iter_num % LOG_INTERVAL == 0:
        err = err_domain+trade_off*(err_c_s_domain + err_c_t_domain)+trade_off*(err_centre_src + err_centre_tar)
    else:
        err = err_domain

    if iter_num % LOG_INTERVAL == 0:
        print('iter_num:{},err_domain:{:.4f}'.format(iter_num, err_domain.item()))
        print('err_c_s_domain: {:.4f}, err_c_t_domain: {:.4f},'
              ' err_centre_src: {:.4f}, err_centre_tar: {:.4f}'.format(err_c_s_domain.item(), err_c_t_domain.item(),
                                                                       err_centre_src.item(), err_centre_tar.item()))
    return err

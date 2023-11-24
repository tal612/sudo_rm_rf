"""!
@brief Running an experiment with the Attentive-SuDoRmRf
@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana-Champaign
"""

import os
import sys

current_dir = os.path.dirname(os.path.abspath('__file__'))
root_dir = os.path.abspath(os.path.join(current_dir, '../../../'))
sys.path.append(root_dir)

# from __config__ import API_KEY
# from comet_ml import Experiment

import torch
from torch.nn import functional as F
from tqdm import tqdm
from pprint import pprint
import sudo_rm_rf.dnn.experiments.utils.improved_cmd_args_parser_v2 as parser
import sudo_rm_rf.dnn.experiments.utils.dataset_setup as dataset_setup
from sudo_rm_rf.dnn.models.gender_detector import GenderDetector, ZFNet, ZFNet1, ZFNet2f
from sudo_rm_rf.utils.early_stop import EarlyStopper
from torch import nn
import wandb
from torchinfo import summary


import numpy as np

from datetime import date

WB = False
INSPECT = True


cuda0 = torch.device('cuda:2')

args = parser.get_args()
hparams = vars(args)

if INSPECT:
    flags = ['n_train_val', 'n_test', 'n_val', 'n_train']
    for flag in flags:
        hparams[flag] = 10

generators = dataset_setup.setup(hparams)

if hparams['separation_task'] == 'enh_single':
    hparams['n_sources'] = 1
else:
    hparams['n_sources'] = 2

# hparams['cuda_available_devices'] = ['2']
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([cad for cad in hparams['cuda_available_devices']])

# torch.set_default_device('cuda:2')
# print(','.join(
#     [cad for cad in hparams['cuda_available_devices']]))


# print(summary(model, torch.zeros((2, 1, 32000)), show_input=True, show_hierarchical=False))
model = ZFNet(class_count=3)
model = ZFNet2f(3, 3)
model.to(cuda0)
# summary(model, input_size=(1, 3, 224, 224))
opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])

# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer=opt, mode='max', factor=1. / hparams['divide_lr_by'],
#     patience=hparams['patience'], verbose=True)

dsi_num = hparams["dsi_gpu"][0] if type(hparams["dsi_gpu"]) is list else 0

if WB:
    wandb.init(
        # set the wandb project where this run will be logged
        project="Training Gender Detector ZFNet",
        name=f'{hparams["n_epochs"]} epochs - {date.today()} - dsi_0{dsi_num}',

        # track hyperparameters and run metadata
        config={
            "hyperparameters": hparams
        }
    )


def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)


early_stopper = EarlyStopper(patience=3, min_delta=0.02)
tr_step = 0
val_step = 0
prev_epoch_val_loss = 0.
best_mean = 1000
criterion = nn.CrossEntropyLoss()

for i in range(hparams['n_epochs']):
    batch_step = 0
    sum_loss = 0.

    print("Attentive Sudo-RM-RF: || Epoch: {}/{}".format(i + 1, hparams['n_epochs']))
    model.train()

    values_train = []
    training_gen_tqdm = tqdm(generators['train'], desc='Training')

    corrects = 0
    for j, data in enumerate(training_gen_tqdm):
        # data = [sources_wavs, get_gender(filename), target_wavs]
        clean_wavs = data[-1].cuda(cuda0)
        m1wavs = data[0].cuda(cuda0)
        label_raw = data[1].to(cuda0)  # , dtype=torch.float32)
        # label = torch.zeros(3)
        # label[label_raw] = 1
        # print((label_raw))
        label = label_raw
        # features = model.extract_features(m1wavs).cuda()
        # print(f"{features.device=}")
        output_prob = model(m1wavs)
        # print(output_prob.shape)
        _, decision = torch.max(torch.nn.functional.softmax(output_prob), 1)
        # print(int(decision), int(label))
        # print(output_prob, label)
        # print(f"{decision=}")
        # print(f'{output_prob=}     {label=}')
        loss = criterion(input=output_prob, target=label)

        loss.backward()
        opt.zero_grad()

        sum_loss += loss.item()
        if int(decision) == int(label):
            corrects += 1
        # if j % 2000 == 1999:  # print every 2000 mini-batches
        #     print(f'[{i + 1}, {j + 1:5d}] loss: {sum_loss / 2000:.3f}')
        #     running_loss = 0.0

        training_gen_tqdm.set_description(
            f"Training, Running Avg Loss: {sum_loss / (batch_step + 1)}   |   Accuracy: {corrects/(batch_step + 1)}    "
        )
        batch_step += 1

    tr_step += 1
    if WB:
        wandb.log({"loss_train": sum_loss / batch_step,
                   "tr_step_train": tr_step})

    # model.extract_features(m1wavs)

    q = 0
    val_loss = 0
    corrects = 0
    val_gen_tqdm = tqdm(generators['train_val'])
    for val_set in [x for x in generators if not x == 'train']:
        if generators[val_set] is not None:
            model.eval()
            with torch.no_grad():
                for data in val_gen_tqdm:
                    m1wavs = data[0].cuda(cuda0)
                    label = data[1].cuda(cuda0)

                    output_prob = model(m1wavs)
                    _, decision = torch.max(torch.nn.functional.softmax(output_prob), 1)
                    # print("VAL",int(decision), int(label))
                    # print(output_prob, label)
                    loss = criterion(input=output_prob, target=label)
                    val_loss += loss.item()

                    if int(decision) == int(label):
                        corrects += 1
                    q += 1

                    val_gen_tqdm.set_description('Validation on {}   |   Accuracy: {}      '
                                                 .format(val_set, corrects / q))

    val_step += 1
    if WB:
        wandb.log({"loss_val": val_loss / q,
                   "accuracy": corrects / q,
                   "val_step": val_step})
    # print(f"accuracy is {corrects / q}")

if WB:
    wandb.finish()

# if __name__ == "__main__":
#     loss = nn.CrossEntropyLoss()
#     input = torch.randn(5, requires_grad=True)
#     target = torch.empty(1, dtype=torch.long).random_(5)
#
#     print(input, input.size())
#     print()
#     print(target, target.size())
#     print(loss(input,target))
# # rec_sources_wavs = model(m1wavs.unsqueeze(1))
#
# # print(f"{m1wavs.unsqueeze(1).size()=}")
# # rec_sources_wavs = mixture_consistency.apply(
# #     rec_sources_wavs, m1wavs.unsqueeze(1)
# # )
#
# #     l = torch.clamp(
# #         back_loss_tr_loss(rec_sources_wavs, clean_wavs[:,0:2,:]),
# #         min=-30., max=+30.)
# #     l.backward()
# #     if hparams['clip_grad_norm'] > 0:
# #         torch.nn.utils.clip_grad_norm_(model.parameters(),
# #                                        hparams['clip_grad_norm'])
# #
# #     opt.step()
# #
# #     np_loss_value = l.detach().item()
# #     sum_loss += np_loss_value
# #     training_gen_tqdm.set_description(
# #         f"Training, Running Avg Loss: {sum_loss / (batch_step + 1)}"
# #     )
# #     batch_step += 1
# #
# #     values_train.append(np_loss_value)
# #
# # if hparams['patience'] > 0:
# #     if tr_step % hparams['patience'] == 0:
# #         new_lr = (hparams['learning_rate']
# #                   / (hparams['divide_lr_by'] ** (tr_step // hparams['patience'])))
# #         print('Reducing Learning rate to: {}'.format(new_lr))
# #         for param_group in opt.param_groups:
# #             param_group['lr'] = new_lr
# # tr_step += 1
# #
# #
# # l_name = 'train_val_SISDRi'
# # mean_metric = np.mean(values_train)
# # std_metric = np.std(values_train)
# #
# #
# # for val_set in [x for x in generators if not x == 'train']:
# #     if generators[val_set] is not None:
# #         model.eval()
# #         with torch.no_grad():
# #             for data in tqdm(generators[val_set],
# #                              desc='Validation on {}'.format(val_set)):
# #                 m1wavs = data[0].cuda()
# #                 m1wavs = normalize_tensor_wav(m1wavs.sum(1)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# #                 clean_wavs = data[-1].cuda()
# #
# #                 rec_sources_wavs = model(m1wavs.unsqueeze(1))
# #
# #                 for loss_name, loss_func in val_losses[val_set].items():
# #                     l = loss_func(rec_sources_wavs,
# #                                   clean_wavs[:,0:2,:],
# #                                   initial_mixtures=m1wavs.unsqueeze(1))
# #                     res_dic[loss_name]['acc'] += l.tolist()
# # val_step += 1
# #
# # l_name = 'train_val_SISDRi'
# # values = res_dic[l_name]['acc']
# # mean_metric = np.mean(values)
# # std_metric = np.std(values)
# #
# #
# # for loss_name in res_dic:
# #     res_dic[loss_name]['acc'] = []
# #
# # if hparams["save_best_weights"] == True:
# #     if mean_metric < best_mean:
# #         best_mean = mean_metric
# #         torch.save(
# #             model.state_dict(),
# #             os.path.join(hparams["checkpoints_path"],
# #                          f"best_weights.pt"),
# #         )
# # else:
# #     if hparams["save_checkpoint_every"] > 0:
# #         if tr_step % hparams["save_checkpoint_every"] == 0:
# #             torch.save(
# #                 model.state_dict(),
# #                 os.path.join(hparams["checkpoints_path"],
# #                              f"improved_sudo_epoch_{tr_step}.pt"),
# #             )
# #
# # if EarlyStopper.early_stop(-mean_metric):
# #     break
#
# # [optional] finish the wandb run, necessary in notebooks

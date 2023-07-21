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
import sudo_rm_rf.dnn.experiments.utils.test_renana_args_parser as parser
import sudo_rm_rf.dnn.experiments.utils.dataset_setup as dataset_setup
import sudo_rm_rf.dnn.losses.sisdr as sisdr_lib
import sudo_rm_rf.dnn.experiments.utils.mixture_consistency as \
    mixture_consistency
import sudo_rm_rf.dnn.models.improved_sudormrf as improved_sudormrf
import sudo_rm_rf.dnn.models.attentive_sudormrf as attentive_sudormrf
import sudo_rm_rf.dnn.models.attentive_sudormrf_v2 as attentive_sudomrf_v2
import sudo_rm_rf.dnn.models.attentive_sudormrf_v3 as \
    attentive_sudomrf_v3
# import sudo_rm_rf.dnn.utils.cometml_loss_report as cometml_report
# import sudo_rm_rf.dnn.utils.cometml_log_audio as cometml_audio_logger
import sudo_rm_rf.dnn.models.sepformer as sepformer
import numpy as np

from pytorch_model_summary import summary

def normalize_tensor_wav(wav_tensor, eps=1e-8, std=None):
    mean = wav_tensor.mean(-1, keepdim=True)
    if std is None:
        std = wav_tensor.std(-1, keepdim=True)
    return (wav_tensor - mean) / (std + eps)
cuda0 = torch.device('cuda:0')

args = parser.get_args()
hparams = vars(args)
generators = dataset_setup.setup(hparams)

if hparams['separation_task'] == 'enh_single':
    hparams['n_sources'] = 1
else:
    hparams['n_sources'] = 2


os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([cad for cad in hparams['cuda_available_devices']])
# print(','.join(
#     [cad for cad in hparams['cuda_available_devices']]))

back_loss_tr_loss_name, back_loss_tr_loss = (
    'tr_back_loss_SISDRi',
    sisdr_lib.PITLossWrapper(sisdr_lib.PairwiseNegSDR("sisdr"),
                             pit_from='pw_mtx')
    # sisdr_lib.PermInvariantSISDR(batch_size=hparams['batch_size'],
    #                              n_sources=hparams['n_sources'],
    #                              zero_mean=True,
    #                              backward_loss=True,)
    #                              # improvement=True)
)

val_losses = {}
all_losses = []
for val_set in [x for x in generators if not x == 'train']:
    if generators[val_set] is None:
        continue
    val_losses[val_set] = {}
    all_losses.append(val_set + '_SISDRi')
    val_losses[val_set][val_set + '_SISDRi'] = sisdr_lib.PermInvariantSISDR(
        batch_size=hparams['batch_size'], n_sources=hparams['n_sources'],
        zero_mean=True, backward_loss=False, improvement=True,
        return_individual_results=True)
all_losses.append(back_loss_tr_loss_name)

if hparams['model_type'] == 'relu':
    model = improved_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                       in_channels=hparams['in_channels'],
                                       num_blocks=hparams['num_blocks'],
                                       upsampling_depth=hparams['upsampling_depth'],
                                       enc_kernel_size=hparams['enc_kernel_size'],
                                       enc_num_basis=hparams['enc_num_basis'],
                                       num_sources=hparams['n_sources'])
elif hparams['model_type'] == 'attention':
    model = attentive_sudormrf.SuDORMRF(out_channels=hparams['out_channels'],
                                        in_channels=hparams['in_channels'],
                                        num_blocks=hparams['num_blocks'],
                                        upsampling_depth=hparams['upsampling_depth'],
                                        enc_kernel_size=hparams['enc_kernel_size'],
                                        enc_num_basis=hparams['enc_num_basis'],
                                        n_heads=hparams['att_n_heads'],
                                        att_dims=hparams['att_dims'],
                                        att_dropout=hparams['att_dropout'],
                                        num_sources=hparams['n_sources'])
elif hparams['model_type'] == 'attention_v2':
    model = attentive_sudomrf_v2.SuDORMRF(
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        n_heads=hparams['att_n_heads'],
        att_dims=hparams['att_dims'],
        att_dropout=hparams['att_dropout'],
        num_sources=hparams['n_sources'])
elif hparams['model_type'] == 'attention_v3':
    model = attentive_sudomrf_v3.SuDORMRF(
        out_channels=hparams['out_channels'],
        in_channels=hparams['in_channels'],
        num_blocks=hparams['num_blocks'],
        upsampling_depth=hparams['upsampling_depth'],
        enc_kernel_size=hparams['enc_kernel_size'],
        enc_num_basis=hparams['enc_num_basis'],
        n_heads=hparams['att_n_heads'],
        att_dims=hparams['att_dims'],
        att_dropout=hparams['att_dropout'],
        num_sources=hparams['n_sources'])
elif hparams['model_type'] == 'sepformer':
    dff = 2 ** 11
    n_heads = 8
    N_intra_inter = 4
    model = sepformer.SepformerWrapper(
        encoder_kernel_size=16,
        encoder_in_nchannels=1,
        encoder_out_nchannels=256,
        masknet_chunksize=250,
        masknet_numlayers=2,
        masknet_norm="ln",
        masknet_useextralinearlayer=False,
        masknet_extraskipconnection=True,
        masknet_numspks=2,
        intra_numlayers=N_intra_inter,
        inter_numlayers=N_intra_inter,
        intra_nhead=n_heads,
        inter_nhead=n_heads,
        intra_dffn=dff,
        inter_dffn=dff,
        intra_use_positional=True,
        inter_use_positional=True,
        intra_norm_before=True,
        inter_norm_before=True,
    )
else:
    raise ValueError('Invalid model: {}.'.format(hparams['model_type']))

numparams = 0
for f in model.parameters():
    if f.requires_grad:
        numparams += f.numel()
# experiment.log_parameter('Parameters', numparams)
print('Trainable Parameters: {}'.format(numparams))

# print(summary(model, torch.zeros((2, 1, 32000)), show_input=True, show_hierarchical=False))
model = torch.nn.DataParallel(model).cuda(cuda0)
model.load_state_dict(torch.load("/home/dsi/yechezo/sudo_rm_rf_/weights/100e/improved_sudo_epoch_40"))


opt = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
# lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#     optimizer=opt, mode='max', factor=1. / hparams['divide_lr_by'],
#     patience=hparams['patience'], verbose=True)





tr_step = 0
val_step = 0
prev_epoch_val_loss = 0.
for i in range(hparams['n_epochs']):
    for val_set in [x for x in generators if not x == 'train']:
        if generators[val_set] is not None:
            model.eval()
            with torch.no_grad():
                for data in tqdm(generators[val_set],
                                 desc='Validation on {}'.format(val_set)):
                    m1wavs = data[0].cuda()
                    m1wavs = normalize_tensor_wav(m1wavs.sum(1)) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    clean_wavs = data[-1].cuda()

                    rec_sources_wavs = model(m1wavs.unsqueeze(1))

                    for loss_name, loss_func in val_losses[val_set].items():
                        l = loss_func(rec_sources_wavs,
                                      clean_wavs[:,0:2,:],
                                      initial_mixtures=m1wavs.unsqueeze(1))
                        # res_dic[loss_name]['acc'] += l.tolist()
                        print(f"{loss_name=}, {loss_func=}, {l=}")

    val_step += 1

    # l_name = 'train_val_SISDRi'
    # values = res_dic[l_name]['acc']
    # mean_metric = np.mean(values)
    # std_metric = np.std(values)
    #
    # print(f"{res_dic=}")
    # print(f"{values=}")
    # print(f"{mean_metric=}")
    # print(f"{std_metric=}")


    # for loss_name in res_dic:
    #     res_dic[loss_name]['acc'] = []
    # pprint(res_dic)

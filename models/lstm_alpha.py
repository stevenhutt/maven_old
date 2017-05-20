import pandas as pd
import numpy as np

from utils.load_data import load_data_tn
from utils.process_data import build_inputs_targets

from keras.models import Sequential
from keras.layers.recurrent import LSTM, SimpleRNN
from keras.layers import Dense, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras import metrics
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def use_lstm(data_dict, syms, device):
    stride = 5
    nt_n = 100
    nt_p = 1

    vld_tst_split = 0.1
    action = 'all'
    block = True
    inputs_dict, targets_dict = build_inputs_targets(data_dict, stride, nt_n, nt_p, vld_tst_split, action, block)


    inputs_trn = inputs_dict['trn']
    targets_trn = targets_dict['trn']
    inputs_vld = inputs_dict['vld']
    targets_vld = targets_dict['vld']

    nr_trn = inputs_trn.shape[0]
    permutation = np.random.permutation(nr_trn)
    inputs_trn = inputs_trn[permutation, :, :]
    targets_trn = targets_trn[permutation, :]

    nr, nt_n, ns = inputs_trn.shape
    nz = targets_trn.shape[1]

    nt = nt_n
    nh = 500
    wgt_reg = 0.00
    with K.tf.device('/gpu:' + str(device)):
        config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=config))
        model = Sequential()
        #model.add(LSTM(nh, input_shape=(nt, ns), kernel_regularizer=regularizers.l2(wgt_reg), return_sequences=True, name='lstm_1'))
        model.add(SimpleRNN(nh, input_shape=(nt, ns), kernel_regularizer=regularizers.l2(wgt_reg), return_sequences=False, name='lstm_2'))
        model.add(Dense(nz, input_dim=nh, kernel_regularizer=regularizers.l2(wgt_reg), name='dense_1'))
        model.add(Activation('linear', name='act_1'))

        def acc(targets, predicts):
            nr = K.tf.cast(K.tf.shape(targets)[0], K.tf.float32)
            nx = K.tf.cast(K.tf.shape(targets)[1], K.tf.float32)
            predicts_sgn = K.tf.cast(K.tf.sign(predicts), K.tf.float32)
            targets_sgn = K.tf.cast(K.tf.sign(targets), K.tf.float32)
            failures = K.tf.reduce_sum(K.tf.abs(predicts_sgn - targets_sgn) / 2.0)
            corrects = K.tf.reduce_sum(K.tf.abs(predicts_sgn + targets_sgn) / 2.0)
            zeros = nr * nx - corrects - failures
            return corrects / (nr*nx) * 100.0

        optimizer = optimizers.Adadelta(lr=1, rho=0.95, epsilon=1e-08, decay=0.0)
        model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=[acc])

        epochs = 100
        batch_size = 4096

        loss_hist = model.fit(inputs_trn, targets_trn, validation_data=(inputs_vld, targets_vld), batch_size=batch_size, epochs=epochs)


    plot_loss_hist = False
    if plot_loss_hist:
        fig, ax1 = plt.subplots()
        ax1.plot(loss_hist.history['loss'],  'g', label='loss')
        ax1.plot(loss_hist.history['val_loss'],  'g--', label='val_loss')
        ax1.set_xlabel('epochs')
        ax1.set_ylabel('loss')
        ax1.legend(loc='upper left', prop={'size':10})
        ax2 = ax1.twinx()
        ax2.plot(loss_hist.history['acc'],  'b', label='acc')
        ax2.plot(loss_hist.history['val_acc'],  'b--', label='val_acc')
        ax2.set_ylabel('acc')
        ax2.legend(loc='upper right', prop={'size':10})
        fig.tight_layout()
        plt.show()
    return model, inputs_trn, targets_trn, inputs_vld, targets_vld, syms


def check_lstm(model, inputs_trn, targets_trn, inputs_vld, targets_vld, syms):
    plt.ioff()
    num_syms = len(syms)
    # reduce num_points if slow machine
    num_points = inputs_vld.shape[0]
    inputs_sub_trn = inputs_trn[:num_points]
    targets_sub_trn = targets_trn[:num_points]
    inputs_sub_vld = inputs_vld[:num_points]
    targets_sub_vld = targets_vld[:num_points]


    predicts_sub_trn = model.predict(inputs_sub_trn)
    predicts_sub_vld = model.predict(inputs_sub_vld)
    return targets_sub_vld, predicts_sub_vld

    use_vld = True
    if use_vld:
        predicts = predicts_sub_vld
        targets = targets_sub_vld
    else:
        predicts = predicts_sub_trn
        targets = targets_sub_trn

    alphas = None
    if alphas is None:
        num_plt_rows = 3
        num_plt_cols = 10
        fig, axs = plt.subplots(num_plt_rows, num_plt_cols, figsize=(num_plt_cols*4,num_plt_rows*3.2))
        sym_idx = 0
        for row_idx in range(num_plt_rows):
            for col_idx in range(num_plt_cols):
                if sym_idx==num_syms:
                    break
                ax = axs[row_idx, col_idx]
                xs = targets[:, sym_idx]
                ys = predicts[:, sym_idx]
                ax_min = abs(np.min(xs))
                ax_max = abs(np.max(xs))
                ax_lim = max(ax_min, ax_max) / 4
                ax.scatter(xs, ys, alpha=0.2, s=0.1)
                ax.set_title(syms[sym_idx])
                ax.set_xlim(-ax_lim, ax_lim)
                ax.set_ylim(-ax_lim, ax_lim)
                ax.xaxis.set_tick_params(labelsize=8)
                ax.yaxis.set_tick_params(labelsize=8)
                ax.grid(True, which='both')
                ax.axhline(y=0, color='k')
                ax.axvline(x=0, color='k')
                sym_idx += 1
        fig.tight_layout()
        fig.show()
    if alphas is not None:
        predicts = np.dot(predicts_sub_vld, alphas)
        targets = np.dot(targets_sub_vld, alphas)
        plt.scatter(targets, predicts, alpha=0.1, s=0.1)
        plt.show()


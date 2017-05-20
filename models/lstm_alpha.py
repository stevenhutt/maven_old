import pandas as pd
import numpy as np
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

data_dir = '/home/shutt/repos/my_repos/maven/data/'

def load_data():
    # load market data
    filename = 'TN.csv'
    df_all = pd.read_csv(data_dir + filename)

    # remove 'SDF' and record symbol list
    del df_all['SDF']
    syms = list(df_all)[1:]

    # add columns converting str timestamp to time and date
    df_all['datetime'] = pd.to_datetime(df_all['Timestamp'])
    df_all['time'] = df_all['datetime'].map(lambda x: x.time())
    df_all['date'] = df_all['datetime'].map(lambda x: x.date())
    del df_all['Timestamp']

    # select rows with timestamps  07:00:00.0000000 <= ts <= 15:29:59.999999
    t_start = pd.Timestamp('07:00:00.0000000').time()
    t_end = pd.Timestamp('15:29:59.999999').time()
    df = df_all[(df_all['time'] > t_start) & (df_all['time'] < t_end)]

    # drop date/time columns
    df_syms = df[syms]
    data = df_syms.as_matrix()

    nr, ns = data.shape
    nt_n = 50
    nt_p = 10

    wins = np.zeros((nr-nt_p, nt_n, ns))
    targets = np.zeros((nr-nt_p, ns))
    for r_idx in range(nr-nt_n-nt_p):
        for t_idx in range(nt_n):
            wins[r_idx, t_idx, :] = data[r_idx+t_idx, :]
        targets[r_idx, :] = data[r_idx+nt_n+nt_p, :]

    inputs = np.diff(wins, axis=1).reshape((nr-nt_p, -1))

    nr, nx = inputs.shape
    permutation = np.random.permutation(nr)
    inputs = inputs[permutation, :]
    targets = targets[permutation, :]
    trunc_data = False
    if trunc_data:
        nr = 5000
        data = data[:nr,:]
        inputs = inputs[:nr, :]
        targets = targets[:nr, :]
    return data, inputs, targets, syms


def build_inputs_targets(data, stride, nt_n, nt_p, use_cmp=True):
    # data_shift[i] = data[i+stride]
    data_shift = data[stride:, :]
    # deltas[i] = data[i+stride] - data[i]
    deltas = data_shift[:, :] - data[:-stride, :]

    # compress dynamic range
    if use_cmp:
        deltas_cmp = np.sign(deltas) * np.log(np.abs(deltas) + 1.0)
    else:
        deltas_cmp = deltas

    nr, ns = deltas_cmp.shape
    nw = nr - nt_n - nt_p*stride
    fwd_offset = nt_p * stride
    inputs = np.zeros((nw, nt_n, ns))
    targets = np.zeros((nw, ns))
    for r_idx in range(nw):
        win_start = r_idx
        win_end = r_idx + nt_n
        # min time index: r_idx + stride
        # max time index: r_idx + nt_n + stride
        inputs[r_idx, :] = deltas_cmp[win_start:win_end, :]
        # min time index: r_idx + nt_n + stride
        # max time index: r_idx +m nt+n + stride + nt_p*stride
        targets[r_idx, :] = data_shift[win_end+fwd_offset, :] - data_shift[win_end, :]
    return inputs, targets

def build_inputs_targets2(data, stride, nt_n, nt_p):
    nr, ns = data.shape
    nw = nr - nt_n*stride - nt_p*stride
    inputs = np.zeros((nw, nt_n, ns))
    targets = np.zeros((nw, ns))
    for r_idx in range(nw):
        for tn_idx in range(nt_n):
            inputs[r_idx, tn_idx, :] = data[r_idx + (tn_idx+1)*stride, :] - data[r_idx + tn_idx*stride, :]
        targets[r_idx, :] = data[r_idx + nt_n*stride + nt_p*stride, :] - data[r_idx + nt_n*stride, :]
    inputs =  np.sign(inputs) * np.log(np.abs(inputs) + 1.0)
    return inputs, targets




def rw_data(nr, ns):
    np.random.seed(3893234)
    incs = np.random.normal(size=(nr, ns), loc=0.0, scale=0.1)
    data = np.cumsum(incs, axis=0)
    syms = [i for i in range(ns)]
    return data, syms

def use_lstm(data, syms, device):
    stride = 5
    nt_n = 100
    nt_p = 1

    inputs, targets = build_inputs_targets2(data, stride, nt_n, nt_p)

    unique, counts = np.unique(np.sign(targets), return_counts=True)
    print dict(zip(unique, counts))

    nr, nt, ns = inputs.shape
    nz = targets.shape[1]

    val_split = 0.2
    nr_trn = int(nr * (1.0 - val_split))
    inputs_trn = inputs[:nr_trn]
    targets_trn = targets[:nr_trn, :]
    inputs_vld = inputs[nr_trn:]
    targets_vld = targets[nr_trn:, :]

    permutation = np.random.permutation(nr_trn)
    inputs_trn = inputs_trn[permutation, :, :]
    targets_trn = targets_trn[permutation, :]

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


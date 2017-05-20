import pandas as pd
import numpy as np

from utils.load_data import load_data_tn

from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras import metrics
from keras import initializers
import matplotlib.pyplot as plt
import matplotlib.cm as cm
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
    # remove data prior to (2018, 8, 1) as is a gap in data dates
    t_start = pd.Timestamp('07:00:00.0000000').time()
    t_end = pd.Timestamp('15:29:59.999999').time()
    d_start = pd.Timestamp(2016, 8, 1).date()
    df = df_all[(df_all['time'] > t_start) & (df_all['time'] < t_end) & (df_all['date']>=d_start)]
    split_time = pd.Timestamp('13:00:00.0000000').time()
    df_am = df[df['time']<split_time]
    df_pm = df[df['time']>= split_time]
    # drop date/time columns
    df_syms = df[syms]
    data_all = df_syms.as_matrix()
    data_am = df_am[syms].as_matrix()
    data_pm = df_pm[syms].as_matrix()
    data_dict = {'all': data_all, 'am':data_am, 'pm':data_pm}
    return data_dict, syms

def build_inputs_targets(data, stride, nt_n, nt_p):
    # data_shift[i] = data[i+stride]
    data_shift = data[stride:, :]
    # deltas[i] = data[i+stride] - data[i]
    deltas = data_shift[:, :] - data[:-stride, :]

    # compress dynamic range
    deltas_cmp = np.sign(deltas) * np.log(np.abs(deltas) + 1.0)

    nr, ns = deltas_cmp.shape
    nw = nr - nt_n - nt_p*stride
    nx = nt_n * ns
    fwd_offset = nt_p * stride
    inputs = np.zeros((nw, nx))
    inputs_block = np.zeros((nw, nt_n, ns))
    targets = np.zeros((nw, ns))
    for r_idx in range(nw):
        win_start = r_idx
        win_end = r_idx + nt_n
        inputs[r_idx, :] = deltas_cmp[win_start:win_end, :].flatten()
        inputs_block[r_idx, :, :] = deltas_cmp[win_start:win_end, :]
        targets[r_idx, :] = data_shift[win_end+fwd_offset, :] - data_shift[win_end, :]
    return inputs, targets, inputs_block


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
    incs = np.random.normal(size=(nr, ns), loc=0.0, scale=0.1)
    data = np.cumsum(incs, axis=0)
    syms = [i for i in range(ns)]
    return data, syms

def check_inputs_targets(stride, nt_n, nt_p):
    nr = 1000
    ns = 1
    data = np.cumsum(np.cumsum(np.ones((nr, ns)), axis=0), axis=0)
    stride = 2
    nt_n = 2
    nt_p = 1
    inputs, targets = build_inputs_targets(data, stride, nt_n, nt_p)
    nr, nx = inputs.shape
    for r_idx in range(nr):
        for s_idx in range(ns):
            tgt = targets[r_idx, s_idx]
            if tgt in inputs[r_idx, :]:
                print r_idx, s_idx, tgt
                raw_input('oops')
    return data, inputs, targets

def stats(inputs_nf, targets):
    ds = ['trn', 'vld', 'tst']
    for d in ds:
        xs = inputs[d]
        ys = targets[d]
        xs_mean = np.mean(xs, axis=0)
        plt.scatter(xs_mean, ys)
        plt.show(block=True)



def use_nn(data_dict, syms, device):
    K.tf.reset_default_graph()
    # each time step is 10s
    stride = 5
    nt_n = 100
    nt_p = 1

    use_tgt_syms = syms #['BMW']
    use_tgt_syms_mask = [int(x in use_tgt_syms) for x in syms]

    action = 'all'

    if action == 'cat':
        data_all = data_dict['all']
        inputs, targets, inputs_block = build_inputs_targets(data_all, stride, nt_n, nt_p)
        nr = inputs.shape[0]
        vld_tst_split = 0.1
        nr_vld = int(nr * vld_tst_split)
        nr_tst = nr_vld
        nr_trn = nr - nr_vld - nr_tst
        inputs_trn = inputs[:nr_trn]
        targets_trn = targets[:nr_trn, :]
        inputs_vld = inputs[nr_trn:nr_trn+nr_vld, :]
        targets_vld = targets[nr_trn:nr_trn+nr_vld, :]
        inputs_tst = inputs[nr_trn+nr_vld:, :]
        targets_tst = targets[nr_trn+nr_vld:, :]
    elif action == 'ampm'
        data_am = data_dict['am']
        data_pm = data_dict['pm']
        inputs_trn, targets_trn, _ = build_inputs_targets(data_am, stride, nt_n, nt_p)
        inputs_vld, targets_vld, _ = build_inputs_targets(data_pm, stride, nt_n, nt_p)
        inputs_tst = None
        targets_tst = None
    else:
        data_all = data_dict['all']
        inputs, targets, inputs_block = build_inputs_targets(data_all, stride, nt_n, nt_p)
        nr = inputs.shape[0]
        vld_tst_split = 0.1
        nr_vld = int(nr * vld_tst_split)
        nr_tst = nr_vld
        nr_trn = nr - nr_vld - nr_tst
        inputs_trn = inputs[:nr_trn]
        targets_trn = targets[:nr_trn, :]
        inputs_vld = inputs[nr_trn:nr_trn+nr_vld, :]
        targets_vld = targets[nr_trn:nr_trn+nr_vld, :]
        inputs_tst = inputs[nr_trn+nr_vld:, :]
        targets_tst = targets[nr_trn+nr_vld:, :]

    nr_trn, nx = inputs_trn.shape
    nz = targets_trn.shape[1]

    inputs = {'trn':inputs_trn, 'vld':inputs_vld, 'tst':inputs_tst}
    targets = {'trn':targets_trn, 'vld':targets_vld, 'tst':targets_tst}

    permutation = np.random.permutation(nr_trn)
    inputs_trn = inputs_trn[permutation, :]
    targets_trn = targets_trn[permutation, :]

    nh_1 = 3000
    nh_2 = 2000
    nh_3 = 1000
    reg_wgt = 1e-3
    dp_rate = 0.5
    leaky_alpha = 0.2
    use_dropout = True
    with K.tf.device('/gpu:' + str(device)):
        config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=config))

        kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05)
        kernel_initializer = initializers.glorot_normal()
        bias_initializer = initializers.Zeros()
        kernel_regularizer = regularizers.l1(reg_wgt)
        #activation_nl = LeakyReLU(alpha=leaky_alpha)
        activation_nl = Activation('tanh')

        xs_pred = Input(shape=(nx,), dtype='float32', name='xs_pred')

        hs_pred_1 = Dense(nh_1, input_shape=(nx,),
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name='dense_1')(xs_pred)
        if use_dropout:
            hs_pred_2 = Dropout(dp_rate)(hs_pred_1)
        else:
            hs_pred_2 = hs_pred_1
        hs_pred_3 = activation_nl(hs_pred_2)

        hs_pred_4 = Dense(nh_2,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name='dense_2')(hs_pred_3)
        if use_dropout:
            hs_pred_5 = Dropout(dp_rate)(hs_pred_4)
        else:
            hs_pred_5 = hs_pred_4
        hs_pred_6 = activation_nl(hs_pred_5)

        hs_pred_7 = Dense(nh_3,
                          kernel_initializer=kernel_initializer,
                          bias_initializer=bias_initializer,
                          kernel_regularizer=kernel_regularizer,
                          name='dense_3')(hs_pred_6)
        if use_dropout:
            hs_pred_8 = Dropout(dp_rate)(hs_pred_7)
        else:
            hs_pred_8 = hs_pred_7
        hs_pred_9 = activation_nl(hs_pred_8)

        hs_pred_10 = Dense(nz,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           name='dense_4')(hs_pred_9)


        zs_pred = hs_pred_10

        model = Model(inputs=[xs_pred], outputs=[zs_pred])
        optimizer = optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)

        def mse_loss(targets_, predicts_):
            error = K.tf.reduce_mean((use_tgt_syms_mask*(targets_ - predicts_))**2)
            return error

        def soft_acc(targets, predicts):
            nr = K.tf.cast(K.tf.shape(targets)[0], K.tf.float32)
            nx = K.tf.cast(K.tf.shape(targets)[1], K.tf.float32)
            predicts_soft_sgn = K.tf.cast(K.tf.tanh(10.0*predicts), K.tf.float32)
            targets_soft_sgn = K.tf.cast(K.tf.tanh(10.0*targets), K.tf.float32)
            failures = K.tf.reduce_sum(K.tf.abs(predicts_soft_sgn - targets_soft_sgn) / 2.0)
            corrects = K.tf.reduce_sum(K.tf.abs(predicts_soft_sgn + targets_soft_sgn) / 2.0)
            return failures / (nr*nx) * 100.0

        def acc(targets, predicts):
            nr = K.tf.cast(K.tf.shape(targets)[0], K.tf.float32)
            nx = K.tf.cast(K.tf.shape(targets)[1], K.tf.float32)
            predicts_sgn = K.tf.cast(K.tf.sign(predicts), K.tf.float32)
            targets_sgn = K.tf.cast(K.tf.sign(targets), K.tf.float32)
            failures = K.tf.reduce_sum(K.tf.abs(predicts_sgn - targets_sgn) / 2.0)
            corrects = K.tf.reduce_sum(K.tf.abs(predicts_sgn + targets_sgn) / 2.0)
            return corrects / (nr*nx) * 100.0

        model.compile(loss=mse_loss, optimizer=optimizer, metrics=[acc])

        num_epochs = 100
        batch_size = 4096
        loss_hist = model.fit(inputs_trn, targets_trn,
                                        validation_data=(inputs_vld, targets_vld),
                                        batch_size=batch_size, epochs=num_epochs)
    plot_trn = False
    if plot_trn:
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
    return model, inputs, targets, syms

def check_wgts(model, inputs, targets, syms):
    plt.ioff()
    ns = len(syms)
    nt_n = inputs['trn'].shape[1] / ns
    Ws = []
    bs = []
    for layer in model.layers:
        if 'dense' in layer.name:
            W = layer.get_weights()[0]
            b = layer.get_weights()[1]
            Ws.append(W)
            bs.append(b)
    num_plt_rows = 3
    num_plt_cols = 10
    fig, axs = plt.subplots(num_plt_rows, num_plt_cols, figsize=(num_plt_cols*4,num_plt_rows*3.2))
    sym_idx = 0
    for row_idx in range(num_plt_rows):
        for col_idx in range(num_plt_cols):
            ax = axs[row_idx, col_idx]
            W_block = Ws[0].reshape((nt_n, ns, -1))
            cax = ax.matshow(W_block[:, sym_idx, :], cmap=plt.cm.Greys, interpolation='nearest', aspect='auto')
            ax.set_title(syms[sym_idx])
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.grid(True, which='both')
            sym_idx += 1
    fig.tight_layout()
    fig.colorbar(cax)
    fig.show()
    return Ws, bs

def check_nn_alpha(model, inputs_, targets_, syms, ds):
    plt.ioff()
    num_syms = len(syms)
    inputs = inputs_[ds]
    targets = targets_[ds]
    predicts = model.predict(inputs)

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
            ax_lim = max(ax_min, ax_max)
            #ax_lim = max(-1, 1)
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


def plot_hists(data, inputs, syms):
    num_syms = inputs.shape[1]
    deltas = np.diff(data, axis=0)
    num_plt_rows = 3
    num_plt_cols = 10
    fig, axs = plt.subplots(num_plt_rows, num_plt_cols, figsize=(num_plt_cols*4,num_plt_rows*3.2))
    sym_idx = 0
    for row_idx in range(num_plt_rows):
        for col_idx in range(num_plt_cols):
            if sym_idx==num_syms:
                break
            ax = axs[row_idx, col_idx]
            ax.hist(inputs[:, sym_idx], 1000)
            ax.set_title(syms[sym_idx])
            ax.xaxis.set_tick_params(labelsize=8)
            ax.yaxis.set_tick_params(labelsize=8)
            ax.grid(True, which='both')
            sym_idx += 1
    fig.tight_layout()
    fig.show()


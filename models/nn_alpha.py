import numpy as np

from utils.load_data import load_data_tn
from utils.process_data import build_inputs_targets
from utils.process_results import check_wgts, check_nn_alpha

from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Dense, Dropout, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Model
from keras import backend as K
from keras import regularizers
from keras import optimizers
from keras import metrics
from keras import initializers
import matplotlib.pyplot as plt


def use_nn(data_dict, syms, device):
    K.tf.reset_default_graph()
    # each time step is 10s
    stride = 5
    nt_n = 100
    nt_p = 1

    use_tgt_syms = syms #['BMW']
    use_tgt_syms_mask = [int(x in use_tgt_syms) for x in syms]

    vld_tst_split = 0.1
    action = 'ampm'
    inputs_dict, targets_dict = build_inputs_targets(data_dict, stride, nt_n, nt_p, vld_tst_split, action)


    nr_trn, nx = inputs_dict['trn'].shape
    nz = targets_dict['trn'].shape[1]

    permutation = np.random.permutation(nr_trn)
    inputs_trn = inputs_dict['trn'][permutation, :]
    targets_trn = targets_dict['trn'][permutation, :]
    inputs_vld = inputs_dict['vld']
    targets_vld = targets_dict['vld']


    nh_1 = 500
    nh_2 = 2000
    nh_3 = 1000
    reg_wgt = 3e-4
    dp_rate = 0.5
    leaky_alpha = 0.2
    use_dropout = True
    with K.tf.device('/gpu:' + str(device)):
        config=K.tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True
        K.set_session(K.tf.Session(config=config))

        kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.1)
        kernel_initializer = initializers.glorot_normal()
        bias_initializer = initializers.Zeros()
        kernel_regularizer = regularizers.l1(reg_wgt)
        activation_nl = LeakyReLU(alpha=leaky_alpha)
        #activation_nl = Activation('tanh')

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
        #hs_pred_2 = BatchNormalization()(hs_pred_2)
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
        hs_pred_5 = BatchNormalization()(hs_pred_5)
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
        hs_pred_8 = BatchNormalization()(hs_pred_8)
        hs_pred_9 = activation_nl(hs_pred_8)

        hs_pred_10 = Dense(nz,
                           kernel_initializer=kernel_initializer,
                           bias_initializer=bias_initializer,
                           kernel_regularizer=kernel_regularizer,
                           name='dense_4')(hs_pred_3)


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

        num_epochs = 30
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
    return model, inputs_dict, targets_dict, syms


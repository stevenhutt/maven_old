import numpy as np
import matplotlib.pyplot as plt




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

import numpy as np

def split_data(inputs, targets, vld_tst_split):
    nr = inputs.shape[0]
    nr_vld = int(nr * vld_tst_split)
    nr_tst = nr_vld
    nr_trn = nr - nr_vld - nr_tst
    inputs_trn = inputs[:nr_trn]
    targets_trn = targets[:nr_trn, :]
    inputs_vld = inputs[nr_trn:nr_trn+nr_vld, :]
    targets_vld = targets[nr_trn:nr_trn+nr_vld, :]
    inputs_tst = inputs[nr_trn+nr_vld:, :]
    targets_tst = targets[nr_trn+nr_vld:, :]

    inputs_dict = {'trn':inputs_trn, 'vld':inputs_vld, 'tst':inputs_tst}
    targets_dict = {'trn':targets_trn, 'vld':targets_vld, 'tst':targets_tst}

    return inputs_dict, targets_dict

def build_inputs_targets(data_dict, stride, nt_n, nt_p, vld_tst_split, action='all'):
    data = data_dict[action]
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

    inputs_dict, targets_dict = split_data(inputs, targets, vld_tst_split)
    return inputs_dict, targets_dict


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

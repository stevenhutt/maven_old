import numpy as np
import pandas as pd
data_dir = '/home/shutt/repos/my_repos/maven/data/'

def load_data_tn():
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

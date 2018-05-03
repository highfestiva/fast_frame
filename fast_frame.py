'''A Pandas-to-numpy optimizer, to allow for speed slicing. The syntax is not
   the same (or nearly as comprehensive), but the performance compared to .iloc
   is a hundredfold or so.

>>> ff = FastFrame(df)
>>> ff['D']
array([0., 1., 5.])
>>> ff[0:2]
     A    B   C    D
0  NaN  2.0 NaN  0.0
1  3.0  4.0 NaN  1.0
>>> type(ff[0:2])
<class 'fast_frame.FastFrame'>
'''

import numpy as np
import pandas as pd


class FastFrame:
    def __init__(self, df=[]):
        self.columns = [col for col in df]
        self.col2idx = {col:i for i,col in enumerate(self.columns)}
        try:
            sers = [df[col].values for col in self.columns]
        except:
            sers = [df[col] for col in self.columns]
        self.col_vals = np.array(sers)
        self.row_vals = self.col_vals.T

    def append(self, frames):
        self.row_vals = np.concatenate([self.row_vals]+[f.row_vals for f in frames])
        self.col_vals = self.row_vals.T

    def df(self):
        return pd.DataFrame(self.row_vals, columns=self.columns)

    @property
    def iloc(self):
        return self

    def __iter__(self):
        for c in self.columns:
            yield c

    def __getitem__(self, arg):
        if type(arg) == slice:
            ff = FastFrame()
            ff.columns = self.columns
            ff.col2idx = self.col2idx
            ff.row_vals = self.row_vals[arg]
            ff.col_vals = ff.row_vals.T
            return ff
        idx = self.col2idx[arg]
        return self.col_vals[idx]

    def __setitem__(self, col, data):
        if col in self.col2idx:
            idx = self.col2idx[col]
            self.col_vals[idx] = data
        else:
            self.col_vals = np.append(self.col_vals, [data], axis=0)
            self.row_vals = self.col_vals.T
            self.col2idx[col] = len(self.columns) # keep consistent by ensuring value xforms succeeded
            self.columns.append(col)

    def __len__(self):
        return len(self.row_vals)

    def __str__(self):
        return str(self.df())

    def __repr__(self):
        return repr(self.df())


def fast_concat(frames):
    f = frames[0]
    f.append(frames[1:])
    return f

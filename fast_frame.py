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
            sers = [np.array(df[col]) for col in self.columns]
        self.col_vals = sers
        self.row_vals = None

    def append(self, *frames):
        self.row_vals = np.concatenate([self.rows]+[f.rows for f in frames])
        self.col_vals = self.row_vals.T
        return self

    @property
    def df(self):
        return pd.DataFrame({col: self.col_vals[self.col2idx[col]] for col in self.columns})

    @property
    def rows(self):
        if self.row_vals is None:
            self.row_vals = np.array(self.col_vals).T
        return self.row_vals

    @property
    def iloc(self):
        return self

    def __iter__(self):
        for c in self.columns:
            yield c

    def __getattr__(self, attr):
        i = self.col2idx.get(attr, -1)
        if i >= 0:
            return self.col_vals[i]
        return super().__getattr__(attr)

    def __getitem__(self, arg):
        if type(arg) == slice:
            ff = FastFrame()
            ff.columns = list(self.columns)
            ff.col2idx = dict(self.col2idx)
            ff.row_vals = self.rows[arg]
            ff.col_vals = ff.row_vals.T
            return ff
        if type(arg) == list:
            ff = FastFrame()
            ff.columns = list(arg)
            ff.col2idx = {col:i for i,col in enumerate(arg)}
            ff.col_vals = [self.col_vals[self.col2idx[col]] for col in arg]
            ff.row_vals = None
            return ff
        idx = self.col2idx[arg]
        return self.col_vals[idx]

    def __setitem__(self, col, data):
        if col in self.col2idx:
            idx = self.col2idx[col]
            self.col_vals[idx] = np.array(data)
        else:
            if type(self.col_vals) != list:
                self.col_vals = list(self.col_vals)
            self.col_vals.append(np.array(data))
            self.row_vals = None
            self.col2idx[col] = len(self.columns) # keep consistent by ensuring value xforms succeeded
            self.columns.append(col)

    def __len__(self):
        return len(self.col_vals[0]) if len(self.col_vals) else 0

    def __str__(self):
        return str(self.df)

    def __repr__(self):
        return repr(self.df)


def fast_concat(*frames):
    f = frames[0]
    f.append(frames[1:])
    return f


def to_frame(df):
    return df if df is FastFrame else FastFrame(df)

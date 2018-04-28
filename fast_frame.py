'''A Pandas-to-numpy optimizer, to allow for speed slicing. The syntax is not
   the same, but the performance compared to .iloc is a hundredfold or so.

>>> ff = FastFrame(df)
>>> ff['D']
array([0., 1., 5.])
>>> ff[0:2]
A       B       C       D
[[nan  2. nan  0.]
 [ 3.  4. nan  1.]]
>>> type(ff[0:2])
<class 'fast_frame.FastFrame'>
'''

import numpy as np


class FastFrame:
    def __init__(self, df=[]):
        if len(df):
            self.cols = [col for col in df]
            self.col2idx = {col:i for i,col in enumerate(self.cols)}
            sers = [df[col] for col in self.cols]
            self.col_vals = np.array(sers)
            self.row_vals = self.col_vals.T

    def __iter__(self):
        for c in self.cols:
            yield c

    def __getitem__(self, arg):
        if type(arg) == slice:
            ff = FastFrame()
            ff.cols = self.cols
            ff.col2idx = self.col2idx
            ff.row_vals = self.row_vals[arg]
            ff.col_vals = ff.row_vals.T
            return ff
        idx = self.col2idx[arg]
        return self.col_vals[idx]

    def __setitem__(self, col, data):
        self.col_vals = np.append(self.col_vals, [data], axis=0)
        self.row_vals = self.col_vals.T
        self.col2idx[col] = len(self.cols) # keep consistent by ensuring value xforms succeeded
        self.cols.append(col)

    def __len__(self):
        return len(self.row_vals)

    def __str__(self):
        header = '\t'.join(self.cols)
        return header + '\n' + str(self.row_vals)

    def __repr__(self):
        return str(self)

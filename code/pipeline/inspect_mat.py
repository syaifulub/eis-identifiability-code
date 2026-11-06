"""Print the structure of a MATLAB .mat file (variables, struct fields, array shapes, dtypes, a few values),
to write the converter of an external wide-band dataset to the CSV format of wideband_test.py.
Usage: python -I inspect_mat.py file.mat"""
import sys, numpy as np, scipy.io as sio

def show(x, name, d=0, maxd=5):
    pad = '  ' * d
    if isinstance(x, np.ndarray) and x.dtype.names:
        print(f'{pad}{name}: struct array {x.shape} fields={x.dtype.names}')
        if x.size and d < maxd:
            for fn in x.dtype.names: show(x.flat[0][fn], f'{name}[0].{fn}', d + 1)
    elif isinstance(x, np.ndarray) and x.dtype == object:
        print(f'{pad}{name}: cell {x.shape}')
        if x.size and d < maxd: show(x.flat[0], f'{name}{{0}}', d + 1)
    elif isinstance(x, np.ndarray):
        v = x.ravel()[:4]
        print(f'{pad}{name}: {x.dtype} {x.shape} e.g. {v}')
    else:
        print(f'{pad}{name}: {type(x).__name__} {str(x)[:80]}')

m = sio.loadmat(sys.argv[1], squeeze_me=False, struct_as_record=True)
for k, v in m.items():
    if not k.startswith('__'): show(v, k)

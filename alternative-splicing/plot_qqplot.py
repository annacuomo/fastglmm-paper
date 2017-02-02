import pandas as pd
import numpy as np
from limix_genetics import qqplot
import bokeh
from bokeh.io import output_notebook
from bokeh.io import output_file

DFn = pd.read_pickle('chrom_null_ready.pkl')
output_file("qqplot.html")
qqplot(DFn, colors={'lmm-rank':'#FF3399', 'qep':'#E24A33', 'lmm':'#348ABD'}, atleast_points=1.0, tools=['save'])

import pandas as pd
from preprocess import *

preprocessCls = Preprocessing('data/test.csv', isClassification=False)
preprocessCls.preprocess()
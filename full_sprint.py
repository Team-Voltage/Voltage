# Open train.parquet

import pandas as pd
import pyarrow.parquet as pq
train = pq.read_table('/home/jlartey/train.parquet')
train = train.to_pandas()
print(train.head())

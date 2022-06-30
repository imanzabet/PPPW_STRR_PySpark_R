import pandas as pd
import pyarrow

# read
# df = pd.read_parquet('C:/Users/iZabett/Downloads/part-00000-267787a0-c0be-432d-bd36-4778afe653cd-c000.snappy.parquet')
df = pd.read_parquet('C:/Users/iZabett/Downloads/SFRAB17.parquet')

# write
# df.to_parquet('my_newfile.parquet')

df.head()
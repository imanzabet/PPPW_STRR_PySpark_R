import pandas as pd
import datetime
import boto3
import io
def Write_Parquet(data, path, filename):
  """ Write input data frames in parquet format"""
  import pandas as pd
  import pyarrow
  import os
  ## cache should be replaced by run id or timestamp in future
  data = pd.DataFrame(data)
  data.to_parquet(path+filename)
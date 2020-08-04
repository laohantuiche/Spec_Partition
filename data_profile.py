import pandas as pd
import pandas_profiling

data = pd.read_csv('ST_or_CS_44.csv', sep=',')
profile = data.profile_report(title='stream or cache sensitive')
profile.to_file(output_file='profile.html')

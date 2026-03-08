import pandas as pd

def analyze_dataset(file_path):
    df=pd.read_csv(file_path)

    info={
        'num_samples': len(df),
        'num_features': len(df.columns)-1,
        'missing_values': df.isnull().sum().sum()
    }

    target=df.columns[-1]

    if df[target].nunique()<20:
        info['task']='classification'
        info['num_classes']=df[target].nunique()
    else:
        info['task']='regression'

    print(df.head())


    return df, info
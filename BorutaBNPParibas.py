import pandas as pd
from boruta import BorutaPy
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
import category_encoders as ce


# Function to convert to hexavigesimal base
def az_to_int(az, nanVal=None):
    if az==az:  #catch NaN
        hv = 0
        for i in range(len(az)):
            hv += (ord(az[i].lower())-ord('a')+1)*26**(len(az)-1-i)
        return hv
    else:
        if nanVal is not None:
            return nanVal
        else:
            return az


def clean_data(df):
    df.v22 = df.v22.apply(az_to_int)
    df.drop(columns='ID', inplace=True)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    df.dropna(subset=['target'], inplace=True)    
    for cat_col in cat_cols:
        df[cat_col].fillna("__MISS__", inplace=True)
        df[cat_col] = df[cat_col].astype("category")
    return df


if __name__ == '__main__':
    df = pd.read_csv("./dataset/train.csv.zip", compression="zip")
    train, valid = train_test_split(df, train_size=0.7)

    print(f"Train shape {train.shape}")
    print(f"Test shape {valid.shape}")

    train = clean_data(train)
    valid = clean_data(valid)
    cat_cols = train.select_dtypes(include=['category']).columns.tolist()
    cont_cols = train.select_dtypes(include=['number']).columns.tolist()

    imputer = SimpleImputer(fill_value=-1)
    train[cont_cols] = imputer.fit_transform(train[cont_cols])
    valid[cont_cols] = imputer.transform(valid[cont_cols])

    y_train = train.pop('target')
    y_valid = valid.pop('target')

    cat_encoder = ce.TargetEncoder(cols=cat_cols,  smoothing=100)
    train = cat_encoder.fit_transform(train, y=y_train)
    valid = cat_encoder.transform(valid, y=y_valid)

    rf = RandomForestClassifier(n_jobs=-2, class_weight='balanced', max_depth=10, random_state=1234)
    feat_selector = BorutaPy(rf, n_estimators='auto', verbose=1, random_state=1234)
    feat_selector.fit(train.values, y_train.values)

    selected_features = train.columns[feat_selector.support_].tolist() + train.columns[feat_selector.support_weak_].tolist()
    train_filtered = train[selected_features]
    train_filtered['target'] = y_train
    valid_filtered = valid[selected_features]
    valid_filtered['target'] = y_valid

    train_filtered.to_csv("./dataset/train_filtered.csv.zip", index=False, compression="zip")
    valid_filtered.to_csv("./dataset/valid_filtered.csv.zip", index=False, compression="zip")


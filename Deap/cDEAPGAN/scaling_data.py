from sklearn.preprocessing import StandardScaler
from joblib import dump

scaler = MinMaxScaler(feature_range=(-1, 1))  # Declaring the scaler
scaler.fit(trainx_min)  # Fitting the scaler
trainx_min_scaled = scaler.transform(trainx_min)  # Transforming the DATA

scalers = {}
for i in range(X_train.shape[1]):
    scalers[i] = StandardScaler()
    X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :])
    dump(scaler, 'minmax_scaler.bin', compress=True)

for i in range(X_test.shape[1]):
    X_test[:, i, :] = scalers[i].transform(X_test[:, i, :])
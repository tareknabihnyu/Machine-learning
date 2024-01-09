
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error

data = np.loadtxt('data.csv', delimiter=',')

X = data[:, :-1]
y = data[:, -1]

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

joblib.dump(scaler, 'my_scaler11.pkl')

def create_model():


    layers = [
        Dense(256, activation='relu', input_shape=(X.shape[1],), kernel_regularizer=l2(0.5)),
        BatchNormalization(),
    
    ]
    
    layers_nodes = [257, 257, 257, 257, 257, 257, 257, 257]


    for nodes in layers_nodes:
        layers.extend([
            Dense(nodes, activation='relu', kernel_regularizer=l2(0.5)),
            BatchNormalization(),
            Dropout(0.03),
        ])


    layers.append(Dense(1))

    model = Sequential(layers)
    model.compile(optimizer=Adam(learning_rate=0.0003), loss='mean_squared_error')

    return model


early_stopping = EarlyStopping(monitor='val_loss', patience=20)

num_folds = 5
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

model = create_model()

fold_no = 1
y_tests = []
y_preds = []
losses = []
history = {'loss': [], 'val_loss': []}
for train_index, test_index in kf.split(X, y):
    
    h = model.fit(X[train_index], y[train_index], validation_data=(X[test_index], y[test_index]), epochs=25, batch_size=512, callbacks=[early_stopping])
    
    history['loss'].extend(h.history['loss'])
    history['val_loss'].extend(h.history['val_loss'])


    y_pred = model.predict(X[test_index])
    y_preds.append(y_pred)
    y_tests.append(y[test_index])
    

    loss = model.evaluate(X[test_index], y[test_index], verbose=0)
    losses.append(loss)
    print(f'Loss for fold {fold_no}: {loss}')
    

    fold_no = fold_no + 1

mean_loss_cv = np.mean(losses)
print(f'Mean loss during cross-validation: {mean_loss_cv}')

y_tests = np.concatenate(y_tests)
y_preds = np.concatenate(y_preds)
total_r2 = r2_score(y_tests, y_preds)
total_mse = mean_squared_error(y_tests, y_preds)
print(f'Total R2 Score: {total_r2}')
print(f'Total Mean Squared Error: {total_mse}')

model.save('my_model37.h5')

data_test = np.loadtxt('testing.csv', delimiter=',')
X_test = data_test[:, :-1]
y_test = data_test[:, -1]

X_test = scaler.transform(X_test)

test_loss = model.evaluate(X_test, y_test, verbose=0)
history['val_loss'].append(test_loss)

y_test_pred = model.predict(X_test)

r2_test = r2_score(y_test, y_test_pred)
mse_test = mean_squared_error(y_test, y_test_pred)
print(f'Test R2 Score: {r2_test}')
print(f'Test Mean Squared Error: {mse_test}')



plt.figure(figsize=(12, 6))
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.savefig('hello37')

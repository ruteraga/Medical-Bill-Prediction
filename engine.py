# get all the needed libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import keras
from keras import layers

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

#get data
df = pd.read_csv('insurance.csv')

#encode the object datatypes into integers that can be studied by the model
label_Encoder = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        if len(df[column].unique()) <= 4:
            df[column] = label_Encoder.fit_transform(df[column])
        else:
            d = pd.get_dummies(df[column])
            df = pd.concat([df, d], axis=1)
            df = df.drop(column, axis=1)

#prepare datasets for training and testing
shuffled_df= df.sample(frac=1, random_state=42)
train_df, test_df = train_test_split(shuffled_df, test_size=0.2, random_state=42)

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

train_labels = train_df.pop('expenses')
test_labels = test_df.pop('expenses')

#model training
normalizer = layers.Normalization()
normalizer.adapt(np.array(train_df))

model = keras.Sequential([
    normalizer,
    layers.Dense(16),
    layers.Dense(4),
    layers.Dropout(.2),
    layers.Dense(1),
])

model.compile(
    optimizer= keras.optimizers.Adam(learning_rate=0.1),
    loss='mae',
    metrics=['mae', 'mse']
)
# Models data
model.build()
model.summary()

history = model.fit(
    train_df,
    train_labels,
    epochs=100,
    validation_split=0.5,
    verbose=0, 
)

loss, mae, mse = model.evaluate(test_df, test_labels, verbose=2)
test_predictions = model.predict(test_df).flatten()
#Saving model
model.save('health_model.keras')





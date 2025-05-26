#importing required library
import pandas as pd 
import tensorflow as tf 
import tensorflow_hub as hub 
import tensorflow_text as text 
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

#Loading data and encoding Category
data = pd.read_csv('spam.csv')
data.Category = data.Category.map({'spam':1,'ham':0})

#resolving imbalanced data 
spam = data[data.Category==1]
ham = data[data.Category==0].sample(len(spam))
data = pd.concat([spam,ham])

#splitting data 
x_train, x_test, y_train, y_test = train_test_split(
    data['Message'], 
    data['Category'], 
    stratify=data['Category'],
    shuffle=True,
    random_state=42
)

# functional model 
preprocessor = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")

#__BERT__
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text_input')
preprocessed_text = preprocessor(text_input)
outputs = encoder(preprocessed_text)

#__ANN__
net = tf.keras.layers.Dropout(0.1)(outputs['pooled_output'])
net = tf.keras.layers.Dense(10, activation='relu')(net)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid', name='output')(net)

model = tf.keras.Model(inputs=[text_input], outputs=[output_layer], name='detector')
model.summary()

METRICS = [
    tf.keras.metrics.BinaryAccuracy(name='accuracy'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall')
]

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=METRICS
)

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',         # Monitor validation loss
    patience=3,                # Number of epochs with no improvement before stopping
    restore_best_weights=True  # Restore model weights from the epoch with best value
)

# Train with early stopping and validation split
history = model.fit(
    x_train, 
    y_train, 
    epochs=13,                  # Increased max epochs since we're using early stopping
    batch_size=32,
    validation_split=0.2,       
    callbacks=[early_stopping]  
)

# Evaluate with batch size
model.evaluate(x_test, y_test, batch_size=32)

model.save('spam_detector.h5')
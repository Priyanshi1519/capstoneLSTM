# ===== LSTM Model Setup =====
from tensorflow.keras import layers, models, callbacks

# Define LSTM Model
model = models.Sequential()

# Adding LSTM Layer
model.add(layers.LSTM(units=50, activation='relu', input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))

# Output Layer
model.add(layers.Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Model Summary
model.summary()

# ===== Model Training =====
callbacks_list = [
    callbacks.ModelCheckpoint("best_model.h5", save_best_only=True),
    callbacks.EarlyStopping(monitor='val_loss', patience=10)
]

# Train the model
history = model.fit(
    X_train_lstm, y_train_lstm,
    epochs=50,
    batch_size=64,
    validation_data=(X_test_lstm, y_test_lstm),
    callbacks=callbacks_list
)

# Log training progress
for epoch in range(len(history.history['accuracy'])):
    log(f"Epoch {epoch+1}: Accuracy: {history.history['accuracy'][epoch]} - Loss: {history.history['loss'][epoch]}")

# ===== Model Evaluation =====
test_loss, test_accuracy = model.evaluate(X_test_lstm, y_test_lstm, verbose=1)
log(f"Test Loss: {test_loss}")
log(f"Test Accuracy: {test_accuracy}")

# ===== Fraud Prediction & Blockchain Logging =====
log("Making predictions on test data and logging to blockchain...")
predictions = model.predict(X_test_lstm)

# Log predictions to blockchain and print logs
for i, (x, pred) in enumerate(zip(X_test_lstm, predictions)):
    is_fraud = 1 if pred > 0.5 else 0  # Assuming threshold of 0.5 for binary classification
    log_transaction_to_blockchain(f"pred_test_{i}", x[0][0], is_fraud)
    log(f"Test {i}: Predicted Fraud: {is_fraud}")

# ===== Blockchain Transaction Logging Test =====
# Log a sample transaction to the blockchain after model training
receipt = log_transaction_to_blockchain("test_id", 100, False)
log(f"Transaction Receipt: {receipt}")

# ===== Event Listening (optional) =====
def listen_for_events():
    event_filter = contract.events.TransactionLogged.createFilter(fromBlock='latest')
    while True:
        events = event_filter.get_new_entries()
        if events:
            for event in events:
                log(f"New Event: {event}")

# Start event listener in a separate thread
from threading import Thread
event_thread = Thread(target=listen_for_events)
event_thread.start()

# ===== Save the Model =====
model.save("fraud_detection_model.h5")
log("Model saved successfully.")

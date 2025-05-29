import os
import sys
import pickle
import tensorflow as tf
from sklearn.metrics import f1_score, precision_score, roc_auc_score
import numpy as np
from tensorflow.keras import layers, models
from web3 import Web3
from datetime import datetime
import logging
import json

# ===== Configuration =====
CONFIG = {
    'output_dir': 'data',
    'preprocessed_data_path': os.path.join('data', 'preprocessed_data.pkl'),
    'model_save_path': os.path.join('models', 'lstm_fraud_model.h5'),
    'config_path': os.path.join('models', 'model_config.json')
}

# Ensure directories exist
for directory in ['models', 'logs', 'data']:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ===== Logging Setup =====
log_filename = f"logs/train_lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="w"
)

def log(message):
    print(message)
    logging.info(message)
    for handler in logging.getLogger().handlers:
        handler.flush()

# ===== Blockchain Setup =====
blockchain_available = False
try:
    w3 = Web3(Web3.HTTPProvider('http://127.0.0.1:7545'))
    blockchain_available = w3.is_connected()
    if blockchain_available:
        contract_address = '0xb27A31f1b0AF2946B7F582768f03239b1eC07c2c'  # Contract address
        contract_abi = [
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "id",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "amount",
				"type": "uint256"
			},
			{
				"internalType": "bool",
				"name": "isFraud",
				"type": "bool"
			}
		],
		"name": "logTransaction",
		"outputs": [],
		"stateMutability": "nonpayable",
		"type": "function"
	},
	{
		"anonymous": False,
		"inputs": [
			{
				"indexed": False,
				"internalType": "string",
				"name": "id",
				"type": "string"
			},
			{
				"indexed": False,
				"internalType": "uint256",
				"name": "amount",
				"type": "uint256"
			},
			{
				"indexed": False,
				"internalType": "bool",
				"name": "isFraud",
				"type": "bool"
			},
			{
				"indexed": False,
				"internalType": "uint256",
				"name": "timestamp",
				"type": "uint256"
			}
		],
		"name": "TransactionLogged",
		"type": "event"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "id",
				"type": "string"
			}
		],
		"name": "getTransaction",
		"outputs": [
			{
				"components": [
					{
						"internalType": "string",
						"name": "id",
						"type": "string"
					},
					{
						"internalType": "uint256",
						"name": "amount",
						"type": "uint256"
					},
					{
						"internalType": "bool",
						"name": "isFraud",
						"type": "bool"
					},
					{
						"internalType": "uint256",
						"name": "timestamp",
						"type": "uint256"
					}
				],
				"internalType": "struct FraudDetection.Transaction",
				"name": "",
				"type": "tuple"
			}
		],
		"stateMutability": "view",
		"type": "function"
	},
	{
		"inputs": [],
		"name": "getTransactionStats",
		"outputs": [
			{
				"internalType": "uint256",
				"name": "totalTransactions",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "fraudCount",
				"type": "uint256"
			},
			{
				"internalType": "uint256",
				"name": "legitimateCount",
				"type": "uint256"
			}
		],
		"stateMutability": "pure",
		"type": "function"
	},
	{
		"inputs": [
			{
				"internalType": "string",
				"name": "",
				"type": "string"
			}
		],
		"name": "transactions",
		"outputs": [
			{
				"internalType": "string",
				"name": "id",
				"type": "string"
			},
			{
				"internalType": "uint256",
				"name": "amount",
				"type": "uint256"
			},
			{
				"internalType": "bool",
				"name": "isFraud",
				"type": "bool"
			},
			{
				"internalType": "uint256",
				"name": "timestamp",
				"type": "uint256"
			}
		],
		"stateMutability": "view",
		"type": "function"
	}
]
        contract = w3.eth.contract(address=contract_address, abi=contract_abi)
        
        # Additional logging for blockchain setup
        log(f"Connected to blockchain: {blockchain_available}")
        log(f"Using contract at address: {contract_address}")
        log(f"Available accounts: {w3.eth.accounts}")
        log(f"Balance of account[0]: {w3.eth.get_balance(w3.eth.accounts[0])}")
    else:
        log("Warning: Could not connect to blockchain. Continuing without blockchain integration.")
except Exception as e:
    log(f"Blockchain setup error: {str(e)}. Continuing without blockchain integration.")
    blockchain_available = False

# ===== Function to test blockchain connection =====
def test_blockchain_connection():
    """Test the blockchain connection with a simple transaction"""
    if not blockchain_available:
        log("Blockchain not available - skipping test")
        return False
    
    try:
        log("Testing blockchain connection with a simple transaction...")
        tx_hash = contract.functions.logTransaction(
            "test_transaction",
            50,
            False
        ).transact({
            'from': w3.eth.accounts[0],
            'gas': 200000,
            'gasPrice': w3.eth.gas_price
        })
        
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        log(f"Test transaction sent: {tx_hash.hex()}, status: {receipt['status']}")
        return True
    except Exception as e:
        log(f"Test transaction failed: {str(e)}")
        return False

# Test blockchain connection if available
if blockchain_available:
    blockchain_available = test_blockchain_connection()

# ===== Function to log transactions to blockchain =====
def log_to_blockchain(transaction_id, fraud_probability, is_fraud):
    """Log a transaction prediction to the blockchain"""
    if not blockchain_available:
        log("Blockchain not available - skipping transaction logging")
        return None
    
    try:
        # Convert probability to integer percentage (0-100)
        amount = int(fraud_probability * 100)
        
        # Get the nonce for the account
        nonce = w3.eth.get_transaction_count(w3.eth.accounts[0])
        
        # Estimate gas for the transaction
        gas_estimate = contract.functions.logTransaction(
            transaction_id,
            amount,
            bool(is_fraud)
        ).estimate_gas({'from': w3.eth.accounts[0]})
        
        # Log the transaction to the blockchain with proper parameters
        tx_hash = contract.functions.logTransaction(
            transaction_id,
            amount,
            bool(is_fraud)
        ).transact({
            'from': w3.eth.accounts[0],
            'gas': gas_estimate + 100000,  # Add buffer to gas estimate
            'gasPrice': w3.eth.gas_price,
            'nonce': nonce
        })
        
        # Wait for the transaction receipt
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        
        log(f"Transaction logged to blockchain: {tx_hash.hex()}, status: {receipt['status']}")
        return tx_hash.hex()
    except Exception as e:
        log(f"Error logging to blockchain: {str(e)}")
        return None

# ===== Main function =====
def train_and_evaluate_lstm():
    """Train LSTM model and evaluate performance"""
    try:
        log("Starting LSTM training script...")

        # Load preprocessed data
        if not os.path.exists(CONFIG['preprocessed_data_path']):
            log(f"Error: Data file '{CONFIG['preprocessed_data_path']}' not found.")
            sys.exit(1)

        with open(CONFIG['preprocessed_data_path'], 'rb') as f:
            data = pickle.load(f)
        log("Data loaded successfully.")

        # Extract data
        X_train_lstm = data['X_train_lstm']
        y_train_lstm = data['y_train_lstm']
        X_test_lstm = data['X_test_lstm']
        y_test_lstm = data['y_test_lstm']

        # Log data shapes to verify
        log(f"X_train_lstm shape: {X_train_lstm.shape}")
        log(f"y_train_lstm shape: {y_train_lstm.shape}")
        log(f"X_test_lstm shape: {X_test_lstm.shape}")
        log(f"y_test_lstm shape: {y_test_lstm.shape}")

        # Define LSTM model with proper input shape
        log("Creating LSTM model...")
        model = models.Sequential()
        model.add(layers.LSTM(units=64, activation='relu', 
                             input_shape=(X_train_lstm.shape[1], X_train_lstm.shape[2]), 
                             recurrent_dropout=0.2))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(16, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))  # Binary classification

        # Use a lower learning rate for smoother probability output
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
        
        log("Model compiled. Summary:")
        model.summary(print_fn=lambda x: log(x))
        
        # Add early stopping and model checkpointing
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss', 
                patience=5, 
                restore_best_weights=True
            )
        ]
        
        # Train the model
        log("Training model...")
        history = model.fit(
            X_train_lstm, y_train_lstm, 
            epochs=15,  # Increased epochs, early stopping will prevent overfitting
            batch_size=32, 
            validation_data=(X_test_lstm, y_test_lstm),
            callbacks=callbacks,
            verbose=2
        )

        # Generate predictions
        log("Generating predictions...")
        y_pred_probs = model.predict(X_test_lstm)
        
        # Calculate various thresholds to find optimal value
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_pred_probs > threshold).astype(int).flatten()
            curr_f1 = f1_score(y_test_lstm, y_pred)
            log(f"Threshold {threshold}: F1 Score = {curr_f1:.4f}")
            
            if curr_f1 > best_f1:
                best_f1 = curr_f1
                best_threshold = threshold
        
        log(f"Best threshold: {best_threshold} with F1 Score: {best_f1:.4f}")
        
        # Use best threshold for final evaluation
        y_pred = (y_pred_probs > best_threshold).astype(int).flatten()

        # Check distribution of predictions to ensure we're getting a range of values
        log(f"Prediction statistics - Min: {np.min(y_pred_probs):.4f}, Max: {np.max(y_pred_probs):.4f}, Mean: {np.mean(y_pred_probs):.4f}")
        
        # Compute metrics
        train_loss = history.history['loss'][-1]
        train_acc = history.history['accuracy'][-1]
        val_loss = history.history['val_loss'][-1]
        val_acc = history.history['val_accuracy'][-1]
        f1 = f1_score(y_test_lstm, y_pred)
        precision = precision_score(y_test_lstm, y_pred)
        roc_auc = roc_auc_score(y_test_lstm, y_pred_probs)

        # Log performance metrics
        log(f"F1 Score: {f1:.4f}")
        log(f"Precision: {precision:.4f}")
        log(f"ROC-AUC Score: {roc_auc:.4f}")
        log(f"Final Training Loss: {train_loss:.4f}")
        log(f"Final Training Accuracy: {train_acc:.4f}")
        log(f"Final Validation Loss: {val_loss:.4f}")
        log(f"Final Validation Accuracy: {val_acc:.4f}")

        # Save the model and configuration
        log(f"Saving model to {CONFIG['model_save_path']}...")
        model.save(CONFIG['model_save_path'])
        
        # Save model configuration and metrics
        model_config = {
            'input_shape': [X_train_lstm.shape[1], X_train_lstm.shape[2]],
            'metrics': {
                'f1_score': float(f1),
                'precision': float(precision),
                'roc_auc': float(roc_auc),
                'val_accuracy': float(val_acc),
                'val_loss': float(val_loss),
                'best_threshold': float(best_threshold)
            },
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        with open(CONFIG['config_path'], 'w') as f:
            json.dump(model_config, f, indent=4)
        log(f"Model configuration saved to {CONFIG['config_path']}")

        # Log sample predictions to blockchain (just a few examples)
        if blockchain_available:
            log("Logging sample predictions to blockchain...")
            sample_size = min(10, len(y_pred))  # Log at most 10 transactions
            for i in range(sample_size):
                transaction_id = f"test_transaction_{i}"
                fraud_prob = float(y_pred_probs[i][0])
                is_fraud = bool(y_pred[i])
                
                tx_hash = log_to_blockchain(transaction_id, fraud_prob, is_fraud)
                if tx_hash:
                    log(f"Logged transaction {i}: {'FRAUD' if is_fraud else 'LEGITIMATE'} (prob: {fraud_prob:.4f})")

        log("Model training and evaluation complete.")
        return model, history, {
            'f1': f1,
            'precision': precision,
            'roc_auc': roc_auc,
            'val_acc': val_acc
        }

    except Exception as e:
        log(f"Unhandled exception: {str(e)}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)

# Execute if run directly
if __name__ == "__main__":
    try:
        train_and_evaluate_lstm()
    finally:
        logging.shutdown()
from web3 import Web3
#from web3.middleware.geth_poa import geth_poa_middleware


# Connect to Ganache local blockchain
w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
#w3.middleware_onion.inject(geth_poa_middleware, layer=0)

# Set account and private key (Use only for local/dev environments!)
private_key = '0xf5c453ce5153ba2a467ed851c89e2358b52c4a67a1c4e34c5b8bb174f309eec7'
sender_address = w3.to_checksum_address('0xb4790feAf3aa01Ac1afAf03633F8a6984474ac22')

# Smart contract ABI
abi = [
    {
        "inputs": [
            {"internalType": "string", "name": "id", "type": "string"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "bool", "name": "isFraud", "type": "bool"}
        ],
        "name": "logTransaction",
        "outputs": [],
        "stateMutability": "nonpayable",
        "type": "function"
    },
    {
        "anonymous": False,
        "inputs": [
            {"indexed": False, "internalType": "string", "name": "id", "type": "string"},
            {"indexed": False, "internalType": "uint256", "name": "amount", "type": "uint256"},
            {"indexed": False, "internalType": "bool", "name": "isFraud", "type": "bool"},
            {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "name": "TransactionLogged",
        "type": "event"
    },
    {
        "inputs": [
            {"internalType": "string", "name": "", "type": "string"}
        ],
        "name": "transactions",
        "outputs": [
            {"internalType": "string", "name": "id", "type": "string"},
            {"internalType": "uint256", "name": "amount", "type": "uint256"},
            {"internalType": "bool", "name": "isFraud", "type": "bool"},
            {"internalType": "uint256", "name": "timestamp", "type": "uint256"}
        ],
        "stateMutability": "view",
        "type": "function"
    }
]

# Replace with your actual deployed contract address
contract_address = w3.to_checksum_address('0xd9145CCE52D386f254917e481eB44e9943F39138')
contract = w3.eth.contract(address=contract_address, abi=abi)

# Example transaction data
transactions = [
    ("test_1190", -79, False),   # This will be skipped
    ("test_1200", 150, True),
    ("test_1210", -80, False),   # This will be skipped
    ("test_1220", 200, False),
]

# Process and send each transaction
for tx_id, amount, is_fraud in transactions:
    if amount < 0:
        print(f"Skipping transaction {tx_id}: negative amount ({amount}) is invalid for uint256.")
        continue

    try:
        nonce = w3.eth.get_transaction_count(sender_address)

        txn = contract.functions.logTransaction(tx_id, amount, is_fraud).build_transaction({
            'from': sender_address,
            'nonce': nonce,
            'gas': 3000000,
            'gasPrice': w3.to_wei('20', 'gwei'),
        })

        signed_txn = w3.eth.account.sign_transaction(txn, private_key)
        tx_hash = w3.eth.send_raw_transaction(signed_txn.raw_transaction)

        print(f"✅ Transaction '{tx_id}' sent! TX Hash: {tx_hash.hex()}")

        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        print(f"✅ Confirmed in block {receipt.blockNumber}\n")

    except Exception as e:
        print(f"❌ Error with transaction '{tx_id}': {e}\n")

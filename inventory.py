
cryptocurrencies_tickers = {
    'Tether USD': 'USDT-USD',
    'Bitcoin USD': 'BTC-USD',
    'Ethereum USD': 'ETH-USD',
    'Binance Coin': 'BNB-USD',
    'Cardano USD': 'ADA-USD',
    'Solana USD': 'SOL-USD',
    'Polkadot USD': 'DOT-USD',
    'Avalanche USD': 'AVAX-USD',
    'Cosmos USD': 'ATOM-USD',
    'Litecoin USD': 'LTC-USD',
    'Chainlink USD': 'LINK-USD',
    'Decentraland USD': 'MANA-USD',
    'The Sandbox USD': 'SAND-USD',
    'Monero USD': 'XMR-USD',
    'Aave USD': 'AAVE-USD',
    'Terra USD': 'LUNA1-USD',
    'Uniswap USD': 'UNI1-USD',
    'Polygon USD': 'MATIC-USD'
}

tickers = list(cryptocurrencies_tickers.values())
names = list(cryptocurrencies_tickers.keys())
inv_map = {v: k for k, v in cryptocurrencies_tickers.items()}

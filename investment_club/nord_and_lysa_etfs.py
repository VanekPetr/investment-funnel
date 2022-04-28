# *** TICKERS ****
# --------------------------------------------------------------------------------------------------------------------
# Dictionaries with ETF's full name and its ticker

lysa_stock_etfs = {
    'Vanguard US': 'VUN.TO',
<<<<<<< HEAD
    'Vanguard North America': 'VNRT',   # Nan
=======
    'Vanguard North America': 'VNRT.MU',   # Munich Delayed Price. Currency in EUR
>>>>>>> main
    'Vanguard Europe': 'VGK',
    'Vanguard Emerging Markets': 'VWO',
    'Vanguard Global Small-Cap': 'VB',
    'iShares US': 'ITOT',
<<<<<<< HEAD
    'Vanguard Japan': 'VJPN',   # Nan
    'Vanguard Pacific': 'VPL',
    'Lyxor Europe': 'DR'    # Nan
}

lysa_bond_etfs = {
    'Vanguard Europe Government Bond': 'VETY',  # Nan
=======
    'Vanguard Japan': 'VJPN.MU',   # Munich Delayed Price. Currency in EUR
    'Vanguard Pacific': 'VPL',
    'Lyxor Europe': 'MEUD.PA'    # Paris Delayed Price. Currency in EUR
}

lysa_bond_etfs = {
    'Vanguard Europe Government Bond': 'VETY.MI',  # Milan Delayed Price. Currency in EUR
>>>>>>> main
    'iShares Europe Government Bond': 'LU',
    'Vanguard Global Bond': 'BNDW',
    'Vanguard Global Short-Term Bond': 'BSV',
    'Vanguard Euro Investment Grade Bond': 'VECP',
    'Vanguard Eurozone Inflation-Linked Bond': 'VTIP',
<<<<<<< HEAD
    'Vanguard Global Aggregate Bond': 'VGAB',   # Nan
    'iShares Global Inflation Linked Govt Bond': 'IGIL',    # Nan
    'Vanguard EUR Corporate Bond': 'VECP',  # Nan
    'iShares Core € Corp Bond': 'IEAC'  # Nan
=======
    'Vanguard Global Aggregate Bond': 'VGAB.TO',   # Toronto Real Time Price. Currency in CAD (only in CAD in Yahoofinance)
    'iShares Global Inflation Linked Govt Bond': 'IGIL.L',    # LSE Delayed Price. Currency in USD
    'Vanguard EUR Corporate Bond': 'VECP.DE',  # XETRA Delayed Price. Currency in EUR
    'iShares Core € Corp Bond': 'IEAC.L'  # LSE Delayed Price. Currency in EUR
>>>>>>> main
}

# TODO missing tickers
nord_etfs = {
<<<<<<< HEAD
    'DB Xtrackers II Eurozone Government Bond UCITS': '',
    'iShares Core MSCI World UCITS': '',
    'DB Xtrackers II EUR Corporate Bond UCITS': '',
    'iShares Core S&P 500 UCITS': '',
    'iShares Core Euro STOXX 50 UCITS': '',
    'iShares Core MSCI EM IMI UCITS': '',
    'iShares Core MSCI Japan IMI UCITS': ''
=======
    'DB Xtrackers II Eurozone Government Bond UCITS': 'DBXN.DE',   # XETRA Delayed Price. Currency in EUR
    'iShares Core MSCI World UCITS': 'SWDA.MI',   # Milan Delayed Price. Currency in EUR
    'DB Xtrackers II EUR Corporate Bond UCITS': 'XBLC.MI',   #  Milan Delayed Price. Currency in EUR
    'iShares Core S&P 500 UCITS': 'CSPX.L',  # LSE Delayed Price. Currency in USD
    'iShares Core Euro STOXX 50 UCITS': 'SXRT.DE',   #  XETRA Delayed Price. Currency in EUR
    'iShares Core MSCI EM IMI UCITS': 'EIMI.L',   # LSE Delayed Price. Currency in USD
    'iShares Core MSCI Japan IMI UCITS': 'IJPA.L'  # LSE Delayed Price. Currency in USD
>>>>>>> main
}

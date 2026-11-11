"""
SECTOR MAP for the ranking backtest's "max N open positions per sector" rule.

DISCLAIMER: Independent personal project. Not tied to any employer or agency.
Public market data and paper money only. No internal, personal or protected data (no SSNs/PHI).

Groups follow the GICS sectors (as of the 2023 reshuffle: V/MA/PYPL = Financials,
UBER = Industrials, GOOGL/META/NFLX = Communication). Information Technology is split
into Semiconductors vs Software & IT services vs Hardware, because chip stocks move
together much more tightly than the rest of tech.
"""

SECTOR = {}
_GROUPS = {
    'Semiconductors':   'NVDA AVGO AMD INTC QCOM TXN AMAT MU LRCX ADI KLAC MRVL',
    'Software/IT':      'MSFT CRM ORCL ADBE ACN INTU IBM NOW PLTR PANW CRWD',
    'Hardware':         'AAPL CSCO ANET DELL SMCI',
    'Communication':    'GOOGL GOOG META NFLX DIS VZ CMCSA T WBD',
    'Consumer Disc.':   'AMZN TSLA HD MCD BKNG LOW SBUX NKE F GM CCL',
    'Financials':       'BRK.B JPM V MA BAC WFC GS MS SPGI C SCHW BLK AXP COIN PYPL SOFI',
    'Health Care':      'LLY UNH JNJ ABBV MRK TMO ABT PFE AMGN ISRG MDT GILD CVS',
    'Consumer Staples': 'COST PG KO WMT PEP MO PM',
    'Energy':           'XOM CVX COP SLB OXY',
    'Industrials':      'GE CAT HON UNP RTX BA DE DAL UBER',
    'Materials':        'LIN FCX',
    'Utilities':        'NEE',
}
for _sector, _syms in _GROUPS.items():
    for _s in _syms.split():
        assert _s not in SECTOR, f'{_s} listed twice'
        SECTOR[_s] = _sector


def sector_of(symbol: str) -> str:
    """Unknown symbols get their own one-stock 'sector', so the cap never blocks them wrongly."""
    return SECTOR.get(symbol, f'other:{symbol}')

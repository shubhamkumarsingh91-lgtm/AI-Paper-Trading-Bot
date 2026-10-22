/* perf.js — "Performance by bot": one money table per bot, built from the files the bots save.
 * Independent personal project, not tied to any employer or agency. Alpaca paper (fake-money)
 * accounts and public market data only; no internal or personal data (no SSNs/PHI). Not financial advice.
 *
 * Inputs per bot: trade_log.json (BUY/SELL rows), a daily scorecard.csv (Alpaca closing balance,
 * SPY close) and learning_history.json (model accuracy/precision from testing).
 * Everything is in dollars of the bot's own starting capital ($100,000 or $25,000).
 */
(function (root) {

function perf({start, log, card, hist, spyReturn}) {
  const rows = Array.isArray(log) ? log : [];
  const buys = rows.filter(t => (t.action || '').toUpperCase() === 'BUY');
  const sells = rows.filter(t => (t.action || '').toUpperCase() === 'SELL');
  const num = v => (v == null || v === '' || isNaN(+v)) ? null : +v;
  const cost = t => (num(t.qty) ?? num(t.shares) ?? 0) * (num(t.entry_price) ?? num(t.price) ?? 0);

  // balance: last daily close from Alpaca; else start + closed P&L
  const cardRows = (card || []).filter(r => num(r.equity) != null);
  const realised = sells.reduce((a, t) => a + (num(t.pnl) ?? 0), 0);
  // no daily balance yet and no trades: unknown (don't pretend it's still the starting amount)
  const balance = cardRows.length ? num(cardRows[cardRows.length - 1].equity) : (sells.length ? start + realised : null);
  const balanceDate = cardRows.length ? cardRows[cardRows.length - 1].date : null;
  const totalPnl = balance == null ? null : balance - start;

  // max drawdown on the daily balance (peak to trough)
  let peak = start, mdd = 0;
  for (const r of cardRows) { const e = num(r.equity); peak = Math.max(peak, e); mdd = Math.min(mdd, e / peak - 1); }

  // trade quality (closed round trips)
  const pnl = t => num(t.pnl) ?? 0;
  const pctOf = t => num(t.pct) ?? (cost(t) ? pnl(t) / cost(t) : null);
  const wins = sells.filter(t => t.win === true || (t.win == null && pnl(t) > 0));
  const losses = sells.filter(t => !wins.includes(t));
  const sum = a => a.reduce((x, y) => x + y, 0);
  const avg = a => a.length ? sum(a) / a.length : null;
  const grossWin = sum(wins.map(pnl)), grossLoss = -sum(losses.map(pnl));
  const closedCost = sum(sells.map(cost));
  const best = sells.length ? sells.reduce((a, t) => pnl(t) > pnl(a) ? t : a) : null;
  const worst = sells.length ? sells.reduce((a, t) => pnl(t) < pnl(a) ? t : a) : null;
  const last = Array.isArray(hist) && hist.length ? hist[hist.length - 1] : {};
  // most money invested at the same moment: replay buys and sells in time order
  let held = {}, peakCash = 0, peakAt = null;
  [...rows].sort((a, b) => String(a.timestamp || '').localeCompare(String(b.timestamp || ''))).forEach(t => {
    const a = (t.action || '').toUpperCase();
    if (a === 'BUY') held[t.symbol] = (held[t.symbol] || 0) + cost(t);
    else if (a === 'SELL') delete held[t.symbol];
    const now = Object.values(held).reduce((x, y) => x + y, 0);
    if (now > peakCash){ peakCash = now; peakAt = t.timestamp; }
  });

  return {
    start, balance, balanceDate, totalPnl, totalPct: totalPnl == null ? null : totalPnl / start,
    vsSpy: spyReturn == null || totalPnl == null ? null : totalPnl / start - spyReturn,
    realised, other: balance == null ? null : balance - start - sells.filter(t => !balanceDate || String(t.timestamp || '').slice(0, 10) <= balanceDate).reduce((a, t) => a + (num(t.pnl) ?? 0), 0),
    maxDrawdown: cardRows.length ? mdd : null, days: cardRows.length,
    buys: buys.length, invested: sum(buys.map(cost)), peakInvested: buys.length ? peakCash : null, peakAt, avgSize: buys.length ? sum(buys.map(cost)) / buys.length : null,
    returnOnTraded: closedCost ? realised / closedCost : null,
    closed: sells.length, wins: wins.length, losses: losses.length,
    winRate: sells.length ? wins.length / sells.length : null, lossRate: sells.length ? losses.length / sells.length : null,
    avgWin: avg(wins.map(pnl)), avgLoss: avg(losses.map(pnl)),
    avgWinPct: avg(wins.map(pctOf).filter(x => x != null)), avgLossPct: avg(losses.map(pctOf).filter(x => x != null)),
    profitFactor: grossLoss > 0 ? grossWin / grossLoss : (grossWin > 0 ? Infinity : null),
    expectancy: sells.length ? realised / sells.length : null,
    best, worst,
    livePrecision: sells.length ? wins.length / sells.length : null,
    modelPrecision: last.walk_forward_precision ?? last.precision ?? null,
    modelAccuracy: last.accuracy ?? null,
    open: buys.length - sells.length,
  };
}

// ── table ──
const $f = (n, d = 0) => n == null ? '—' : (n < 0 ? '−$' : '$') + Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const $s = n => n == null ? '—' : (n >= 0 ? '+' : '−') + '$' + Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
const pc = (n, d = 1, sign = false) => n == null ? '—' : (sign && n >= 0 ? '+' : n < 0 ? '−' : '') + Math.abs(n * 100).toFixed(d) + '%';
const cls = n => n == null ? '' : n >= 0 ? 'up' : 'down';

function perfTable(bot, P, opts = {}) {
  const R = (label, value, help, c = '') => `<tr><td>${label}<i class="tip" data-tip="${String(help).replace(/"/g, '&quot;')}">i</i></td><td class="mono ${c}">${value}</td></tr>`;
  const H = t => `<tr class="grp"><td colspan="2">${t}</td></tr>`;
  const money = [
    H('Money'),
    R('Started with', $f(P.start), 'Fake money in this bot’s Alpaca paper account on day one.'),
    R('Balance now', P.balance == null ? 'from today’s close' : $f(P.balance, 2), P.balanceDate ? `Alpaca closing balance on ${P.balanceDate}. Updates once a day at the close.` : P.balance == null ? 'The bot starts recording its Alpaca balance at the end of the next trading day.' : 'Start + closed-trade profit (no daily balance recorded yet).'),
    R('Total profit / loss', P.totalPnl == null ? '—' : `${$s(P.totalPnl)} · ${pc(P.totalPct, 2, true)}`, 'Balance now minus what it started with, in dollars and as % of the starting money.', cls(P.totalPnl)),
    R('vs just holding SPY', P.vsSpy == null ? '—' : pc(P.vsSpy, 2, true) + ' pts', 'Positive = beating the market over the same days.', cls(P.vsSpy)),
    R('Biggest drop', P.maxDrawdown == null ? '—' : pc(P.maxDrawdown, 2), 'Worst fall from a high point in the daily balance. Smaller is safer.', P.maxDrawdown < 0 ? 'down' : ''),
  ];
  if (opts.holder) return tableHtml(bot, [...money,
    H('How it invests'),
    R('Holding', opts.holding || '—', 'This bot holds the market instead of making many trades, so trade stats don’t apply.'),
    R('Market mood', opts.regime || '—', 'Its daily read of the market: BULL, SIDEWAYS or BEAR.'),
    R('Days running', String(P.days), 'Trading days with a recorded closing balance.'),
  ], opts.note, P, true);
  const t = P.closed;
  return tableHtml(bot, [...money,
    R('From closed trades', $s(P.realised), 'Profit or loss locked in by trades that were bought and then sold.', cls(P.realised)),
    R('Open & other', $s(P.other), `Balance change not explained by trades closed by ${P.balanceDate || 'the last close'}: positions still open then, and fees.`, cls(P.other)),
    H('Money put to work'),
    R('Most invested at once', P.peakInvested == null ? '—' : `${$f(P.peakInvested)} (${pc(P.peakInvested / P.start, 0)})`, `The most money it had in stocks at the same moment${P.peakAt ? ' (' + String(P.peakAt).slice(0, 16).replace('T', ' ') + ')' : ''}. It can never be more than the account.`),
    R('Total traded', $f(P.invested), `All ${P.buys} buy${P.buys === 1 ? '' : 's'} added up. This can be bigger than the account: after each sale the same cash is used again for the next buy.`),
    R('Average trade size', P.avgSize == null ? '—' : `${$f(P.avgSize)} (${pc(P.avgSize / P.start, 1)})`, 'How much goes into one trade, in dollars and as % of the account.'),
    R('Return on money traded', pc(P.returnOnTraded, 2, true), 'Closed-trade profit ÷ the dollars those trades used.', cls(P.returnOnTraded)),
    H('Trade results'),
    R('Closed trades', String(t), t && t < 30 ? `Small sample: win rates mean little until about 30 trades.` : 'Round trips: bought, then sold.'),
    R('Win ratio', P.winRate == null ? '—' : `${pc(P.winRate, 0)} · ${P.wins} won`, 'Share of closed trades that made money.', P.winRate >= 0.5 ? 'up' : P.winRate == null ? '' : 'down'),
    R('Loss ratio', P.lossRate == null ? '—' : `${pc(P.lossRate, 0)} · ${P.losses} lost`, 'Share of closed trades that lost money.'),
    R('Average win', P.avgWin == null ? '—' : `${$s(P.avgWin)} · ${pc(P.avgWinPct, 2, true)}`, 'Typical winning trade.', 'up'),
    R('Average loss', P.avgLoss == null ? '—' : `${$s(P.avgLoss)} · ${pc(P.avgLossPct, 2, true)}`, 'Typical losing trade. Wins should be bigger than losses.', 'down'),
    R('Profit factor', P.profitFactor == null ? '—' : P.profitFactor === Infinity ? 'no losses yet' : P.profitFactor.toFixed(2), 'Dollars won ÷ dollars lost. Above 1.0 makes money; 1.5+ is good.', P.profitFactor == null ? '' : P.profitFactor >= 1 ? 'up' : 'down'),
    R('Average per trade', $s(P.expectancy), 'What one trade earns on average (the “expectancy”).', cls(P.expectancy)),
    R('Best trade', P.best ? `${P.best.symbol} ${$s(+P.best.pnl)}` : '—', 'Single biggest gain.', P.best ? cls(+P.best.pnl) : ''),
    R('Worst trade', P.worst ? `${P.worst.symbol} ${$s(+P.worst.pnl)}` : '—', 'Single biggest loss.', P.worst ? cls(+P.worst.pnl) : ''),
    H('How good are the buy calls'),
    R('Live precision', P.livePrecision == null ? '—' : pc(P.livePrecision, 1), 'Of the real trades it closed, how often the buy made money.'),
    R('Model precision (test)', pc(P.modelPrecision, 1), 'When the AI said “buy” on past data it hadn’t seen, how often it was right. Random ≈ 24%.'),
    R('Model accuracy (test)', pc(P.modelAccuracy, 1), 'Share of all buy/skip calls that were right in testing. Less useful than precision: skipping is easy.'),
  ], opts.note, P, false);
}
// Split the rows at each group heading so an opened table lays its groups out side by side.
function groups(rows) {
  const out = []; let cur = [];
  for (const r of rows) { if (r.startsWith('<tr class="grp"') && cur.length) { out.push(cur); cur = []; } cur.push(r); }
  if (cur.length) out.push(cur);
  return out.map(g => `<table><tbody>${g.join('')}</tbody></table>`).join('');
}
// Collapsed by default: one compact line (balance, P&L, win ratio); click to open the full table.
function tableHtml(bot, rows, note, P, holder) {
  const kv = (k, v, c = '') => `<span class="pk"><small>${k}</small><b class="mono ${c}">${v}</b></span>`;
  const line = [
    kv('Balance', P.balance == null ? '—' : $f(P.balance)),
    kv('Profit / loss', P.totalPnl == null ? '—' : `${$s(P.totalPnl)} · ${pc(P.totalPct, 2, true)}`, cls(P.totalPnl)),
    holder ? kv('vs SPY', P.vsSpy == null ? '—' : pc(P.vsSpy, 2, true), cls(P.vsSpy))
           : kv('Win ratio', P.winRate == null ? '—' : `${pc(P.winRate, 0)} of ${P.closed}`, P.winRate == null ? '' : P.winRate >= 0.5 ? 'up' : 'down'),
  ].join('');
  return `<details class="card perf fade" data-bot="${bot.id}">
    <summary><div class="ph"><span class="sw" style="background:${bot.color}"></span><b>${bot.name}</b><span class="tag">${$f(bot.start)} account</span><span class="chev">▾</span></div>
      <div class="pline">${line}</div></summary>
    ${note ? `<div class="pnote">${note}</div>` : ''}
    <div class="ptable pgroups">${groups(rows)}</div></details>`;
}

root.perf = perf; root.perfTable = perfTable;
if (typeof module !== 'undefined') module.exports = {perf, perfTable};
})(typeof window !== 'undefined' ? window : globalThis);

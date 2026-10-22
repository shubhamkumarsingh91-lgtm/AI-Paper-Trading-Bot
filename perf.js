/* perf.js — "Performance by bot": one money table per bot, built from the files the bots save.
 * Independent personal project, not tied to any employer or agency. Alpaca paper (fake-money)
 * accounts and public market data only; no internal or personal data (no SSNs/PHI). Not financial advice.
 *
 * Inputs per bot: trade_log.json (BUY/SELL rows), a daily scorecard.csv (Alpaca closing balance,
 * SPY close) and learning_history.json (model accuracy/precision from testing).
 * Everything is in dollars of the bot's own starting capital ($100,000 or $25,000).
 */
(function (root) {

// Pair the broker's filled buys and sells per stock, first in first out: one row per closed trade.
function roundTrips(orders) {
  const lots = {}, trips = [];
  [...(orders || [])].sort((a, b) => String(a.filled_at).localeCompare(String(b.filled_at))).forEach(o => {
    const q = +o.qty, px = +o.price; if (!q || !px) return;
    const L = lots[o.symbol] = lots[o.symbol] || [];
    if (o.side === 'buy'){ L.push({q, px, t:o.filled_at}); return; }
    let left = q, cost = 0, got = 0, opened = null;
    while (left > 1e-9 && L.length){ const lot = L[0], take = Math.min(left, lot.q); cost += take * lot.px; got += take; lot.q -= take; left -= take; opened = opened || lot.t; if (lot.q <= 1e-9) L.shift(); }
    if (got > 0) trips.push({symbol:o.symbol, qty:got, cost, pnl:got * px - cost, pct:(got * px - cost) / cost, openedAt:opened, closedAt:o.filled_at});
  });
  return trips;
}

function perf({start, log, card, hist, spyReturn, baseline, baselineDate, broker}) {
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
  const base = baseline ?? start;                     // what the account held when this bot's records begin
  const totalPnl = balance == null ? null : balance - base;

  // max drawdown on the daily balance (peak to trough)
  let peak = baseline ?? start, mdd = 0;   // drops are measured from when the bot's records begin
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
    start, base, baselineDate, before: baseline == null ? null : baseline - start,
    balance, balanceDate, totalPnl, totalPct: totalPnl == null ? null : totalPnl / base,
    vsSpy: spyReturn == null || totalPnl == null ? null : totalPnl / base - spyReturn,
    trips: broker ? roundTrips(broker.orders) : null, brokerUpdated: broker ? broker.updated : null,
    realised, other: balance == null ? null : balance - base - sells.filter(t => !balanceDate || String(t.timestamp || '').slice(0, 10) <= balanceDate).reduce((a, t) => a + (num(t.pnl) ?? 0), 0),
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
    ...(P.before == null ? [] : [
      R('Before this bot', $s(P.before), `Change in this Alpaca account before this bot’s records begin (${P.baselineDate}), from earlier bots that used the same account. Not this bot’s trading.`, cls(P.before)),
      R(`Balance on ${P.baselineDate}`, $f(P.base, 2), 'What the account held when this bot’s own records begin. Its results below are measured from here.')]),
    R(P.before == null ? 'Total profit / loss' : 'Profit / loss since then', P.totalPnl == null ? '—' : `${$s(P.totalPnl)} · ${pc(P.totalPct, 2, true)}`, P.before == null ? 'Balance now minus what it started with, in dollars and as % of the starting money.' : `Balance now minus the balance on ${P.baselineDate}: this bot’s own result.`, cls(P.totalPnl)),
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
    ...brokerRows(P),
    H('How good are the buy calls'),
    R('Live precision', P.livePrecision == null ? '—' : pc(P.livePrecision, 1), 'Of the real trades it closed, how often the buy made money.'),
    R('Model precision (test)', pc(P.modelPrecision, 1), 'When the AI said “buy” on past data it hadn’t seen, how often it was right. Random ≈ 24%.'),
    R('Model accuracy (test)', pc(P.modelAccuracy, 1), 'Share of all buy/skip calls that were right in testing. Less useful than precision: skipping is easy.'),
  ], opts.note, P, false);
}
// Every trade on the account according to Alpaca: totals, the earlier-bot part, and the result per stock.
function brokerRows(P) {
  const R = (label, value, help, c = '') => `<tr><td>${label}<i class="tip" data-tip="${String(help).replace(/"/g, '&quot;')}">i</i></td><td class="mono ${c}">${value}</td></tr>`;
  const out = [`<tr class="grp"><td colspan="2">Every trade on the account (Alpaca)</td></tr>`];
  if (!P.trips) return [...out, R('Broker record', 'after the next run', 'Saved from Alpaca each morning (8:30 AM ET) and at the close. To get it now: GitHub → Actions → this bot → Run workflow → mode: history.')];
  const sum = a => a.reduce((x, t) => x + t.pnl, 0), won = a => a.filter(t => t.pnl > 0).length;
  const T = P.trips;
  if (!T.length) return [...out, R('Closed trades', '0', `Alpaca has no completed buy-then-sell on this account in the last year (checked ${String(P.brokerUpdated || '').slice(0, 10)}).`)];
  out.push(R('Closed trades', `${T.length} · ${won(T)} won, ${T.length - won(T)} lost`, `Every buy that was later sold, paired from Alpaca’s own order history (checked ${String(P.brokerUpdated || '').slice(0, 10)}).`));
  out.push(R('Profit / loss from them', $s(sum(T)), 'Sum of all those closed trades, as Alpaca recorded the prices.', cls(sum(T))));
  if (P.baselineDate){
    const pre = T.filter(t => String(t.closedAt).slice(0, 10) < P.baselineDate);
    if (pre.length) out.push(R('…by earlier bots', `${pre.length} trades · ${$s(sum(pre))}`, `Closed before ${P.baselineDate}, when this bot’s records begin.`, cls(sum(pre))));
  }
  const by = {};
  T.forEach(t => { const b = by[t.symbol] = by[t.symbol] || {n:0, pnl:0, w:0, last:''}; b.n++; b.pnl += t.pnl; if (t.pnl > 0) b.w++; if (t.closedAt > b.last) b.last = t.closedAt; });
  Object.entries(by).sort((a, b) => Math.abs(b[1].pnl) - Math.abs(a[1].pnl)).slice(0, 8).forEach(([sym, b]) =>
    out.push(R(sym, `${$s(b.pnl)} · ${b.n} trade${b.n === 1 ? '' : 's'}`, `${b.w} won, ${b.n - b.w} lost; last closed ${String(b.last).slice(0, 10)}.`, cls(b.pnl))));
  return out;
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

root.perf = perf; root.perfTable = perfTable; root.roundTrips = roundTrips;
if (typeof module !== 'undefined') module.exports = {perf, perfTable, roundTrips};
})(typeof window !== 'undefined' ? window : globalThis);

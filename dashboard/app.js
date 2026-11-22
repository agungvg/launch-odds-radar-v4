/* Launch-Odds Radar: renders window.RADAR (written by main.ts) — no fetch, no network. */
(function () {
  const D = window.RADAR;
  const $ = (id) => document.getElementById(id);
  const params = new URLSearchParams(location.search);

  function money(n) {
    if (n >= 1e9) return '$' + (n / 1e9).toFixed(2) + 'B';
    if (n >= 1e6) return '$' + (n / 1e6).toFixed(1) + 'M';
    if (n >= 1e3) return '$' + (n / 1e3).toFixed(1) + 'K';
    return '$' + Math.round(n);
  }
  function moneyFull(n) { return '$' + Math.round(n).toLocaleString('en-US'); }
  function pct(n, d) { return (n * 100).toFixed(d === undefined ? 1 : d) + '%'; }
  function esc(s) { return String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c])); }

  /* ---- headline strip ---- */
  const m = D.model, h1 = D.holdout_5fold, h2 = D.holdout_loo;
  $('asof').textContent = 'Snapshot ' + D.as_of + ' · floor ' + money(D.floor_usd);
  $('src').textContent = 'Model radar-v4 · ' + D.board_size + ' ranked rows';
  $('src2').textContent = D.source;
  $('kpis').innerHTML = [
    ['Board rows', D.board_size, D.watchlist_size + ' watchlist + ' + D.early_size + ' early'],
    ['Watchlist covered', D.watchlist_size + ' / ' + D.watchlist_size, 'every watchlist slug is ranked'],
    ['Full-fit error', '×' + m.fit_medae.toFixed(2), 'vs naive ×' + m.naive_medae.toFixed(2) + ' (median abs log error)'],
    ['5-fold hold-out', '×' + h1.fitted_medae.toFixed(2), 'vs naive ×' + h1.naive_medae.toFixed(2) + ' · peer + coefs refit per fold'],
    ['Leave-one-out', '×' + h2.fitted_medae.toFixed(2), 'vs naive ×' + h2.naive_medae.toFixed(2) + ' · peer refit, coefs held'],
    ['Verdicts', 'Farm ' + D.verdicts.farm + ' · Thin ' + D.verdicts.thin, 'Skip ' + D.verdicts.skip + ' · deposit $75–$250 by TVL'],
  ].map((k) => '<div class="kpi"><b>' + k[1] + '</b><span>' + k[0] + '</span><small>' + k[2] + '</small></div>').join('');

  /* ---- model, odds, hold-outs, accounting ---- */
  $('model-text').innerHTML =
    'Terminal value = TVL<sup>' + m.beta + '</sup> × peer multiple × growth<sup>' + m.growth_a + '</sup> × maturity<sup>' + m.maturity_b + '</sup> × breadth<sup>' + m.breadth_c + '</sup> × e<sup>' + m.scale.toFixed(2) + '</sup>. ' +
    'The peer multiple is each category\u2019s median log(market cap / TVL) over its traded protocols, shrunk toward the global median ' + m.global_multiple.toFixed(4) + '× with weight n/(n+8). ' +
    'Beta, the three weights and the scale are grid-fitted on the ' + m.traded_n + ' traded protocols by smallest median absolute log error — the fit picks beta ' + m.beta + ', at the grid\u2019s lower half, because market cap grows more slowly than TVL.';
  $('odds-text').innerHTML =
    'Launch odds are a logistic regression on the same five features (log TVL, growth, maturity, breadth, peer), trained on all 758 protocols with the label \u201Crecords a token\u201D (base rate ' + pct(D.odds.base_rate) + '). ' +
    'It separates weakly — ' + (D.odds.accuracy_at_half * 100).toFixed(1) + '% right at 0.5, mean fitted odds ' + pct(D.odds.mean_prob_token) + ' for protocols that record a token vs ' + pct(D.odds.mean_prob_no_token) + ' for the rest — so odds are a prior, not a forecast. Expected value = terminal value × launch odds.';
  $('holdout-text').innerHTML =
    'Hold-outs, typical error as a multiple: full fit ×' + m.fit_medae.toFixed(2) + ' vs naive ×' + m.naive_medae.toFixed(2) + '; 5-fold ×' + h1.fitted_medae.toFixed(2) + ' vs naive ×' + h1.naive_medae.toFixed(2) + '; leave-one-out ×' + h2.fitted_medae.toFixed(2) + ' vs naive ×' + h2.naive_medae.toFixed(2) + '. Naive is the global median multiple alone. The fitted model beats it on all three.';
  $('stability-text').textContent =
    'Honest caveat: the grid coefficients move across folds — beta ' + h1.fold_coefs.map((c) => c.beta).join(', ') + '; growth weight ' + h1.fold_coefs.map((c) => c.growth_a).join(', ') + ' — so the exact weights are unstable and the board\u2019s order near the middle should be read as a band, not a precise rank.';
  const a = D.accounting;
  $('accounting').innerHTML = '<tr><th>Ranked pre-token</th><th>Excluded pre-token</th><th>Token, no market cap</th><th>Traded (market cap)</th><th>Total</th></tr>' +
    '<tr><td>' + a.ranked + '</td><td>' + a.excluded_pretoken + ' (wrapped, staked, bridged or CEX products)</td><td>' + a.token_no_mcap + '</td><td>' + a.traded + '</td><td><b>' + a.total + '</b></td></tr>';
  $('acct').textContent = 'Accounting of all ' + a.total + ' protocols: ' + a.ranked + ' ranked + ' + a.excluded_pretoken + ' excluded pre-token (wrapped, staked, bridged or CEX) + ' + a.token_no_mcap + ' with a token but no market cap + ' + a.traded + ' traded = ' + a.total + '.';

  /* ---- board ---- */
  let verdict = params.get('verdict') || 'All';
  let sortKey = 'ev';
  let sortDir = -1;
  const qInput = $('q'), watchBox = $('watchonly'), sortSel = $('sort');
  if (params.get('watchlist') === '1') watchBox.checked = true;
  if (['Farm', 'Thin', 'Skip'].includes(verdict)) {
    document.querySelectorAll('#verdict-filters button').forEach((b) => b.classList.toggle('on', b.dataset.v === verdict));
  } else verdict = 'All';
  const sorters = {
    ev: (x, y) => x.implied_value_usd - y.implied_value_usd,
    terminal: (x, y) => x.terminal_value_usd - y.terminal_value_usd,
    odds: (x, y) => x.launch_odds - y.launch_odds,
    tvl: (x, y) => x.tvl_usd - y.tvl_usd,
    growth: (x, y) => (x.change_7d || 0) - (y.change_7d || 0),
    age: (x, y) => (x.age_days || 0) - (y.age_days || 0),
    name: (x, y) => x.name.localeCompare(y.name),
    rank: (x, y) => x.rank - y.rank,
  };
  function visibleRows() {
    const q = qInput.value.trim().toLowerCase();
    let rows = D.rows.filter((r) => (verdict === 'All' || r.verdict === verdict));
    if (watchBox.checked) rows = rows.filter((r) => D.watchlist_slugs.includes(r.slug));
    if (q) rows = rows.filter((r) => (r.name + ' ' + r.slug + ' ' + (r.category || '')).toLowerCase().includes(q));
    const s = sorters[sortKey] || sorters.ev;
    rows = rows.slice().sort((x, y) => s(x, y) * sortDir);
    return rows;
  }
  function renderBoard() {
    const rows = visibleRows();
    $('board-count').textContent = '· ' + rows.length + ' of ' + D.board_size + ' rows';
    $('rows').innerHTML = rows.map((r) => {
      const g = r.change_7d === null ? '—' : '<span class="' + (r.change_7d >= 0 ? 'pos' : 'neg') + '">' + (r.change_7d >= 0 ? '+' : '') + r.change_7d.toFixed(1) + '%</span>';
      const chainShort = r.chains.slice(0, 3).join(', ') + (r.chains.length > 3 ? ' +' + (r.chains.length - 3) : '');
      return '<tr data-slug="' + esc(r.slug) + '"><td>' + r.rank + '</td>' +
        '<td class="name"><b>' + esc(r.name) + '</b><small>' + esc(r.slug) + ' · ' + esc(r.category || 'uncategorised') + ' · ' + esc(chainShort) + '</small></td>' +
        '<td class="num">' + money(r.tvl_usd) + '</td><td class="num">' + g + '</td>' +
        '<td class="num">' + money(r.terminal_value_usd) + '</td><td class="num">' + pct(r.launch_odds) + '</td>' +
        '<td class="num"><b>' + money(r.implied_value_usd) + '</b></td>' +
        '<td><span class="pill ' + r.verdict + '">' + r.verdict + '</span></td><td class="num">$' + r.deposit_usd + '</td></tr>';
    }).join('');
    document.querySelectorAll('#rows tr').forEach((tr) => tr.addEventListener('click', () => openDetail(tr.dataset.slug)));
  }
  document.querySelectorAll('#verdict-filters button').forEach((b) => b.addEventListener('click', () => {
    verdict = b.dataset.v;
    document.querySelectorAll('#verdict-filters button').forEach((x) => x.classList.toggle('on', x === b));
    renderBoard();
  }));
  document.querySelectorAll('#board th[data-k]').forEach((th) => th.addEventListener('click', () => {
    const k = th.dataset.k;
    if (sortKey === k) sortDir = -sortDir; else { sortKey = k; sortDir = k === 'name' ? 1 : -1; }
    renderBoard();
  }));
  qInput.addEventListener('input', renderBoard);
  watchBox.addEventListener('change', renderBoard);
  sortSel.addEventListener('change', () => { sortKey = sortSel.value; sortDir = sortKey === 'name' ? 1 : -1; renderBoard(); });

  /* ---- detail view ---- */
  function openDetail(slug) {
    const r = D.rows.find((x) => x.slug === slug);
    if (!r) return;
    document.querySelectorAll('#rows tr').forEach((tr) => tr.classList.toggle('sel', tr.dataset.slug === slug));
    $('detail').hidden = false;
    $('detail-title').textContent = '#' + r.rank + ' ' + r.name + ' — ' + r.verdict + ' · deposit $' + r.deposit_usd;
    $('detail-body').innerHTML = [
      [money(r.terminal_value_usd), 'Terminal value'],
      [pct(r.launch_odds), 'Launch odds'],
      [money(r.implied_value_usd), 'Expected value'],
      [money(r.tvl_usd), 'TVL · ' + (r.change_7d === null ? '7d n/a' : (r.change_7d >= 0 ? '+' : '') + r.change_7d.toFixed(1) + '% 7d')],
      [r.peer_multiple.toFixed(4) + '×', 'Peer multiple (' + (r.category || 'uncategorised') + ')'],
      [r.age_days === null ? 'unknown' : r.age_days + ' days', 'Age since listing'],
      [r.chains.join(', '), 'Chains'],
      [r.slug, 'Slug'],
    ].map((c) => '<div><b>' + esc(c[0]) + '</b><span>' + c[1] + '</span></div>').join('');
    $('detail-steps').innerHTML = r.farm_steps.map((s) => '<li>' + esc(s) + '</li>').join('');
    if (location.hash !== '#detail') location.hash = 'detail';
    $('detail').scrollIntoView({ block: 'start' });
  }

  /* ---- live table ---- */
  $('live-rows').innerHTML = D.live.map((r) => '<tr><td class="name"><b>' + esc(r.name) + '</b><small>' + esc(r.slug) + '</small></td><td>' + esc(r.category || '—') + '</td><td class="num">' + money(r.tvl_usd) + '</td><td class="num">' + (r.mcap_usd ? money(r.mcap_usd) : '—') + '</td><td class="num">' + (r.change_7d === null ? '—' : (r.change_7d >= 0 ? '+' : '') + r.change_7d.toFixed(1) + '%') + '</td></tr>').join('');

  renderBoard();
  const detailSlug = params.get('detail');
  if (detailSlug) openDetail(detailSlug);
  else if (D.rows.length) { /* keep the board as the first view */ }
})();

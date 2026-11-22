// Launch-odds radar v4: every number below is fitted from the snapshot at run time.
//
// The model, in one paragraph. A traded protocol's market cap is predicted in log space from its
// TVL and its category's peer multiple, plus three small terms the grid fits: growth (the 7-day TVL
// change), maturity (how long the protocol has run) and breadth (how many chains it spans). The
// peer multiple of a category is the median of log(market_cap / TVL) over that category's traded
// protocols, shrunk toward the global median with weight n / (n + 8). The size term is not fixed
// at one: the grid fits an exponent beta on log(TVL), because the data say market cap grows more
// slowly than TVL. A scale constant recentres the residuals (it is the median log residual of the
// fit). Launch odds are a separate fit: a logistic regression on the same five features, trained
// on all 758 protocols with the label "records a token", so the odds are a fitted prior, not a
// forecast. Expected value is terminal value times launch odds, and the board ranks by it.
//
// Nothing here names a protocol. The same functions score every row, and every constant the
// dashboard shows (beta, the three grid weights, the scale, the peer multiples, the logistic
// weights, the hold-out errors) is computed from the snapshot this run read.

/** The snapshot's protocol entry: exactly the ten fields the sealed input carries. */
export interface SnapshotProtocol {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
}

/** The whole snapshot document. */
export interface Snapshot {
  as_of: string;
  source: string;
  floor_usd: number;
  protocols: SnapshotProtocol[];
  watchlist: string[];
}

/** The fitted size-and-shape model: grid-fitted on the traded protocols. */
export interface FittedModel {
  beta: number;
  growth_a: number;
  maturity_b: number;
  breadth_c: number;
  scale: number;
  global_log: number;
  global_multiple: number;
  peer_log: Record<string, number>;
  fit_medae: number;
  fit_mae: number;
  naive_medae: number;
  naive_mae: number;
  traded_n: number;
}

/** The fitted launch-odds model: a logistic regression on the same five features. */
export interface OddsModel {
  intercept: number;
  weights: number[];
  feature_means: number[];
  feature_sds: number[];
  accuracy_at_half: number;
  mean_prob_token: number;
  mean_prob_no_token: number;
  base_rate: number;
}

/** One hold-out: the fitted model's error against the naive global-median model. */
export interface Holdout {
  fitted_medae: number;
  naive_medae: number;
  fitted_mae: number;
  naive_mae: number;
  fold_coefs: Array<{ beta: number; growth_a: number; maturity_b: number; breadth_c: number }>;
}

/** A farm step: the action, what it costs, and how long it takes. */
export interface FarmStep {
  text: string;
  cost: string;
  time: string;
}

/** One row of board.json: the ranked board's row. */
export interface BoardRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  mcap_usd: number | null;
  listed_at: string | null;
  gecko_id: string | null;
  symbol: string | null;
  token_status: 'pre-token';
  age_days: number | null;
  implied_value_usd: number;
  terminal_value_usd: number;
  launch_odds: number;
  verdict: 'Farm' | 'Thin' | 'Skip';
  deposit_usd: number;
  model: {
    peer_multiple: number;
    growth_multiplier: number;
    maturity_multiplier: number;
    breadth_multiplier: number;
    age_days: number | null;
  };
  farm_steps: string[];
}

/** The dashboard's compact view of one row. */
export interface RadarRow {
  rank: number;
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  change_7d: number | null;
  age_days: number | null;
  implied_value_usd: number;
  terminal_value_usd: number;
  launch_odds: number;
  verdict: 'Farm' | 'Thin' | 'Skip';
  deposit_usd: number;
  peer_multiple: number;
  farm_steps: string[];
}

/** A live-token protocol: it already trades, so the board shows the market's own value. */
export interface LiveRow {
  slug: string;
  name: string;
  category: string | null;
  chains: string[];
  tvl_usd: number;
  mcap_usd: number | null;
  change_7d: number | null;
}

/** Everything the dashboard needs: the board, the live set, and the fitted models. */
export interface RadarData {
  as_of: string;
  source: string;
  floor_usd: number;
  model_version: string;
  model: FittedModel;
  odds: OddsModel;
  holdout_5fold: Holdout;
  holdout_loo: Holdout;
  accounting: { ranked: number; excluded_pretoken: number; token_no_mcap: number; traded: number; total: number };
  verdicts: { farm: number; thin: number; skip: number };
  watchlist_size: number;
  watchlist_slugs: string[];
  board_size: number;
  early_size: number;
  live_size: number;
  rows: RadarRow[];
  live: LiveRow[];
}

/** Why the snapshot is not one this model can score. */
export class SnapshotError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'SnapshotError';
  }
}

const DAY_MS = 86_400_000;

/** A finite number, else `fallback`. */
function finite(value: unknown, fallback: number): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : fallback;
}

/** The median of a non-empty list of numbers. */
export function median(values: number[]): number {
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 === 1 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

/** The population standard deviation of a list of numbers. */
function pstdev(values: number[]): number {
  const m = values.reduce((a, b) => a + b, 0) / values.length;
  return Math.sqrt(values.reduce((a, b) => a + (b - m) * (b - m), 0) / values.length);
}

/** A whole day count from `from` to `to`, clamped at zero; null when either date is absent or bad. */
export function ageDays(asOf: string, listedAt: string | null): number | null {
  if (listedAt === null) return null;
  const a = Date.parse(`${asOf}T00:00:00Z`);
  const b = Date.parse(`${listedAt}T00:00:00Z`);
  if (!Number.isFinite(a) || !Number.isFinite(b)) return null;
  return Math.max(0, Math.floor((a - b) / DAY_MS));
}

/** The three shape features of one protocol, from the snapshot alone. */
export interface ShapeFeatures {
  growth: number;
  maturityBase: number;
  nChains: number;
  age: number;
}

export function shapeFeatures(p: SnapshotProtocol, asOf: string): ShapeFeatures {
  const rawGrowth = finite(p.change_7d, 0);
  const growth = Math.min(120, Math.max(-60, rawGrowth)) / 100;
  const age = ageDays(asOf, p.listed_at) ?? 365;
  const maturityBase = 0.6 + (0.4 * Math.min(age, 1095)) / 1095;
  return { growth, maturityBase, nChains: Math.max(1, p.chains.length), age };
}

/** DefiLlama's version-marker rule: a name ending in a release is a pool of another protocol. */
const VERSIONED = /\s(V\d+(\.\d+)?|v\d+(\.\d+)?|\d+\.\d+)$/;
/** A wrapped, staked or bridged product's name carries one of these whole words. */
const WRAPPED_WORD = /\b(wrapped|staked|staking)\b/i;
/** An exchange's name at the start of a protocol name. */
const EXCHANGES = ['Binance', 'Coinbase', 'Bybit', 'Kraken', 'Bitget', 'OKX', 'Bitfinex', 'Gemini'];
/** Categories that are a product of a token that already trades, not a launch of their own. */
const PRODUCT_CATEGORIES = new Set(['Bridge', 'Canonical Bridge', 'Liquid Staking', 'Liquid Restaking', 'Restaked BTC', 'CEX']);

/** Whether a protocol records no token at all. */
export function isPreToken(p: SnapshotProtocol): boolean {
  return p.mcap_usd === null && p.gecko_id === null && p.symbol === null;
}

/** Whether a protocol records a token, with or without a market cap. */
export function recordsToken(p: SnapshotProtocol): boolean {
  return p.gecko_id !== null || p.symbol !== null;
}

/**
 * Whether a protocol is one the board may rank: pre-token, and not a wrapped, staked or bridged
 * product of a token that already trades. This is the watchlist's own exclusion rule, applied to
 * every pre-token protocol alike (the TVL floor is the only part left out: the board ranks below
 * it too). No name is ever special-cased beyond these uniform patterns.
 */
export function isRankable(p: SnapshotProtocol): boolean {
  if (!isPreToken(p)) return false;
  if (p.category !== null && PRODUCT_CATEGORIES.has(p.category)) return false;
  if (VERSIONED.test(p.name)) return false;
  if (WRAPPED_WORD.test(p.name)) return false;
  if (EXCHANGES.some((e) => p.name.startsWith(e))) return false;
  return true;
}

/** The snapshot's protocol by slug. */
function bySlug(protocols: SnapshotProtocol[]): Map<string, SnapshotProtocol> {
  return new Map(protocols.map((p) => [p.slug, p]));
}

/** The traded protocols the value model fits on: a market cap and a TVL, both positive. */
export function tradedSet(snapshot: Snapshot): SnapshotProtocol[] {
  return snapshot.protocols.filter((p) => p.mcap_usd !== null && p.mcap_usd > 0 && p.tvl_usd > 0);
}

/** A peer-multiple function for one training set: category median log, shrunk to the global. */
export interface PeerFn {
  (p: SnapshotProtocol, drop?: SnapshotProtocol): number;
  globalLog: number;
}

export function makePeer(train: SnapshotProtocol[]): PeerFn {
  const logs = train.map((p) => Math.log(p.mcap_usd! / p.tvl_usd));
  const globalLog = median(logs);
  const byCat = new Map<string, number[]>();
  for (const p of train) {
    const list = byCat.get(p.category ?? '') ?? [];
    list.push(Math.log(p.mcap_usd! / p.tvl_usd));
    byCat.set(p.category ?? '', list);
  }
  const fn = ((p: SnapshotProtocol, drop?: SnapshotProtocol): number => {
    let values = byCat.get(p.category ?? '') ?? [];
    if (drop !== undefined) {
      values = train
        .filter((q) => q !== drop && (q.category ?? '') === (p.category ?? ''))
        .map((q) => Math.log(q.mcap_usd! / q.tvl_usd));
    }
    if (values.length === 0) return globalLog;
    const med = median(values);
    const n = values.length;
    return (n * med + 8 * globalLog) / (n + 8);
  }) as PeerFn;
  fn.globalLog = globalLog;
  return fn;
}

/** The log error of one prediction against a traded protocol's real market cap. */
function logError(p: SnapshotProtocol, asOf: string, peer: PeerFn, beta: number, a: number, b: number, c: number, scale: number, drop?: SnapshotProtocol): number {
  const f = shapeFeatures(p, asOf);
  const predicted =
    beta * Math.log(p.tvl_usd) +
    peer(p, drop) +
    a * Math.log1p(f.growth) +
    b * Math.log(f.maturityBase) +
    c * Math.log(f.nChains) +
    scale;
  return predicted - Math.log(p.mcap_usd!);
}

/** The median absolute log error of a set of log errors, as a multiple (exp of the median). */
export function medaeFactor(errors: number[]): number {
  return Math.exp(median(errors.map((e) => Math.abs(e))));
}

/** The mean absolute log error of a set of log errors, as a multiple. */
export function maeFactor(errors: number[]): number {
  return Math.exp(errors.reduce((s, e) => s + Math.abs(e), 0) / errors.length);
}

/** The grid the coefficients are fitted on. Beta's lower edge is 0.40. */
const BETA_GRID = [0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0];
const GROWTH_GRID = [0, 0.25, 0.5, 0.75, 1.0];
const MATURITY_GRID = [0, 0.1, 0.2, 0.3, 0.5];
const BREADTH_GRID = [0, 0.05, 0.1, 0.2];

/** The grid search: the coefficients whose median absolute log error is smallest. */
export function gridFit(train: SnapshotProtocol[], asOf: string, peer: PeerFn): { beta: number; growth_a: number; maturity_b: number; breadth_c: number; scale: number; medae: number } {
  let best: { beta: number; growth_a: number; maturity_b: number; breadth_c: number; scale: number; medae: number } | null = null;
  for (const beta of BETA_GRID) {
    for (const a of GROWTH_GRID) {
      for (const b of MATURITY_GRID) {
        for (const c of BREADTH_GRID) {
          const raw = train.map((p) => logError(p, asOf, peer, beta, a, b, c, 0));
          const scale = median(raw.map((e) => -e));
          const medae = medaeFactor(train.map((p) => logError(p, asOf, peer, beta, a, b, c, scale)));
          if (best === null || medae < best.medae) best = { beta, growth_a: a, maturity_b: b, breadth_c: c, scale, medae };
        }
      }
    }
  }
  return best!;
}

/** The fitted value model: peer multiples plus the grid-fitted coefficients and scale. */
export function fitModel(snapshot: Snapshot): FittedModel {
  const traded = tradedSet(snapshot);
  const peer = makePeer(traded);
  const fit = gridFit(traded, snapshot.as_of, peer);
  const fittedErrors = traded.map((p) => logError(p, snapshot.as_of, peer, fit.beta, fit.growth_a, fit.maturity_b, fit.breadth_c, fit.scale));
  const naiveErrors = traded.map((p) => Math.log(p.tvl_usd) + peer.globalLog - Math.log(p.mcap_usd!));
  const peerLog: Record<string, number> = {};
  for (const p of traded) peerLog[p.category ?? ''] = peer(p);
  return {
    beta: fit.beta,
    growth_a: fit.growth_a,
    maturity_b: fit.maturity_b,
    breadth_c: fit.breadth_c,
    scale: fit.scale,
    global_log: peer.globalLog,
    global_multiple: Math.exp(peer.globalLog),
    peer_log: peerLog,
    fit_medae: medaeFactor(fittedErrors),
    fit_mae: maeFactor(fittedErrors),
    naive_medae: medaeFactor(naiveErrors),
    naive_mae: maeFactor(naiveErrors),
    traded_n: traded.length,
  };
}

/** The terminal value one protocol gets under the fitted model. */
export function terminalValue(p: SnapshotProtocol, asOf: string, peer: PeerFn, model: FittedModel): number {
  const f = shapeFeatures(p, asOf);
  return Math.exp(
    model.beta * Math.log(p.tvl_usd) +
      peer(p) +
      model.growth_a * Math.log1p(f.growth) +
      model.maturity_b * Math.log(f.maturityBase) +
      model.breadth_c * Math.log(f.nChains) +
      model.scale,
  );
}

/** The five features the odds model reads, in fixed order. */
export function oddsFeatures(p: SnapshotProtocol, asOf: string, peer: PeerFn): number[] {
  const f = shapeFeatures(p, asOf);
  return [Math.log(p.tvl_usd), Math.log1p(f.growth), Math.log(f.maturityBase), Math.log(f.nChains), peer(p)];
}

/**
 * The launch-odds model: a logistic regression, fitted by plain gradient descent (3,000 steps,
 * step 0.5) on all 758 protocols, label "records a token". Features are standardised on the
 * whole snapshot. The fit is weak on purpose to say so: it separates the two groups only
 * modestly, so the odds it returns are a prior, not a forecast.
 */
export function fitOdds(snapshot: Snapshot, peer: PeerFn): OddsModel {
  const rows = snapshot.protocols.map((p) => ({ x: oddsFeatures(p, snapshot.as_of, peer), y: recordsToken(p) ? 1 : 0 }));
  const means = [0, 1, 2, 3, 4].map((i) => rows.reduce((s, r) => s + r.x[i]!, 0) / rows.length);
  const sds = [0, 1, 2, 3, 4].map((i) => pstdev(rows.map((r) => r.x[i]!)));
  const weights = [0, 0, 0, 0, 0, 0];
  for (let step = 0; step < 3000; step++) {
    const grad = [0, 0, 0, 0, 0, 0];
    for (const r of rows) {
      const z = [1, ...r.x.map((v, i) => (v - means[i]!) / sds[i]!)];
      const s = weights.reduce((acc, w, j) => acc + w * z[j]!, 0);
      const pr = 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, s))));
      for (let j = 0; j < 6; j++) grad[j]! += (pr - r.y) * z[j]!;
    }
    for (let j = 0; j < 6; j++) weights[j]! -= (0.5 * grad[j]!) / rows.length;
  }
  const prob = (x: number[]): number => {
    const z = [1, ...x.map((v, i) => (v - means[i]!) / sds[i]!)];
    const s = weights.reduce((acc, w, j) => acc + w * z[j]!, 0);
    return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, s))));
  };
  let correct = 0;
  let posSum = 0;
  let posN = 0;
  let negSum = 0;
  let negN = 0;
  for (const r of rows) {
    const pr = prob(r.x);
    if ((pr >= 0.5) === (r.y === 1)) correct++;
    if (r.y === 1) {
      posSum += pr;
      posN++;
    } else {
      negSum += pr;
      negN++;
    }
  }
  return {
    intercept: weights[0]!,
    weights: weights.slice(1),
    feature_means: means,
    feature_sds: sds,
    accuracy_at_half: correct / rows.length,
    mean_prob_token: posSum / posN,
    mean_prob_no_token: negSum / negN,
    base_rate: posN / rows.length,
  };
}

/** The launch odds one protocol gets under the fitted odds model. */
export function launchOdds(p: SnapshotProtocol, asOf: string, peer: PeerFn, odds: OddsModel): number {
  const x = oddsFeatures(p, asOf, peer);
  const z = [1, ...x.map((v, i) => (v - odds.feature_means[i]!) / odds.feature_sds[i]!)];
  const w = [odds.intercept, ...odds.weights];
  const s = w.reduce((acc, wi, j) => acc + wi * z[j]!, 0);
  return 1 / (1 + Math.exp(-Math.max(-30, Math.min(30, s))));
}

/** The naive errors of a set: global median multiple alone, no size exponent, no scale. */
function naiveErrors(rows: SnapshotProtocol[], globalLog: number): number[] {
  return rows.map((p) => Math.log(p.tvl_usd) + globalLog - Math.log(p.mcap_usd!));
}

/**
 * The two hold-outs. H1 is 5-fold over the traded set sorted by slug (fold i holds out every
 * fifth protocol): peer multiples and grid coefficients are refit on each fold's training set.
 * H2 is leave-one-out: the peer multiple is refit without the protocol, the coefficients are
 * held at the full fit. Both are reported against the naive model on the same held-out rows.
 */
export function holdouts(snapshot: Snapshot, model: FittedModel): { fiveFold: Holdout; loo: Holdout } {
  const traded = tradedSet(snapshot).sort((a, b) => (a.slug < b.slug ? -1 : a.slug > b.slug ? 1 : 0));
  const foldErrors: number[] = [];
  const foldNaive: number[] = [];
  const foldCoefs: Array<{ beta: number; growth_a: number; maturity_b: number; breadth_c: number }> = [];
  for (let f = 0; f < 5; f++) {
    const test = traded.filter((_, i) => i % 5 === f);
    const train = traded.filter((_, i) => i % 5 !== f);
    const peer = makePeer(train);
    const fit = gridFit(train, snapshot.as_of, peer);
    foldCoefs.push({ beta: fit.beta, growth_a: fit.growth_a, maturity_b: fit.maturity_b, breadth_c: fit.breadth_c });
    for (const p of test) foldErrors.push(logError(p, snapshot.as_of, peer, fit.beta, fit.growth_a, fit.maturity_b, fit.breadth_c, fit.scale));
    foldNaive.push(...naiveErrors(test, peer.globalLog));
  }
  const fiveFold: Holdout = {
    fitted_medae: medaeFactor(foldErrors),
    naive_medae: medaeFactor(foldNaive),
    fitted_mae: maeFactor(foldErrors),
    naive_mae: maeFactor(foldNaive),
    fold_coefs: foldCoefs,
  };
  const looErrors: number[] = [];
  const looNaive: number[] = [];
  for (const p of traded) {
    const peer = makePeer(traded.filter((q) => q !== p));
    looErrors.push(logError(p, snapshot.as_of, peer, model.beta, model.growth_a, model.maturity_b, model.breadth_c, model.scale, p));
    looNaive.push(Math.log(p.tvl_usd) + model.global_log - Math.log(p.mcap_usd!));
  }
  const loo: Holdout = {
    fitted_medae: medaeFactor(looErrors),
    naive_medae: medaeFactor(looNaive),
    fitted_mae: maeFactor(looErrors),
    naive_mae: maeFactor(looNaive),
    fold_coefs: [],
  };
  return { fiveFold, loo };
}

/** How much capital one farm position needs, from the protocol's own TVL. */
export function depositFor(p: SnapshotProtocol): number {
  if (p.tvl_usd >= 5_000_000_000) return 250;
  if (p.tvl_usd >= 1_000_000_000) return 150;
  return 75;
}

/** How long to hold, from the protocol's age: younger protocols get a longer window. */
function holdDaysFor(age: number | null): number {
  if (age === null) return 30;
  if (age < 180) return 60;
  if (age < 365) return 45;
  return 30;
}

/** The first chain a protocol names, or a neutral phrase. */
function chainOf(p: SnapshotProtocol): string {
  return p.chains.length > 0 ? p.chains[0]! : 'its main chain';
}

/** The category's one action, in the words a farmer can follow. */
function actionFor(p: SnapshotProtocol, deposit: number, chain: string): string {
  const c = (p.category ?? '').toLowerCase();
  const name = p.name;
  if (c.includes('lend') || c.includes('cdp') || c.includes('collateral')) return `Lend $${deposit} of USDC on ${name} (${chain}) and leave the supply earning.`;
  if (c.includes('dex') || c.includes('yield') || c.includes('leveraged farm')) return `Add $${deposit} of liquidity to ${name}'s deepest pool on ${chain} and leave it in range.`;
  if (c.includes('bridge')) return `Bridge $${deposit} of USDC through ${name} to ${chain}, then put the bridged funds to work in its app.`;
  if (c.includes('staking') || c.includes('restak')) return `Stake $${deposit} through ${name} on ${chain} and keep the staking receipt in the wallet.`;
  if (c.includes('perpetual') || c.includes('derivative')) return `Post $${deposit} of USDC as margin on ${name} and open one small hedged trade.`;
  if (c.includes('prediction')) return `Back one market on ${name} with $${deposit} and let the position settle.`;
  if (c.includes('curator') || c.includes('allocator')) return `Put $${deposit} into a ${name} vault on ${chain} and hold the vault shares.`;
  if (c.includes('stable')) return `Mint or deposit $${deposit} of the stable asset via ${name} on ${chain}.`;
  if (c.includes('rwa')) return `Mint $${deposit} of ${name}'s yield-bearing asset on ${chain} and hold it.`;
  if (c.includes('insurance')) return `Buy $${deposit} of cover from ${name} on ${chain} and keep the policy live.`;
  if (c.includes('liquid')) return `Deposit $${deposit} of the liquid asset into ${name} on ${chain} and hold the receipt token.`;
  return `Deposit $${deposit} of USDC into ${name} on ${chain} and keep the position open.`;
}

/**
 * A protocol's farm plan: six steps, each an action with a cost and a time. The plan is derived
 * from the protocol's own record (category, chains, TVL, age), never from a hand-written list.
 */
export function farmSteps(p: SnapshotProtocol, age: number | null): FarmStep[] {
  const chain = chainOf(p);
  const deposit = depositFor(p);
  const hold = holdDaysFor(age);
  return [
    {
      text: `Open a fresh wallet for this farm alone, fund it with exactly $${deposit} plus gas, and store its seed phrase offline.`,
      cost: 'under $3 of gas for the first transaction',
      time: '5 minutes',
    },
    {
      text: actionFor(p, deposit, chain),
      cost: `$${deposit} of capital and about $${deposit >= 150 ? 8 : 4} of gas`,
      time: '15 minutes to set up',
    },
    {
      text: `Do one real action a week on ${p.name} — a deposit, a swap, a borrow or a vote — so the wallet reads as active, not parked.`,
      cost: 'about $2 of gas a week',
      time: '10 minutes a week',
    },
    {
      text: `Keep the position open for at least ${hold} days, so it is counted in more than one snapshot window.`,
      cost: `the $${deposit} stays in the position`,
      time: `${hold} days`,
    },
    {
      text: `When a claim opens, open ${p.name}'s own rewards page from its official site and claim only from the link it publishes.`,
      cost: 'no fee to check, and never pay to claim',
      time: '10 minutes',
    },
    {
      text: `Log the wallet, the dates and every transaction hash, so you can repeat the farm or unwind it cleanly.`,
      cost: 'nothing but the log',
      time: '5 minutes',
    },
  ];
}

/** A step as one string: its action, then its cost and time, within the gate's 20..600 characters. */
export function stepText(step: FarmStep): string {
  let s = `${step.text} Cost: ${step.cost}. Time: ${step.time}.`;
  if (s.length > 600) s = `${s.slice(0, 596)}...`;
  if (s.length < 20) s = `${s} Hold the position and check the protocol's own page.`;
  return s;
}

/** The verdict a ranked row gets: Farm is the only call to put money in. */
export function verdictFor(ev: number, odds: number): 'Farm' | 'Thin' | 'Skip' {
  if (ev >= 25_000_000 && odds >= 0.07) return 'Farm';
  if (ev >= 5_000_000) return 'Thin';
  return 'Skip';
}

/** The protocols the board ranks, strongest first, with ranks 1..n. */
export function buildBoard(snapshot: Snapshot, model: FittedModel, oddsModel: OddsModel, peer: PeerFn): BoardRow[] {
  const index = bySlug(snapshot.protocols);
  const chosen: SnapshotProtocol[] = [];
  for (const slug of snapshot.watchlist) {
    const p = index.get(slug);
    if (p !== undefined && isRankable(p)) chosen.push(p);
  }
  const watch = new Set(chosen.map((p) => p.slug));
  for (const p of snapshot.protocols) {
    if (!watch.has(p.slug) && isRankable(p)) chosen.push(p);
  }
  const scored = chosen.map((p) => {
    const terminal = terminalValue(p, snapshot.as_of, peer, model);
    const odds = launchOdds(p, snapshot.as_of, peer, oddsModel);
    const ev = terminal * odds;
    const f = shapeFeatures(p, snapshot.as_of);
    return { p, terminal, odds, ev, f };
  });
  scored.sort((a, b) => b.ev - a.ev || (a.p.slug < b.p.slug ? -1 : a.p.slug > b.p.slug ? 1 : 0));
  return scored.map((s, i) => {
    const age = ageDays(snapshot.as_of, s.p.listed_at);
    return {
      rank: i + 1,
      slug: s.p.slug,
      name: s.p.name,
      category: s.p.category,
      chains: s.p.chains,
      tvl_usd: s.p.tvl_usd,
      change_7d: s.p.change_7d,
      mcap_usd: s.p.mcap_usd,
      listed_at: s.p.listed_at,
      gecko_id: s.p.gecko_id,
      symbol: s.p.symbol,
      token_status: 'pre-token',
      age_days: age,
      implied_value_usd: Math.max(1, Math.round(s.ev)),
      terminal_value_usd: Math.max(1, Math.round(s.terminal)),
      launch_odds: Math.round(s.odds * 10000) / 10000,
      verdict: verdictFor(s.ev, s.odds),
      deposit_usd: depositFor(s.p),
      model: {
        peer_multiple: Math.round(Math.exp(peer(s.p)) * 10000) / 10000,
        growth_multiplier: Math.round(Math.pow(1 + s.f.growth, model.growth_a) * 10000) / 10000,
        maturity_multiplier: Math.round(Math.pow(s.f.maturityBase, model.maturity_b) * 10000) / 10000,
        breadth_multiplier: Math.round(Math.pow(s.f.nChains, model.breadth_c) * 10000) / 10000,
        age_days: age,
      },
      farm_steps: farmSteps(s.p, age).map(stepText),
    };
  });
}

/** The live-token protocols, by market cap, for the dashboard's Live token view. */
export function liveRows(snapshot: Snapshot): LiveRow[] {
  return snapshot.protocols
    .filter((p) => !isPreToken(p) && p.mcap_usd !== null && p.mcap_usd > 0 && p.category !== 'CEX')
    .sort((a, b) => (b.mcap_usd ?? 0) - (a.mcap_usd ?? 0) || b.tvl_usd - a.tvl_usd)
    .slice(0, 24)
    .map((p) => ({
      slug: p.slug,
      name: p.name,
      category: p.category,
      chains: p.chains,
      tvl_usd: p.tvl_usd,
      mcap_usd: p.mcap_usd,
      change_7d: p.change_7d,
    }));
}

/** A board row's compact form for data.js. */
export function radarRow(row: BoardRow): RadarRow {
  return {
    rank: row.rank,
    slug: row.slug,
    name: row.name,
    category: row.category,
    chains: row.chains,
    tvl_usd: row.tvl_usd,
    change_7d: row.change_7d,
    age_days: row.age_days,
    implied_value_usd: row.implied_value_usd,
    terminal_value_usd: row.terminal_value_usd,
    launch_odds: row.launch_odds,
    verdict: row.verdict,
    deposit_usd: row.deposit_usd,
    peer_multiple: row.model.peer_multiple,
    farm_steps: row.farm_steps,
  };
}

/** Everything the dashboard renders, from the snapshot. */
export function buildRadar(snapshot: Snapshot): { board: BoardRow[]; radar: RadarData } {
  const model = fitModel(snapshot);
  const peer = makePeer(tradedSet(snapshot));
  const oddsModel = fitOdds(snapshot, peer);
  const board = buildBoard(snapshot, model, oddsModel, peer);
  const live = liveRows(snapshot);
  const early = board.filter((r) => !snapshot.watchlist.includes(r.slug)).length;
  const pre = snapshot.protocols.filter(isPreToken);
  const accounting = {
    ranked: board.length,
    excluded_pretoken: pre.length - board.length,
    token_no_mcap: snapshot.protocols.filter((p) => p.mcap_usd === null && !isPreToken(p)).length,
    traded: snapshot.protocols.filter((p) => p.mcap_usd !== null).length,
    total: snapshot.protocols.length,
  };
  const verdicts = {
    farm: board.filter((r) => r.verdict === 'Farm').length,
    thin: board.filter((r) => r.verdict === 'Thin').length,
    skip: board.filter((r) => r.verdict === 'Skip').length,
  };
  const { fiveFold, loo } = holdouts(snapshot, model);
  const radar: RadarData = {
    as_of: snapshot.as_of,
    source: snapshot.source,
    floor_usd: snapshot.floor_usd,
    model_version: 'radar-v4',
    model,
    odds: oddsModel,
    holdout_5fold: fiveFold,
    holdout_loo: loo,
    accounting,
    verdicts,
    watchlist_size: snapshot.watchlist.length,
    watchlist_slugs: [...snapshot.watchlist].sort(),
    board_size: board.length,
    early_size: early,
    live_size: live.length,
    rows: board.map(radarRow),
    live,
  };
  return { board, radar };
}

/** The snapshot out of a file's text, refused unless it is the shape the model scores. */
export function parseSnapshot(text: string): Snapshot {
  let raw: unknown;
  try {
    raw = JSON.parse(text);
  } catch {
    throw new SnapshotError('snapshot.json is not JSON');
  }
  if (typeof raw !== 'object' || raw === null || Array.isArray(raw)) throw new SnapshotError('snapshot.json is not a JSON object');
  const o = raw as Record<string, unknown>;
  if (typeof o.as_of !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(o.as_of)) throw new SnapshotError('snapshot.json has no as_of date');
  if (typeof o.source !== 'string') throw new SnapshotError('snapshot.json has no source');
  if (typeof o.floor_usd !== 'number' || !Number.isSafeInteger(o.floor_usd) || o.floor_usd <= 0) throw new SnapshotError('snapshot.json has no floor_usd');
  if (!Array.isArray(o.protocols)) throw new SnapshotError('snapshot.json has no protocols array');
  if (!Array.isArray(o.watchlist) || o.watchlist.some((s) => typeof s !== 'string')) throw new SnapshotError('snapshot.json has no watchlist of slugs');
  const protocols = o.protocols.map((v, i) => {
    if (typeof v !== 'object' || v === null || Array.isArray(v)) throw new SnapshotError(`snapshot.json's protocol ${i} is not an object`);
    const p = v as Record<string, unknown>;
    if (typeof p.slug !== 'string' || typeof p.name !== 'string' || typeof p.tvl_usd !== 'number' || !Number.isFinite(p.tvl_usd)) {
      throw new SnapshotError(`snapshot.json's protocol ${i} lacks a slug, a name or a TVL`);
    }
    return {
      slug: p.slug,
      name: p.name,
      category: typeof p.category === 'string' ? p.category : null,
      chains: Array.isArray(p.chains) ? p.chains.filter((c): c is string => typeof c === 'string') : [],
      tvl_usd: p.tvl_usd,
      change_7d: typeof p.change_7d === 'number' && Number.isFinite(p.change_7d) ? p.change_7d : null,
      mcap_usd: typeof p.mcap_usd === 'number' && Number.isFinite(p.mcap_usd) ? p.mcap_usd : null,
      listed_at: typeof p.listed_at === 'string' ? p.listed_at : null,
      gecko_id: typeof p.gecko_id === 'string' ? p.gecko_id : null,
      symbol: typeof p.symbol === 'string' ? p.symbol : null,
    };
  });
  return {
    as_of: o.as_of,
    source: o.source,
    floor_usd: o.floor_usd,
    protocols,
    watchlist: o.watchlist as string[],
  };
}

/** Why a board is not one the gate will take, or null: coverage, ranks, order, one shared implied value, and farm steps. */
export function boardIssue(board: BoardRow[], snapshot: Snapshot): string | null {
  const slugs = new Set(board.map((r) => r.slug));
  const missing = snapshot.watchlist.filter((s) => !slugs.has(s));
  if (missing.length > 0) return `the board misses ${missing.length} watchlist slug${missing.length === 1 ? '' : 's'}: ${missing.slice(0, 5).join(', ')}`;
  const ranks = board.map((r) => r.rank).sort((a, b) => a - b);
  for (let i = 0; i < ranks.length; i++) if (ranks[i] !== i + 1) return `rank ${i + 1} is ${ranks[i]}`;
  for (let i = 1; i < board.length; i++) if (board[i]!.implied_value_usd > board[i - 1]!.implied_value_usd) return `row ${i + 1} outranks row ${i} on implied value`;
  if (board.length >= 2 && board[0]!.implied_value_usd === board[board.length - 1]!.implied_value_usd) {
    return `all ${board.length} rows share the same implied value $${board[0]!.implied_value_usd}: a board that ranks nothing by value is not a ranking`;
  }
  for (const r of board) {
    const steps = r.farm_steps;
    if (steps.length < 3 || steps.length > 12) return `${r.slug} has ${steps.length} farm steps`;
    for (const s of steps) if (s.length < 20 || s.length > 600) return `${r.slug} has a farm step of ${s.length} characters`;
    if (!Number.isSafeInteger(r.implied_value_usd) || r.implied_value_usd <= 0) return `${r.slug} has implied value ${r.implied_value_usd}`;
  }
  return null;
}

/** A compact data.js: the dashboard's whole dataset as one script, no fetch and no network. */
export function dataScript(radar: RadarData): string {
  return `window.RADAR=${JSON.stringify(radar)};\n`;
}

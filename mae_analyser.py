"""
mae_analyser.py  —  Maximum Adverse / Favorable Excursion analysis

What it does
────────────
After the bot accumulates enough closed trades it reads the trade log CSV
and answers one question:

  "Given how far price moved against us before winners resolved, where
   should our stop loss actually sit?"

Key outputs
───────────
1.  MAE distribution for winners vs losers — the core insight
    • Winners rarely go deep into the red before recovering
    • Losers tend to keep going — they breach a threshold and never come back
    → The "sweet spot" SL is just beyond where most winners bounce

2.  Optimal SL threshold — the MAE% that:
    • Captures ≥ KEEP_PCT (default 85%) of winning trades
    • Still cuts ≥ CUT_PCT  (default 70%) of losing trades

3.  Suggested sl_atr_mult adjustment
    The current SL is expressed in ATR multiples (e.g. 1.5×ATR).
    We back-calculate the ATR-equivalent from the MAE data and suggest
    a new multiplier for config.yaml.

4.  Wick breach rate — what % of trades had price go below (long) or
    above (short) the entry candle's wick before the trade resolved.
    High wick-breach rate = entry candle low/high is not a reliable SL anchor.

5.  Per-pair breakdown — some pairs may be far more "noisy" than others.

Usage
─────
Standalone:
    python mae_analyser.py

As a module (called from bot.py):
    from mae_analyser import MAEAnalyser
    report = MAEAnalyser("/data/trades.csv").analyse()
    # report is a dict; report["suggested_sl_atr_mult"] can update risk config

Integration with bot
────────────────────
The bot calls MAEAnalyser().analyse() inside startup() once there are
≥ MIN_TRADES closed trades, and logs the suggestions.  The RiskManager's
sl_atr_mult is updated automatically if the confidence is high enough
(≥ MIN_TRADES_FOR_AUTO_APPLY trades with a consistent signal).
"""

import os
import csv
import logging
import statistics
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Tuning knobs ──────────────────────────────────────────────────────────────
MIN_TRADES          = 20    # minimum trades before analysis is meaningful
KEEP_PCT            = 0.85  # we want to keep at least this fraction of winners
CUT_PCT             = 0.70  # we want to cut at least this fraction of losers
MIN_TRADES_FOR_AUTO = 50    # trades needed before auto-updating sl_atr_mult
AUTO_APPLY_MARGIN   = 0.05  # only auto-apply if suggested mult differs by > 5%


# ─────────────────────────────────────────────────────────────────────────────
class MAEAnalyser:

    def __init__(self, trades_file: str = "/data/trades.csv"):
        self.trades_file = trades_file
        self.trades: List[dict] = []

    # ── I/O ───────────────────────────────────────────────────────────────────

    def load(self) -> bool:
        """Load trades CSV. Returns False if file missing or too few trades."""
        if not os.path.exists(self.trades_file):
            return False
        try:
            with open(self.trades_file, newline="") as f:
                reader = csv.DictReader(f)
                self.trades = [
                    r for r in reader
                    if float(r.get("mae_pct", 0) or 0) > 0   # only trades with MAE data
                ]
        except Exception as exc:
            logger.warning("MAE analyser: could not load trades: %s", exc)
            return False
        return len(self.trades) >= MIN_TRADES

    # ── Core analysis ─────────────────────────────────────────────────────────

    def analyse(self,
                current_sl_atr_mult: float = 1.5,
                current_atr_pct: float = 0.0) -> Dict:
        """
        Run the full MAE analysis. Returns a results dict.

        Parameters
        ----------
        current_sl_atr_mult : float
            The bot's current stop-loss ATR multiplier (from config.yaml)
        current_atr_pct : float
            Average ATR as a % of price across recent trades.
            If 0, we estimate it from the data.
        """
        if not self.load():
            logger.info("MAE analyser: not enough trades yet (need %d with MAE data).",
                        MIN_TRADES)
            return {}

        winners = [t for t in self.trades if float(t.get("pnl_usdt", 0)) > 0]
        losers  = [t for t in self.trades if float(t.get("pnl_usdt", 0)) <= 0]

        if not winners or not losers:
            logger.info("MAE analyser: need both wins and losses to optimise SL.")
            return {}

        win_maes  = sorted(float(t["mae_pct"]) for t in winners)
        loss_maes = sorted(float(t["mae_pct"]) for t in losers)
        win_mfes  = [float(t.get("mfe_pct", 0)) for t in winners]

        # ── 1. Optimal SL threshold ───────────────────────────────────────────
        optimal_sl_pct, keep_actual, cut_actual = self._find_optimal_threshold(
            win_maes, loss_maes
        )

        # ── 2. Suggest new sl_atr_mult ────────────────────────────────────────
        # We know: SL_price = entry ± (ATR × sl_mult)
        # So:      ATR%     = optimal_sl_pct / sl_mult    (if we know ATR%)
        # Or:      sl_mult  = optimal_sl_pct / ATR%       (if we know ATR%)
        #
        # If we don't have average ATR% passed in, estimate it:
        #   current config gives: current_sl% ≈ ATR% × current_sl_atr_mult
        #   average observed MAE on losers ≈ actual SL distance
        avg_loss_mae = statistics.median(loss_maes) if loss_maes else 0
        if current_atr_pct > 0:
            suggested_mult = round(optimal_sl_pct / current_atr_pct, 2)
        elif avg_loss_mae > 0:
            # Infer ATR% from: loser MAE ≈ how far they moved before being cut
            # This is an approximation — assume losers hit SL at about 1× MAE median
            implied_atr_pct = avg_loss_mae / max(current_sl_atr_mult, 0.1)
            suggested_mult  = round(optimal_sl_pct / implied_atr_pct, 2) if implied_atr_pct > 0 else current_sl_atr_mult
        else:
            suggested_mult = current_sl_atr_mult

        # Clamp to sensible range
        suggested_mult = max(0.5, min(5.0, suggested_mult))

        # ── 3. Wick breach stats ──────────────────────────────────────────────
        total_wick = sum(int(t.get("wick_breach", 0) or 0) for t in self.trades)
        wick_breach_rate = total_wick / len(self.trades) if self.trades else 0.0

        win_wick  = sum(int(t.get("wick_breach", 0) or 0) for t in winners)
        loss_wick = sum(int(t.get("wick_breach", 0) or 0) for t in losers)

        # ── 4. MFE stats (how far winners ran) ───────────────────────────────
        avg_mfe = statistics.mean(win_mfes) if win_mfes else 0.0
        p80_mfe = _percentile(win_mfes, 80) if win_mfes else 0.0

        # ── 5. Per-pair breakdown ─────────────────────────────────────────────
        pairs = {t["pair"] for t in self.trades}
        per_pair = {}
        for pair in sorted(pairs):
            pt = [t for t in self.trades if t["pair"] == pair]
            pw = [t for t in pt if float(t.get("pnl_usdt", 0)) > 0]
            pl = [t for t in pt if float(t.get("pnl_usdt", 0)) <= 0]
            if pw and pl:
                pm, _, _ = self._find_optimal_threshold(
                    sorted(float(t["mae_pct"]) for t in pw),
                    sorted(float(t["mae_pct"]) for t in pl),
                )
            else:
                pm = None
            per_pair[pair] = {
                "trades":      len(pt),
                "wins":        len(pw),
                "avg_mae_win": round(statistics.mean(float(t["mae_pct"]) for t in pw), 3) if pw else 0,
                "avg_mae_loss": round(statistics.mean(float(t["mae_pct"]) for t in pl), 3) if pl else 0,
                "opt_sl_pct":  round(pm, 3) if pm else None,
            }

        result = {
            "total_trades":         len(self.trades),
            "winners":              len(winners),
            "losers":               len(losers),
            "win_mae_p50":          round(_percentile(win_maes,  50), 3),
            "win_mae_p80":          round(_percentile(win_maes,  80), 3),
            "win_mae_p95":          round(_percentile(win_maes,  95), 3),
            "loss_mae_p50":         round(_percentile(loss_maes, 50), 3),
            "loss_mae_p80":         round(_percentile(loss_maes, 80), 3),
            "optimal_sl_pct":       round(optimal_sl_pct, 3),  # as % of entry
            "keep_winners_pct":     round(keep_actual * 100, 1),
            "cut_losers_pct":       round(cut_actual * 100, 1),
            "suggested_sl_atr_mult": suggested_mult,
            "current_sl_atr_mult":  current_sl_atr_mult,
            "wick_breach_rate":     round(wick_breach_rate * 100, 1),
            "win_wick_breach_rate": round(win_wick  / max(len(winners), 1) * 100, 1),
            "loss_wick_breach_rate": round(loss_wick / max(len(losers),  1) * 100, 1),
            "avg_mfe_winner":       round(avg_mfe, 3),
            "p80_mfe_winner":       round(p80_mfe, 3),
            "per_pair":             per_pair,
        }

        self._log_report(result)
        return result

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _find_optimal_threshold(
        self,
        win_maes:  List[float],
        loss_maes: List[float],
    ) -> Tuple[float, float, float]:
        """
        Binary-search for the MAE threshold T such that:
          • P(winner MAE < T) ≥ KEEP_PCT   (keep 85% of winners)
          • P(loser  MAE < T) ≥ CUT_PCT    (cut 70% of losers)

        Returns (optimal_threshold_pct, keep_fraction, cut_fraction).
        """
        candidates = sorted(set(win_maes + loss_maes))
        best_t, best_score = candidates[-1], -1.0
        best_keep, best_cut = 1.0, 1.0

        for t in candidates:
            keep = sum(1 for m in win_maes  if m <= t) / max(len(win_maes), 1)
            cut  = sum(1 for m in loss_maes if m <= t) / max(len(loss_maes), 1)

            # We want to minimise t while satisfying both constraints
            if keep >= KEEP_PCT and cut >= CUT_PCT:
                # Lower t = tighter SL = better (less drawdown)
                score = -t  # maximise by minimising t
                if score > best_score:
                    best_t, best_score = t, score
                    best_keep, best_cut = keep, cut

        return best_t, best_keep, best_cut

    def _log_report(self, r: dict):
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("  📐 MAE / MFE STOP-LOSS ANALYSIS")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("  Trades analysed : %d  (%d W / %d L)",
                    r["total_trades"], r["winners"], r["losers"])
        logger.info("")
        logger.info("  ── Winner MAE (how far they went against before winning) ──")
        logger.info("  Median: %.3f%%   P80: %.3f%%   P95: %.3f%%",
                    r["win_mae_p50"], r["win_mae_p80"], r["win_mae_p95"])
        logger.info("  ── Loser MAE ────────────────────────────────────────────")
        logger.info("  Median: %.3f%%   P80: %.3f%%",
                    r["loss_mae_p50"], r["loss_mae_p80"])
        logger.info("")
        logger.info("  ── Optimal stop-loss ────────────────────────────────────")
        logger.info("  SL at %.3f%% from entry  →  keeps %.1f%% of winners,  cuts %.1f%% of losers",
                    r["optimal_sl_pct"], r["keep_winners_pct"], r["cut_losers_pct"])
        logger.info("  Current  sl_atr_mult = %.2f", r["current_sl_atr_mult"])
        logger.info("  Suggested sl_atr_mult = %.2f", r["suggested_sl_atr_mult"])

        diff = abs(r["suggested_sl_atr_mult"] - r["current_sl_atr_mult"])
        if diff < 0.1:
            logger.info("  ✅ Current SL is already well-calibrated (Δ < 0.1)")
        elif r["suggested_sl_atr_mult"] < r["current_sl_atr_mult"]:
            logger.info("  ⚡ Suggestion: TIGHTEN stop — you're giving back too much on losers")
        else:
            logger.info("  ⚡ Suggestion: WIDEN stop — winners are getting clipped prematurely")

        logger.info("")
        logger.info("  ── MFE (how far winners ran in your favour) ─────────────")
        logger.info("  Avg: %.3f%%   P80: %.3f%%  ← use these to calibrate TP",
                    r["avg_mfe_winner"], r["p80_mfe_winner"])
        logger.info("")
        logger.info("  ── Entry candle wick breach ──────────────────────────────")
        logger.info("  Overall: %.1f%%   Winners: %.1f%%   Losers: %.1f%%",
                    r["wick_breach_rate"],
                    r["win_wick_breach_rate"],
                    r["loss_wick_breach_rate"])
        if r["win_wick_breach_rate"] < 20:
            logger.info("  ✅ Entry candle wick is a reliable SL anchor for winners")
        elif r["win_wick_breach_rate"] > 50:
            logger.info("  ⚠️  Entry wick is frequently breached — SL below wick may be too tight")
        logger.info("")
        logger.info("  ── Per-pair ──────────────────────────────────────────────")
        for pair, ps in r["per_pair"].items():
            opt_str = f"{ps['opt_sl_pct']:.3f}%" if ps["opt_sl_pct"] is not None else "N/A"
            logger.info("  %-22s  trades=%-3d  MAE_W=%.3f%%  MAE_L=%.3f%%  opt_SL=%s",
                        pair, ps["trades"],
                        ps["avg_mae_win"], ps["avg_mae_loss"], opt_str)
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ── Utility ───────────────────────────────────────────────────────────────────

def _percentile(data: list, pct: float) -> float:
    """Simple percentile (linear interpolation) — no numpy dependency."""
    if not data:
        return 0.0
    s = sorted(data)
    n = len(s)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    frac = idx - lo
    return s[lo] + frac * (s[hi] - s[lo])


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")
    trades_path = sys.argv[1] if len(sys.argv) > 1 else "/data/trades.csv"
    current_mult = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    MAEAnalyser(trades_path).analyse(current_sl_atr_mult=current_mult)

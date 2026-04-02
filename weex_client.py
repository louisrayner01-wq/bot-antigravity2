"""
weex_client.py
Handles all communication with the Weex REST API.
Supports both Spot (SPBL) and Futures/Perpetuals (UMCBL).
Includes HMAC-SHA256 authentication, rate-limit awareness,
and a clean interface for market data + order management.

Futures order sides:
  open_long   — buy to open a long position
  open_short  — sell to open a short position
  close_long  — sell to close a long position
  close_short — buy to close a short position
"""

import hashlib
import hmac
import base64
import time
import json
import logging
import requests
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# ── Weex Spot API endpoints ────────────────────────────────────────────────────
ENDPOINTS = {
    "ticker":        "/api/v3/market/ticker",       # GET ?symbol=BTCUSDT
    "ticker_v2":     "/api/v2/spot/market/tickers", # fallback v2 endpoint
    "candles":       "/api/v3/market/klines",       # confirmed correct endpoint
    "depth":         "/api/v3/market/depth",
    "balance":       "/api/v2/account/assets",
    "place_order":   "/api/v2/trade/orders",
    "cancel_order":  "/api/v2/trade/cancel-order",
    "get_order":     "/api/v2/trade/order",
    "open_orders":   "/api/v2/trade/unfilled-orders",
}

# ── Weex Contract (Futures/Perpetuals) API ────────────────────────────────────
# Contract API lives on a separate domain and uses /capi/v3/ paths.
# Symbol format: plain BTCUSDT (no _UMCBL suffix), same as spot.
CONTRACT_BASE_URL = "https://api-contract.weex.com"

FUTURES_ENDPOINTS = {
    "place_order":   "/capi/v3/order",
    "cancel_order":  "/capi/v3/order/cancel",
    "set_leverage":  "/capi/v3/account/setLeverage",
    "account":       "/capi/v3/account",
    "position":      "/capi/v3/position/getPositions",
    "open_orders":   "/capi/v3/order/current",
    # Contract OHLCV candles — plain symbol, interval param, same format as spot
    "candles":       "/capi/v3/market/klines",
    # Contract mark/last price ticker — returns markPrice, lastPr, indexPrice
    "ticker":        "/capi/v3/market/ticker",
}

# Map our minute-based timeframe config values → Weex spot interval strings
INTERVAL_MAP = {
    "1":    "1m",
    "3":    "3m",
    "5":    "5m",
    "15":   "15m",
    "30":   "30m",
    "60":   "1h",
    "120":  "2h",
    "240":  "4h",
    "360":  "6h",
    "720":  "12h",
    "1440": "1d",
}

# Map our minute-based timeframe config values → Weex contract interval strings
# Contract API (/capi/v3/) uses the same interval format as spot (/api/v3/).
FUTURES_INTERVAL_MAP = {
    "1":    "1m",
    "3":    "3m",
    "5":    "5m",
    "15":   "15m",
    "30":   "30m",
    "60":   "1h",
    "120":  "2h",
    "240":  "4h",
    "360":  "6h",
    "720":  "12h",
    "1440": "1d",
}

# Weex futures candles API returns at most 200 candles per request
FUTURES_CANDLE_LIMIT = 200

def _market_symbol(symbol: str) -> str:
    """
    Weex market data endpoints use plain symbols (e.g. BTCUSDT),
    while order endpoints use the suffixed form (e.g. BTCUSDT_SPBL).
    Strip the suffix for market data calls.
    """
    return symbol.replace("_SPBL", "").replace("_UMCBL", "").replace("_DMCBL", "")


class WeexClient:
    """Authenticated Weex Spot API client."""

    def __init__(self, api_key: str, api_secret: str, passphrase: str,
                 base_url: str = "https://api-spot.weex.com"):
        self.api_key    = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.base_url   = base_url.rstrip("/")
        self.session    = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    # ── Signature ──────────────────────────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = "") -> str:
        """HMAC-SHA256 → Base64 per Weex auth spec."""
        message = timestamp + method.upper() + path + body
        mac = hmac.new(
            bytes(self.api_secret, "utf-8"),
            bytes(message, "utf-8"),
            digestmod=hashlib.sha256,
        )
        return base64.b64encode(mac.digest()).decode()

    def _auth_headers(self, method: str, path: str, body: str = "") -> Dict[str, str]:
        ts = str(int(time.time() * 1000))
        return {
            "ACCESS-KEY":        self.api_key,
            "ACCESS-SIGN":       self._sign(ts, method, path, body),
            "ACCESS-PASSPHRASE": self.passphrase,
            "ACCESS-TIMESTAMP":  ts,
        }

    # ── HTTP helpers ───────────────────────────────────────────────────────────

    # Fallback domains tried in order if the primary base_url fails DNS resolution.
    _FALLBACK_DOMAINS = [
        "https://api-spot.weex.com",
        "https://api.weex.com",
    ]

    def _get(self, path: str, params: Optional[Dict] = None, auth: bool = True) -> Dict:
        qs = ""
        if params:
            qs = "?" + "&".join(f"{k}={v}" for k, v in params.items())

        # /capi/ paths belong to the contract API domain — route there directly.
        # All other paths (spot /api/v2/, /api/v3/) use the configured base URL.
        if path.startswith("/capi/"):
            candidates = [CONTRACT_BASE_URL]
        else:
            # Build list of base URLs to try: configured domain first, then fallbacks
            candidates = [self.base_url] + [
                d for d in self._FALLBACK_DOMAINS if d != self.base_url
            ]

        last_exc: Optional[Exception] = None
        for base in candidates:
            headers = self._auth_headers("GET", path + qs) if auth else {}
            try:
                resp = self.session.get(base + path + qs, headers=headers, timeout=10)
                resp.raise_for_status()
                data = resp.json()
                # If a fallback domain worked, switch to it permanently
                if base != self.base_url:
                    logger.info("⚡ Switched API base URL: %s → %s", self.base_url, base)
                    self.base_url = base
                # Some Weex endpoints (e.g. klines) return a raw list, not {"data": [...]}
                if isinstance(data, list):
                    return {"data": data, "code": "0"}
                if data.get("code") not in (None, "0", 0, "00000"):
                    logger.warning("API warning [GET %s]: %s", path, data)
                return data
            except Exception as exc:
                last_exc = exc
                is_dns = "NameResolution" in str(type(exc).__name__) or "Name or service not known" in str(exc)
                if is_dns and base != candidates[-1]:
                    logger.debug("DNS failure on %s — trying next domain…", base)
                    continue   # try the next domain
                # Non-DNS error or last candidate — log and give up
                level = logging.DEBUG if "404" in str(exc) else logging.ERROR
                logger.log(level, "GET %s failed: %s", path, exc)
                return {}

        return {}

    def _post(self, path: str, payload: Dict, auth: bool = True) -> Dict:
        body = json.dumps(payload, separators=(",", ":"))
        headers = self._auth_headers("POST", path, body) if auth else {}
        base = CONTRACT_BASE_URL if path.startswith("/capi/") else self.base_url
        try:
            resp = self.session.post(base + path, headers=headers, data=body, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") not in (None, "0", 0, "00000"):
                logger.warning("API warning [POST %s]: %s", path, data)
            return data
        except Exception as exc:
            logger.error("POST %s failed: %s", path, exc)
            return {}

    # ── Public market data (no auth needed) ───────────────────────────────────

    def get_ticker(self, symbol: str) -> Optional[Dict]:
        """Return current price data for a symbol.

        For futures (_UMCBL / _DMCBL) symbols, queries the contract ticker
        endpoint first (returns markPrice, lastPr, indexPrice).  Falls back
        to the spot ticker endpoints if the contract call fails.

        Returns None quietly if all endpoints fail so callers can fall back
        to the last candle close without noisy log spam.
        """
        sym = _market_symbol(symbol)
        is_futures = "_UMCBL" in symbol or "_DMCBL" in symbol

        # For futures, try the contract domain ticker first.
        # The _get() router sends /capi/ paths to CONTRACT_BASE_URL automatically.
        if is_futures:
            data   = self._get(FUTURES_ENDPOINTS["ticker"], {"symbol": sym}, auth=False)
            result = data.get("data")
            if result:
                if isinstance(result, list):
                    for item in result:
                        if item.get("symbol") == sym:
                            return item
                else:
                    return result

        # Spot fallback (also used for non-futures symbols)
        for endpoint in (ENDPOINTS["ticker"], ENDPOINTS["ticker_v2"]):
            data = self._get(endpoint, {"symbol": sym}, auth=False)
            result = data.get("data")
            if not result:
                continue
            # Some endpoints return a list — find our symbol
            if isinstance(result, list):
                for item in result:
                    if item.get("symbol") == sym:
                        return item
                continue
            return result
        return None

    def get_candles(self, symbol: str, granularity: str = "60", limit: int = 300,
                    start_time: Optional[int] = None,
                    end_time: Optional[int] = None) -> List[List]:
        """
        Fetch OHLCV candles from Weex.
        granularity: minutes as string from config ("60" = 1 hour)
        start_time / end_time: optional Unix timestamps in milliseconds for
            paginating through historical data.
        Returns list of [timestamp_ms, open, high, low, close, volume] in
            chronological order.

        Routes automatically:
          _UMCBL symbols → contract endpoint (api-contract.weex.com/capi/v3/market/klines),
                           plain symbol (no suffix), interval param, max 200 candles per call.
          plain symbols  → spot endpoint (api-spot.weex.com/api/v3/market/klines),
                           interval param, up to 1000 candles per call.
        """
        is_futures = "_UMCBL" in symbol or "_DMCBL" in symbol

        if is_futures:
            # Contract API uses plain symbol (no _UMCBL suffix) and interval param —
            # same format as spot.  Routes to api-contract.weex.com via _get().
            interval  = FUTURES_INTERVAL_MAP.get(str(granularity), "1h")
            endpoint  = FUTURES_ENDPOINTS["candles"]
            limit     = min(limit, FUTURES_CANDLE_LIMIT)
            params: Dict[str, Any] = {
                "symbol":   _market_symbol(symbol),   # strip _UMCBL
                "interval": interval,
                "limit":    limit,
            }
        else:
            interval  = INTERVAL_MAP.get(str(granularity), "1h")
            endpoint  = ENDPOINTS["candles"]
            params = {
                "symbol":   _market_symbol(symbol),
                "interval": interval,
                "limit":    limit,
            }

        # Weex klines endpoints accept timestamps in SECONDS, not milliseconds.
        # Callers pass ms (standard internal unit) so we divide here.
        if start_time is not None:
            params["startTime"] = start_time // 1000
        if end_time is not None:
            params["endTime"] = end_time // 1000

        data = self._get(endpoint, params, auth=False)
        raw = data.get("data", [])
        if not raw:
            logger.debug("Empty candle response for %s: %s", symbol, data)
            return []

        # Normalise: if candles are dicts, convert to [ts, o, h, l, c, v] lists
        if isinstance(raw[0], dict):
            raw = [
                [
                    c.get("t") or c.get("time") or c.get("openTime"),
                    c.get("o") or c.get("open"),
                    c.get("h") or c.get("high"),
                    c.get("l") or c.get("low"),
                    c.get("c") or c.get("close"),
                    c.get("v") or c.get("volume"),
                ]
                for c in raw
            ]

        # Weex returns newest-first — reverse to chronological order
        return list(reversed(raw))

    # ── Private account / trading endpoints ───────────────────────────────────

    def get_balance(self) -> Dict[str, float]:
        """Return {coin: available_balance} dict."""
        data = self._get(ENDPOINTS["balance"])
        assets = data.get("data", {}).get("list", [])
        return {a["coinName"]: float(a.get("available", 0)) for a in assets}

    def place_order(self, symbol: str, side: str, quantity: float,
                    price: Optional[float] = None,
                    order_type: str = "market",
                    client_oid: Optional[str] = None) -> Dict:
        """
        Place a spot order.
        symbol: full form e.g. BTCUSDT_SPBL
        side: 'buy' | 'sell'
        order_type: 'market' | 'limit'
        """
        payload: Dict[str, Any] = {
            "symbol":    symbol,
            "side":      side,
            "orderType": order_type,
            "quantity":  str(round(quantity, 6)),
        }
        if order_type == "limit" and price is not None:
            payload["price"] = str(round(price, 2))
        if client_oid:
            payload["clientOrderId"] = client_oid

        logger.info("Placing %s %s order: %s qty=%s price=%s",
                    order_type, side, symbol, quantity, price)
        return self._post(ENDPOINTS["place_order"], payload)

    def cancel_order(self, symbol: str, order_id: str) -> Dict:
        payload = {"symbol": symbol, "orderId": order_id}
        return self._post(ENDPOINTS["cancel_order"], payload)

    def get_order(self, symbol: str, order_id: str) -> Optional[Dict]:
        data = self._get(ENDPOINTS["get_order"],
                         {"symbol": symbol, "orderId": order_id})
        return data.get("data")

    def get_open_orders(self, symbol: str) -> List[Dict]:
        data = self._get(ENDPOINTS["open_orders"], {"symbol": symbol})
        return data.get("data", {}).get("orderList", [])

    # ── Futures / perpetuals (UMCBL) ──────────────────────────────────────────

    def set_leverage(self, symbol: str, leverage: int,
                     hold_side: str = "long") -> Dict:
        """
        Set leverage for a futures symbol.
        hold_side: 'long' | 'short'  (call once for each side before trading)
        """
        payload = {
            "symbol":     symbol,
            "marginCoin": "USDT",
            "leverage":   str(leverage),
            "holdSide":   hold_side,
        }
        logger.info("Setting leverage %dx for %s (%s)", leverage, symbol, hold_side)
        return self._post(FUTURES_ENDPOINTS["set_leverage"], payload)

    def futures_order(self, symbol: str, side: str, qty: float) -> Dict:
        """
        Place a futures market order.
        side: 'open_long' | 'open_short' | 'close_long' | 'close_short'
        qty: contract size (base currency units, e.g. BTC)
        """
        payload = {
            "symbol":     symbol,
            "marginCoin": "USDT",
            "size":       str(round(qty, 6)),
            "side":       side,
            "orderType":  "market",
        }
        logger.info("Futures %s  %s  qty=%s", side.upper(), symbol, qty)
        return self._post(FUTURES_ENDPOINTS["place_order"], payload)

    def get_futures_balance(self, ref_symbol: str = "BTCUSDT_UMCBL") -> float:
        """Return available USDT balance in the futures (UMCBL) account."""
        data = self._get(FUTURES_ENDPOINTS["account"],
                         {"symbol": ref_symbol, "marginCoin": "USDT"})
        account = data.get("data", {})
        return float(account.get("available", 0) or 0)

    def get_futures_position(self, symbol: str) -> Optional[Dict]:
        """Return the open position dict for a symbol, or None if flat."""
        data = self._get(FUTURES_ENDPOINTS["position"],
                         {"symbol": symbol, "marginCoin": "USDT"})
        positions = data.get("data", [])
        if isinstance(positions, list):
            for p in positions:
                if float(p.get("total", 0) or 0) > 0:
                    return p
        return None

    # ── Utility ───────────────────────────────────────────────────────────────

    def ping(self) -> bool:
        """Quick connectivity check using public ticker endpoint."""
        result = self.get_ticker("BTCUSDT")
        return result is not None

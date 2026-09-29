import os
import sys
import time
import threading
import unicodedata

from pdf2zh.translator import BingTranslator

_lock = threading.Lock()
_cache = {}
_orig_find_sid = BingTranslator.find_sid
TTL = 120.0


def cached_find_sid(self):
    now = time.time()
    with _lock:
        c = _cache.get("sid")
    if c and now - c[5] < TTL:
        return c[0], c[1], c[2], c[3], c[4]
    url, ig, iid, key, token = _orig_find_sid(self)
    with _lock:
        _cache["sid"] = (url, ig, iid, key, token, time.time())
    return url, ig, iid, key, token


def do_translate(self, text):
    text = text[:1000]
    last = None
    for _ in range(4):
        try:
            url, ig, iid, key, token = cached_find_sid(self)
            resp = self.session.post(
                f"{url}ttranslatev3?IG={ig}&IID={iid}",
                data={
                    "fromLang": self.lang_in,
                    "to": self.lang_out,
                    "text": text,
                    "token": token,
                    "key": key,
                },
                headers=self.headers,
                timeout=30,
            )
            resp.raise_for_status()
            out = resp.json()[0]["translations"][0]["text"]
            # Bing 偶发输出 CJK 兼容表意文字（如 U+F967「不」、U+F9A1「李」），
            # 统一归一化为标准汉字，避免字体缺字形。
            return unicodedata.normalize("NFKC", out)
        except Exception as e:  # noqa: BLE001
            last = e
            with _lock:
                _cache.pop("sid", None)
            time.sleep(1.5)
    raise RuntimeError(f"bing translate failed: {last}")


BingTranslator.find_sid = cached_find_sid
BingTranslator.do_translate = do_translate

from pdf2zh.pdf2zh import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

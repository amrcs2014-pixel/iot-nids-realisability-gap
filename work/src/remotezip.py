"""Read members of a large remote zip through HTTP range requests (no full download)."""
import io, os, sys, time, urllib.request, zipfile

UA = {"User-Agent": "Mozilla/5.0 (research downloader)"}


class HttpFile(io.RawIOBase):
    def __init__(self, url, block=4 << 20):
        self.url, self.pos, self.block = url, 0, block
        req = urllib.request.Request(url, headers=dict(UA, Range="bytes=0-0"))
        with urllib.request.urlopen(req, timeout=120) as r:
            self.size = int(r.headers["Content-Range"].split("/")[1])
        self.cache = {}

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def _get(self, a, b):
        for attempt in range(6):
            try:
                req = urllib.request.Request(self.url, headers=dict(UA, Range=f"bytes={a}-{b}"))
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.read()
            except Exception as e:
                time.sleep(5 + 5 * attempt); err = e
        raise err

    def read(self, n=-1):
        if n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        out = bytearray(); end = self.pos + n
        while self.pos < end:
            bi = self.pos // self.block
            if bi not in self.cache:
                if len(self.cache) > 8:
                    self.cache.pop(next(iter(self.cache)))
                a = bi * self.block
                self.cache[bi] = self._get(a, min(a + self.block, self.size) - 1)
            blk = self.cache[bi]; o = self.pos - bi * self.block
            take = blk[o:o + (end - self.pos)]
            out += take; self.pos += len(take)
        return bytes(out)

    def readinto(self, b):
        d = self.read(len(b)); b[:len(d)] = d
        return len(d)


def listing(url):
    z = zipfile.ZipFile(io.BufferedReader(HttpFile(url), buffer_size=1 << 20))
    return z


def extract(url, members, dest):
    z = listing(url)
    os.makedirs(dest, exist_ok=True)
    for m in members:
        out = os.path.join(dest, os.path.basename(m))
        if os.path.exists(out):
            continue
        t0 = time.time()
        with z.open(m) as src, open(out + ".part", "wb") as dst:
            while True:
                b = src.read(1 << 22)
                if not b:
                    break
                dst.write(b)
        os.replace(out + ".part", out)
        print(f"extracted {m} {os.path.getsize(out)/1e6:.1f} MB in {time.time()-t0:.0f} s", flush=True)


if __name__ == "__main__":
    import collections
    for name, url in (("cowrie160 data_all.zip", "https://zenodo.org/api/records/21260400/files/data_all.zip/content"),
                      ("Gotham 2025", "https://zenodo.org/api/records/14502760/files/GothamDataset2025.zip/content")):
        z = listing(url)
        infos = z.infolist()
        print(f"\n=== {name}: {len(infos)} members, {sum(i.compress_size for i in infos)/1e9:.2f} GB compressed, {sum(i.file_size for i in infos)/1e9:.2f} GB raw")
        ext = collections.Counter(os.path.splitext(i.filename)[1] for i in infos)
        print("extensions:", ext.most_common(8))
        top = collections.Counter("/".join(i.filename.split("/")[:2]) for i in infos)
        print("top-level:", top.most_common(12))
        for i in infos[:45]:
            print(f"   {i.filename}  comp {i.compress_size/1e6:.1f} MB  raw {i.file_size/1e6:.1f} MB")
        if len(infos) > 45:
            big = sorted(infos, key=lambda i: -i.file_size)[:12]
            print("  largest:")
            for i in big:
                print(f"   {i.filename}  comp {i.compress_size/1e6:.1f} MB  raw {i.file_size/1e6:.1f} MB")
            small = [i for i in infos if i.filename.lower().endswith((".md", ".txt", ".json", ".yml", ".yaml")) and i.file_size < 200000]
            for i in small[:6]:
                print("  ---", i.filename); print(z.read(i.filename).decode("utf-8", "replace")[:1800])

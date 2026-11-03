#!/usr/bin/env python3
"""Token-by-token comparison of two llama-perplexity KL runs against the same reference (LLAMA_KLD_DUMP files).
   bench/kld_pair.py <name A> <A dump>[,<A dump>...] <name B> <B dump>[,<B dump>...]
Prints both means, the paired difference of the per-token KL with its standard error, and the top-1 agreement with the
discordant counts (tokens where only A or only B agrees with the reference; McNemar z = (a_only - b_only)/sqrt(sum))."""
import math, struct, sys
from array import array

def load(paths):
    kld, top = array('f'), array('B')
    for p in paths.split(','):
        with open(p, 'rb') as f:
            n = struct.unpack('<I', f.read(4))[0]
            k = array('f'); k.frombytes(f.read(4 * n))
            f.read(4 * n)  # delta-p, unused
            t = array('B'); t.frombytes(f.read(n))
        kld.extend(k); top.extend(t)
    return kld, top

na, pa, nb, pb = sys.argv[1:5]
ka, ta = load(pa); kb, tb = load(pb)
n = len(ka)
if n != len(kb):
    sys.exit(f"token counts differ: {n} vs {len(kb)}")
d = [a - b for a, b in zip(ka, kb)]
md = sum(d) / n
sd = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1))
a_only = sum(1 for a, b in zip(ta, tb) if a and not b)
b_only = sum(1 for a, b in zip(ta, tb) if b and not a)
z = (a_only - b_only) / math.sqrt(a_only + b_only) if a_only + b_only else 0.0
same = sum(1 for a, b in zip(ka, kb) if a == b)
print(f"{na} vs {nb}: {n} tokens | KL {sum(ka)/n:.6f} vs {sum(kb)/n:.6f}, paired diff {md:+.6f} ± {sd/math.sqrt(n):.6f} (z {md/(sd/math.sqrt(n)) if sd else 0:+.2f})"
      f" | top-1 {100*sum(ta)/n:.3f} vs {100*sum(tb)/n:.3f} %, only-{na} {a_only}, only-{nb} {b_only}, McNemar z {z:+.2f}"
      f" | identical per-token KL on {100*same/n:.1f} % of tokens")

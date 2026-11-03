#!/usr/bin/env python3
"""Prompt-cache behaviour of a running llama-server: slot reuse on a follow-up, eviction by a different long prompt,
and restore of the first conversation from the --cache-ram host cache. Stdlib only.
Usage: cache_probe.py http://127.0.0.1:8080 [out.json] [prefix_tokens=6000]"""
import glob, json, sys, time, urllib.request
BASE = sys.argv[1]; OUT = sys.argv[2] if len(sys.argv) > 2 else None; N = int(sys.argv[3]) if len(sys.argv) > 3 else 6000
def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)
def ntok(text): return len(json.load(post("/tokenize", {"content": text}))["tokens"])
src = sorted(glob.glob("src/*.cpp")); corpus = "".join(open(f, errors="ignore").read() for f in src)
cpt = len(corpus[:60000]) / max(1, ntok(corpus[:60000]))
A = "Document A.\n" + corpus[: int(N * cpt)]
B = "Document B.\n" + corpus[len(corpus) // 2 : len(corpus) // 2 + int(N * cpt)]
def ask(doc, q, tag):
    t0 = time.time()
    d = json.load(post("/v1/chat/completions", {"messages": [{"role": "system", "content": "Answer in one short sentence."},
          {"role": "user", "content": doc + "\n\nQuestion: " + q}], "max_tokens": 16, "temperature": 0.0}))
    t = d.get("timings", {})
    return {"tag": tag, "wall_s": round(time.time() - t0, 2), "prompt_n": t.get("prompt_n"), "cache_n": t.get("cache_n"),
            "prompt_ms": t.get("prompt_ms"), "predicted_n": t.get("predicted_n"), "text": (d["choices"][0]["message"].get("content") or "")[:80]}
steps = []
steps.append(ask(A, "Which file does the text start with?", "A1 cold"))
steps.append(ask(A, "Name one function mentioned.", "A2 same prefix, new question (slot reuse)"))
steps.append(ask(B, "Which file does the text start with?", "B1 different prefix (evicts the slot; A goes to the host cache)"))
steps.append(ask(A, "How many functions are mentioned, roughly?", "A3 back to A (restore from --cache-ram)"))
steps.append(ask(B, "Name one function mentioned.", "B2 back to B (restore from --cache-ram)"))
rep = {"prefix_tokens_target": N, "steps": steps,
       "verdict": {"slot_reuse": (steps[1]["cache_n"] or 0) > 0.9 * (steps[0]["prompt_n"] or 1),
                   "restore_A": (steps[3]["cache_n"] or 0) > 0.9 * (steps[0]["prompt_n"] or 1),
                   "restore_B": (steps[4]["cache_n"] or 0) > 0.9 * (steps[2]["prompt_n"] or 1)}}
js = json.dumps(rep, indent=1); print(js)
if OUT: open(OUT, "w").write(js)

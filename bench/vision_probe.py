#!/usr/bin/env python3
"""Drop-in API conformance probe against a running llama-server (OpenAI routes, tools, reasoning, streaming),
vision inside a multi-turn conversation, and prompt-cache reuse across requests. Stdlib only.
Usage: vision_probe.py http://127.0.0.1:8080 [out.json]
Prints one JSON object: per-request server timings, the image's prompt-token cost, cache reuse and simple checks."""
import base64, json, struct, sys, time, urllib.request, zlib

BASE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else None

def png_rgb(w, h, pix):
    raw = b"".join(b"\x00" + bytes(pix(y)) for y in range(h))
    def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")

def row(y):
    out = []
    for x in range(512):
        if 60 <= x < 220 and 60 <= y < 220: out += [220, 30, 30]          # red square, top left
        elif (x - 370) ** 2 + (y - 360) ** 2 < 110 ** 2: out += [30, 60, 220]  # blue disc, bottom right
        else: out += [255, 255, 255]
    return out

IMG = base64.b64encode(png_rgb(512, 512, row)).decode()
SYSTEM = "You are a terse assistant for a smart-home dashboard. Answer in at most three sentences. Use tools when they apply."
TOOLS = [{"type": "function", "function": {"name": "set_light", "description": "Set a light's brightness",
          "parameters": {"type": "object", "properties": {"room": {"type": "string"}, "level": {"type": "integer"}}, "required": ["room", "level"]}}}]

def post(path, body, timeout=1800):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=timeout)

def chat(messages, stream=True, max_tokens=96, tools=None, extra=None):
    body = {"messages": messages, "stream": stream, "max_tokens": max_tokens, "temperature": 0.0,
            "stream_options": {"include_usage": True}}
    if tools: body["tools"] = tools
    if extra: body.update(extra)
    t0 = time.time(); text = []; reasoning = []; tool_calls = []; timings = {}; usage = {}; first = None
    with post("/v1/chat/completions", body) as resp:
        if stream:
            for raw in resp:
                line = raw.decode("utf8", "ignore").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"): continue
                d = json.loads(line[5:])
                if first is None: first = time.time()
                timings = d.get("timings") or timings; usage = d.get("usage") or usage
                for ch in d.get("choices", []):
                    delta = ch.get("delta", {})
                    if delta.get("content"): text.append(delta["content"])
                    if delta.get("reasoning_content"): reasoning.append(delta["reasoning_content"])
                    for tc in delta.get("tool_calls", []) or []: tool_calls.append(tc)
        else:
            d = json.load(resp); first = time.time()
            timings = d.get("timings", {}); usage = d.get("usage", {})
            m = d["choices"][0]["message"]; text.append(m.get("content") or ""); reasoning.append(m.get("reasoning_content") or "")
            tool_calls = m.get("tool_calls") or []
    return {"wall_s": round(time.time() - t0, 2), "ttft_s": round((first or time.time()) - t0, 2), "text": "".join(text),
            "reasoning_chars": len("".join(reasoning)), "tool_calls": tool_calls,
            "timings": {k: timings.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second", "cache_n", "draft_n", "draft_n_accepted")},
            "usage": usage}

report = {"base": BASE, "image_png_bytes": len(IMG) * 3 // 4}
hist = [{"role": "system", "content": SYSTEM}]

# 1. tools + streaming on a plain text turn
hist.append({"role": "user", "content": "It is too bright in the kitchen, take it down to 30 percent."})
r1 = chat(hist, tools=TOOLS); report["r1_tools_stream"] = r1
hist.append({"role": "assistant", "content": r1["text"] or "Done."})
# 2. the image, mid-conversation, system prompt and tools still present
hist.append({"role": "user", "content": [{"type": "text", "text": "Now look at this picture. What shapes and colors do you see, and where are they?"},
                                          {"type": "image_url", "image_url": {"url": "data:image/png;base64," + IMG}}]})
r2 = chat(hist, tools=TOOLS, max_tokens=120); report["r2_image"] = r2
low = r2["text"].lower(); report["r2_mentions"] = {"red": "red" in low, "blue": "blue" in low, "square": "square" in low or "rectangle" in low, "circle": "circle" in low or "disc" in low or "round" in low}
hist.append({"role": "assistant", "content": r2["text"]})
# 3. follow-up: the slot should reuse the cached prefix (prompt_n small, cache_n large)
hist.append({"role": "user", "content": "Which of the two is closer to the top left corner?"})
r3 = chat(hist, tools=TOOLS); report["r3_followup_cache"] = r3
# 4. reasoning field through the deepseek format, non-streaming
r4 = chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": "Think step by step: 17 * 23 = ?"}], stream=False, max_tokens=256)
report["r4_reasoning_nonstream"] = {k: r4[k] for k in ("reasoning_chars", "text", "timings")}
# image cost: prompt tokens of r2 minus its text tokens (history was cached), per the server's own timings
p2 = r2["timings"]; report["image_cost"] = {"prompt_n": p2.get("prompt_n"), "cache_n": p2.get("cache_n"), "prompt_ms": p2.get("prompt_ms"),
    "note": "prompt_n of the image turn = image tokens (1280 at these flags) + the new text; prompt_ms is the time to process them"}
for k in ("r1_tools_stream", "r2_image", "r3_followup_cache"):
    report[k]["text"] = report[k]["text"][:400]
js = json.dumps(report, indent=1)
print(js)
if OUT: open(OUT, "w").write(js)

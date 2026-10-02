#!/usr/bin/env python3
"""Phase 5: summarise phase5_footprint_raw.jsonl into three tables (median over repeats).

  phase5_memory.tsv  method, bank: fixed cost, resident, build peak, search peaks
  phase5_speed.tsv   method, bank, setting, crop: latency, max crops/s, utilisation,
                     CPU cores, recall@1, image-score error
  phase5_sweep.tsv   method, bank, fraction of max rate: rate, duty, utilisation

Usage: python3 phase5_summarize.py results/phase5_footprint_raw.jsonl results/
"""
import collections
import json
import os
import statistics
import sys

RAW, OUT = sys.argv[1], sys.argv[2]
recs = [json.loads(l) for l in open(RAW)]
ok = [r for r in recs if r["rc"] == 0]
bad = [r for r in recs if r["rc"] != 0]


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def f(x, p=1):
    return "" if x is None else ("%.*f" % (p, x) if isinstance(x, float) else str(x))


by = collections.defaultdict(list)
for r in ok:
    by[(r["method"], r["bank"])].append(r)

# a method that failed every repeat for a bank still gets a row, with the reason
failed = {}
for r in bad:
    if (r["method"], r["bank"]) not in by:
        failed[(r["method"], r["bank"])] = (r.get("error") or "").strip().splitlines()[-1:] or ["?"]

rows_m, rows_s, rows_w = [], [], []
for (m, b), rs in sorted(by.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    def wpeak(shape):
        return med([max([w["peak_mb"] for w in r.get("windows", []) if w["tag"].endswith(":" + shape)
                         and w["kind"] in ("lat", "max") and w["peak_mb"] is not None] or [None]) for r in rs])
    rows_m.append([m, b, med([r.get("ctx_mb") for r in rs]), med([r.get("lib_mb") for r in rs]),
                   med([r.get("resident_mb") for r in rs]), med([r.get("build_peak_mb") for r in rs]),
                   wpeak("256"), wpeak("512"), med([r.get("process_peak_mb") for r in rs]),
                   med([r.get("build_ms") for r in rs]), med([r.get("rss_after_mb") for r in rs]),
                   rs[0].get("device_mem_mb"), len(rs)])
    tags = sorted({w["tag"] for r in rs for w in r.get("windows", [])})
    rec = {str(x["param"]): x for x in rs[0]["recall"]}
    for t in tags:
        p, shape = t.rsplit(":", 1)
        lat = [w for r in rs for w in r.get("windows", []) if w["tag"] == t and w["kind"] == "lat"]
        mx = [w for r in rs for w in r.get("windows", []) if w["tag"] == t and w["kind"] == "max"]
        rc = rec.get(p, {})
        rows_s.append([m, b, p, shape, med([w["ms"] for w in lat]), med([w["rate"] for w in mx]),
                       med([w["util"] for w in mx]), med([w["cores"] for w in mx]),
                       rc.get("recall1"), rc.get("score_err_pct"), len(lat)])
        ms = med([w["ms"] for w in lat])
        for w in [w for r in rs for w in r.get("windows", []) if w["tag"] == t and w["kind"] == "sweep"]:
            rows_w.append([m, b, p, w["frac"], w["rate"], w["rate"] * ms / 10.0, w["util"], w["cores"]])
for (m, b), why in sorted(failed.items(), key=lambda kv: (kv[0][1], kv[0][0])):
    rows_m.append([m, b] + [None] * 10 + ["failed: " + why[0][:120]])


def write(name, head, rows, prec):
    with open(os.path.join(OUT, name), "w") as fh:
        fh.write("\t".join(head) + "\n")
        for r in rows:
            fh.write("\t".join(f(x, prec.get(i, 1)) for i, x in enumerate(r)) + "\n")


write("phase5_memory.tsv", ["method", "bank", "ctx_mb", "fixed_mb", "resident_mb", "build_peak_mb",
                            "search_peak_256_mb", "search_peak_512_mb", "process_peak_mb", "build_ms",
                            "host_rss_mb", "trt_activation_mb", "runs"], rows_m, {9: 0})
write("phase5_speed.tsv", ["method", "bank", "param", "crop", "ms", "max_crops_s", "util_pct",
                           "cpu_cores", "recall1", "score_err_pct", "runs"], rows_s, {4: 3, 8: 4, 9: 3})
write("phase5_sweep.tsv", ["method", "bank", "param", "frac", "crops_s", "duty_pct", "util_pct",
                           "cpu_cores"], rows_w, {3: 2})
print("ok %d, failed %d runs" % (len(ok), len(bad)))

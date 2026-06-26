"""STAGE A compare: bf16-reduce ANCHOR vs fp32-reduce TEST, cos(g,W) on body matrices.

Reads the two in-situ probe JSONs (same checkpoint step 35500, T=8192, only
FSDP_reduce_dtype differs) and reports median/mean/negfrac of the RAW fresh-gradient
radial cosine cos_grad_W over the 490 body matrices, plus the full NorMuon stage trace.
"""
import json, statistics as st, sys

BODY = ('wo.weight', 'w2.weight', 'wq.weight', 'wk.weight',
        'wv.weight', 'w1.weight', 'w3.weight')
STAGES = ['cos_grad_W', 'cos_warmmom_W', 'cos_afterNS_W',
          'cos_afterscale_W', 'cos_afternormuon_W', 'total_dW_radial']


def summ(vals):
    return dict(median=st.median(vals), mean=st.mean(vals),
                negfrac=sum(1 for v in vals if v < 0) / len(vals), n=len(vals))


def load(path):
    d = json.load(open(path))
    rows = [r for r in d['per_matrix'] if any(r['name'].endswith(s) for s in BODY)]
    return d.get('step'), rows


def report(tag, path):
    step, rows = load(path)
    print(f"=== {tag}  ({path})  step={step}  n_body={len(rows)} ===")
    out = {}
    for k in STAGES:
        v = [r[k] for r in rows if k in r]
        if not v:
            print(f"  {k:22s}  (absent)")
            continue
        s = summ(v)
        out[k] = s
        print(f"  {k:22s}  median={s['median']:+.5f}  mean={s['mean']:+.5f}  "
              f"negfrac={s['negfrac']*100:.0f}%  n={s['n']}")
    return out


if __name__ == '__main__':
    anchor = sys.argv[1] if len(sys.argv) > 1 else \
        '//valhalla/valhalla/code/ckpt/wd_insitu_mf_BF16_t8192.json'
    test = sys.argv[2] if len(sys.argv) > 2 else \
        '//valhalla/valhalla/code/ckpt/wd_insitu_mf_FP32_t8192.json'
    a = report('BF16-REDUCE ANCHOR', anchor)
    print()
    try:
        t = report('FP32-REDUCE TEST', test)
    except FileNotFoundError:
        print(f"FP32 test JSON not present yet: {test}")
        sys.exit(0)
    print("\n=== VERDICT (cos_grad_W body median) ===")
    ac = a['cos_grad_W']['median']; tc = t['cos_grad_W']['median']
    print(f"  bf16-reduce: {ac:+.5f} ({a['cos_grad_W']['negfrac']*100:.0f}% neg)")
    print(f"  fp32-reduce: {tc:+.5f} ({t['cos_grad_W']['negfrac']*100:.0f}% neg)")
    print(f"  delta (fp32 - bf16): {tc-ac:+.5f}")
    if tc < -0.005:
        print("  => fp32-reduce STILL leans anti-radial: bf16 reduce-scatter is NOT the driver.")
    elif abs(tc) < 0.002:
        print("  => fp32-reduce KILLS the lean: bf16 reduce-scatter IS the driver.")
    else:
        print("  => partial: fp32-reduce reduces but does not kill the lean.")

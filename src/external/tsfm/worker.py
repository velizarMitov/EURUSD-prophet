"""SIDE-environment worker: series in, deciles out (design D1, D3, D10).

Runs ONLY under .venv-tsfm, as a subprocess started with `python -I`, so it
imports nothing from this repo -- only the standard library, numpy, torch and
the pinned model packages. One JSON request on stdin, one JSON reply on stdout,
progress on stderr. Weights load from the local pinned folders with the hub
offline; the caller (client.py / finetune.py / replay.py) verifies their digests
first.

Ops
  forecast   {"model", "weights_dir", "jobs": [{"key", "adapter_dir", "context", "series"}]}
             -> {"deciles": {key: [9 floats]}}; one call per model per pass
  rolling    {"model", "weights_dir", "adapter_dir", "context", "series_path",
              "positions_path", "out_path"}: deciles for every as-of position
             of a .npy series, saved to out_path (.npy, n x 9). Replay and holdout.
  finetune   LoRA on a .npy returns series, optionally scoring a holdout every
             `eval_every` steps (stage 1), optionally saving the adapter (stage 2)
  smoke      9 increasing deciles from each model on a synthetic series, and
             TimesFM's training path against the package's own forecast()

Units: the worker is unit-agnostic. Deciles come back in the units of the
series it was given (log returns in % or closes); forecast.py maps them.

TimesFM 2.5 is driven through ONE first-step path (`tfm_first_step`) for
training, holdout, replay and the hourly forecast. It reproduces the package's
compiled decode for a 1-step horizon -- input normalisation, the continuous
quantile head, flip invariance and the quantile-crossing fix -- without the
decode caches that would not fit 8 GB at long context; `smoke` checks it
against the package's forecast().
"""

import json
import os
import sys
import time

os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import numpy as np  # noqa: E402

QUANTILES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
TFM_LORA_TARGETS = r'stacked_xf\.\d+\.(attn\.qkv_proj|attn\.out|ff0|ff1)'
LORA = {'r': 8, 'lora_alpha': 16, 'lora_dropout': 0.0}
GPU_FREE_BYTES = 2 * 1024 ** 3


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def pick_device(pref='auto'):
    import torch
    if pref == 'cpu' or not torch.cuda.is_available():
        return 'cpu'
    if pref == 'cuda':
        return 'cuda'
    free, _total = torch.cuda.mem_get_info()
    return 'cuda' if free >= GPU_FREE_BYTES else 'cpu'


def pinball(deciles, y):
    d = np.asarray(deciles, float)
    e = np.asarray(y, float)[:, None] - d
    q = np.asarray(QUANTILES)[None, :]
    return float(np.mean(np.maximum(q * e, (q - 1.0) * e)))


# ── Chronos-2 ───────────────────────────────────────────────────────────────

def load_chronos(weights_dir, adapter_dir, device):
    from chronos import Chronos2Pipeline
    pipe = Chronos2Pipeline.from_pretrained(weights_dir)
    if adapter_dir:
        from peft import PeftModel
        pipe = Chronos2Pipeline(model=PeftModel.from_pretrained(pipe.model, adapter_dir).merge_and_unload())
    pipe.model.to(device)
    pipe.model.eval()
    return pipe


def chronos_deciles(pipe, windows, context, batch_size=64):
    q, _ = pipe.predict_quantiles([np.asarray(w, np.float32) for w in windows], prediction_length=1,
                                  quantile_levels=QUANTILES, batch_size=batch_size, context_length=context)
    return np.stack([t[0, 0, :].float().numpy() for t in q]).astype(float)


# ── TimesFM 2.5 ─────────────────────────────────────────────────────────────

def load_timesfm(weights_dir, adapter_dir, device):
    import torch
    from safetensors.torch import load_file
    from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch_module
    mod = TimesFM_2p5_200M_torch_module()
    mod.load_state_dict(load_file(os.path.join(weights_dir, 'model.safetensors')), strict=True)
    if adapter_dir:
        tfm_inject_lora(mod)
        sd = load_file(os.path.join(adapter_dir, 'adapter_model.safetensors'))
        missing, unexpected = mod.load_state_dict(sd, strict=False)
        lora_missing = [k for k in missing if 'lora_' in k]
        if unexpected or lora_missing:
            raise RuntimeError(f'adapter does not fit the model: unexpected {unexpected[:3]}, '
                               f'missing {lora_missing[:3]}')
    mod.device = torch.device(device)
    mod.to(device)
    mod.eval()
    return mod


def tfm_inject_lora(mod):
    from peft import LoraConfig, inject_adapter_in_model
    inject_adapter_in_model(LoraConfig(target_modules=TFM_LORA_TARGETS, **LORA), mod)
    return mod


def _tfm_prefill(mod, x, masks):
    """Normalised (B, C) context -> (B, 10) first-step outputs with the
    continuous quantile head applied, still in normalised units."""
    import torch
    from timesfm.torch import util
    b = x.shape[0]
    p = mod.p
    patched = x.reshape(b, -1, p)
    pmask = masks.reshape(b, -1, p)
    n = torch.zeros(b, device=x.device)
    mu = torch.zeros(b, device=x.device)
    sigma = torch.zeros(b, device=x.device)
    mus, sigmas = [], []
    for i in range(patched.shape[1]):
        (n, mu, sigma), _ = util.update_running_stats(n, mu, sigma, patched[:, i], pmask[:, i])
        mus.append(mu)
        sigmas.append(sigma)
    cmu, csig = torch.stack(mus, dim=1), torch.stack(sigmas, dim=1)
    normed = util.revin(patched, cmu, csig, reverse=False)
    normed = torch.where(pmask, 0.0, normed)
    (_, _, out_ts, out_qs), _ = mod(normed, pmask, None)
    pf = util.revin(out_ts, cmu, csig, reverse=True).reshape(b, -1, mod.o, mod.q)[:, -1, 0, :]
    qs = util.revin(out_qs, cmu, csig, reverse=True).reshape(b, -1, mod.os, mod.q)[:, -1, 0, :]
    idx = [1, 2, 3, 4, 6, 7, 8, 9]
    out = pf.clone()
    out[:, idx] = qs[:, idx] - qs[:, 5:6] + pf[:, 5:6]
    return out


def tfm_first_step(mod, x, masks):
    """(B, C) raw windows -> (B, 9) deciles in the input's units: the package's
    compiled decode at horizon 1 with normalize_inputs, the continuous quantile
    head, force_flip_invariance and fix_quantile_crossing."""
    import torch
    from timesfm.torch import util
    mu = torch.mean(x, dim=-1, keepdim=True)
    sigma = torch.std(x, dim=-1, keepdim=True)
    xn = util.revin(x, mu, sigma, reverse=False)
    out = _tfm_prefill(mod, xn, masks)
    flipped = _tfm_prefill(mod, -xn, masks)
    flipped = torch.cat([flipped[:, :1], torch.flip(flipped[:, 1:], dims=(-1,))], dim=-1)
    out = (out - flipped) / 2
    cols = [out[:, i] for i in range(10)]
    for i in (4, 3, 2, 1):
        cols[i] = torch.minimum(cols[i], cols[i + 1])
    for i in (6, 7, 8, 9):
        cols[i] = torch.maximum(cols[i], cols[i - 1])
    out = torch.stack(cols, dim=-1)
    out = util.revin(out, mu, sigma, reverse=True)
    return out[:, 1:10]


def _tfm_batch(windows, context, device):
    """Left-pad each window to `context` with masked zeros, as forecast() does."""
    import torch
    vals, masks = [], []
    for w in windows:
        w = np.asarray(w, float)[-context:]
        pad = context - len(w)
        vals.append(np.pad(w, (pad, 0)))
        masks.append(np.r_[np.ones(pad, bool), np.zeros(len(w), bool)])
    return (torch.tensor(np.array(vals), dtype=torch.float32, device=device),
            torch.tensor(np.array(masks), dtype=torch.bool, device=device))


def timesfm_deciles(mod, windows, context, batch_size=32):
    import torch
    out = []
    with torch.no_grad():
        for i in range(0, len(windows), batch_size):
            x, m = _tfm_batch(windows[i:i + batch_size], context, mod.device)
            out.append(tfm_first_step(mod, x, m).float().cpu().numpy())
    return np.concatenate(out).astype(float)


# ── shared ──────────────────────────────────────────────────────────────────

def load(model, weights_dir, adapter_dir, device):
    return (load_chronos if model == 'chronos2' else load_timesfm)(weights_dir, adapter_dir, device)


def deciles(model, handle, windows, context, batch_size):
    """Deciles with the batch halved on out-of-memory until it fits."""
    import torch
    fn = chronos_deciles if model == 'chronos2' else timesfm_deciles
    while True:
        try:
            return fn(handle, windows, context, batch_size), batch_size
        except torch.OutOfMemoryError:
            if batch_size == 1:
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            log(f'out of memory: batch halved to {batch_size}')


def windows_at(series, positions, context):
    return [series[max(0, int(p) - context + 1):int(p) + 1] for p in positions]


def op_forecast(req):
    device = pick_device(req.get('device', 'auto'))
    out, handles = {}, {}
    for job in req['jobs']:
        ad = job.get('adapter_dir') or None
        if ad not in handles:
            handles[ad] = load(req['model'], req['weights_dir'], ad, device)
        d, _ = deciles(req['model'], handles[ad], [np.asarray(job['series'], float)], int(job['context']), 1)
        out[job['key']] = [float(v) for v in d[0]]
    return {'deciles': out, 'device': device}


def op_rolling(req):
    device = pick_device(req.get('device', 'cuda'))
    handle = load(req['model'], req['weights_dir'], req.get('adapter_dir'), device)
    series = np.load(req['series_path'])
    pos = np.load(req['positions_path'])
    context, bs = int(req['context']), int(req.get('batch_size', 64))
    chunk = int(req.get('chunk', 1024))
    res = []
    for i in range(0, len(pos), chunk):
        d, bs = deciles(req['model'], handle, windows_at(series, pos[i:i + chunk], context), context, bs)
        res.append(d)
        log(f'rolling {req["model"]}: {min(i + chunk, len(pos))}/{len(pos)}')
    arr = np.concatenate(res) if res else np.zeros((0, 9))
    np.save(req['out_path'], arr)
    return {'rows': int(len(arr)), 'device': device, 'batch_size': bs}


# ── fine-tuning ─────────────────────────────────────────────────────────────

def _holdout(req):
    if not req.get('holdout_series_path'):
        return None
    s = np.load(req['holdout_series_path'])
    pos = np.load(req['holdout_positions_path'])
    return s, pos, s[pos + 1]


def _probe(model, weights_dir, adapter_dir, hold, context):
    """Deciles on the first 64 holdout windows from a FRESH load on CPU, for the
    exact-reload check."""
    if hold is None:
        return None
    s, pos, _ = hold
    h = load(model, weights_dir, adapter_dir, 'cpu')
    d, _ = deciles(model, h, windows_at(s, pos[:64], context), context, 16)
    return d.tolist()


def finetune_chronos(req, hold, curve):
    import torch
    from transformers.trainer_callback import TrainerCallback
    from chronos import Chronos2Pipeline
    device = pick_device('cuda')
    base = load_chronos(req['weights_dir'], None, device)
    context, every = int(req['context']), int(req.get('eval_every', 0))

    def score(pipe):
        s, pos, y = hold
        d, _ = deciles('chronos2', pipe, windows_at(s, pos, context), context, 128)
        return pinball(d, y)

    class Holdout(TrainerCallback):
        def on_step_end(self, args, state, control, model=None, **kw):
            if every and state.global_step % every == 0:
                model.eval()
                with torch.no_grad():
                    curve.append([int(state.global_step), score(Chronos2Pipeline(model=model))])
                log(f'chronos2 step {state.global_step}: holdout pinball {curve[-1][1]:.6f}')
                model.train()

    if hold is not None and every:
        curve.append([0, score(base)])
    train = np.load(req['train_path']).astype(np.float32)
    out_dir = req.get('out_dir') or os.path.join(req['work_dir'], 'stage1')
    tuned = base.fit([train], prediction_length=1, finetune_mode='lora', lora_config=dict(
        target_modules=['self_attention.q', 'self_attention.v', 'self_attention.k', 'self_attention.o',
                        'output_patch_embedding.output_layer'], **LORA),
        context_length=context, learning_rate=float(req['learning_rate']), num_steps=int(req['steps']),
        batch_size=int(req['batch_size']), output_dir=req['work_dir'], min_past=context,
        finetuned_ckpt_name='ckpt', callbacks=[Holdout()] if (hold is not None and every) else [],
        remove_printer_callback=True, lr_scheduler_type='constant', seed=int(req['seed']),
        data_seed=int(req['seed']))
    if req.get('out_dir'):
        os.makedirs(out_dir, exist_ok=True)
        tuned.model.save_pretrained(out_dir)
    return tuned


def finetune_timesfm(req, hold, curve):
    import torch
    device = pick_device('cuda')
    seed = int(req['seed'])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mod = load_timesfm(req['weights_dir'], None, device)
    context, every = int(req['context']), int(req.get('eval_every', 0))

    def score():
        s, pos, y = hold
        mod.eval()
        d, _ = deciles('timesfm25', mod, windows_at(s, pos, context), context, 64)
        mod.train()
        return pinball(d, y)

    if hold is not None and every:
        curve.append([0, score()])
    for p in mod.parameters():
        p.requires_grad_(False)
    tfm_inject_lora(mod)
    mod.to(device)
    params = [p for n, p in mod.named_parameters() if 'lora_' in n]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=float(req['learning_rate']))
    train = np.load(req['train_path']).astype(np.float32)
    q = torch.tensor(QUANTILES, device=device)[None, :]
    bs, steps = int(req['batch_size']), int(req['steps'])
    micro = int(req.get('micro_batch') or bs)          # gradient accumulation keeps the batch at bs
    mod.train()
    for step in range(1, steps + 1):
        ends = rng.integers(context - 1, len(train) - 1, size=bs)
        opt.zero_grad(set_to_none=True)
        for part in np.array_split(ends, max(1, bs // micro)):
            x = torch.tensor(np.stack([train[e - context + 1:e + 1] for e in part]), device=device)
            y = torch.tensor(train[part + 1], device=device)
            sig = torch.clamp(torch.std(x, dim=-1), min=1e-6)
            d = tfm_first_step(mod, x, torch.zeros_like(x, dtype=torch.bool))
            e = (y[:, None] - d) / sig[:, None]
            loss = torch.mean(torch.maximum(q * e, (q - 1.0) * e)) * (len(part) / bs)
            loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if every and step % every == 0:
            with torch.no_grad():
                curve.append([step, score()])
            log(f'timesfm25 step {step}: holdout pinball {curve[-1][1]:.6f}')
    if req.get('out_dir'):
        from safetensors.torch import save_file
        os.makedirs(req['out_dir'], exist_ok=True)
        save_file({n: p.detach().cpu().contiguous() for n, p in mod.state_dict().items() if 'lora_' in n},
                  os.path.join(req['out_dir'], 'adapter_model.safetensors'))
        with open(os.path.join(req['out_dir'], 'adapter_config.json'), 'w', encoding='utf-8') as fh:
            json.dump({'target_modules': TFM_LORA_TARGETS, **LORA}, fh, indent=1, sort_keys=True)
    return mod


def op_finetune(req):
    import torch
    hold = _holdout(req)
    curve = []
    t0 = time.time()
    (finetune_chronos if req['model'] == 'chronos2' else finetune_timesfm)(req, hold, curve)
    out = {'curve': curve, 'seconds': round(time.time() - t0, 1),
           'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
           'peak_mem_gb': round(torch.cuda.max_memory_allocated() / 1024 ** 3, 2) if torch.cuda.is_available() else 0}
    if req.get('out_dir'):
        out['probe'] = _probe(req['model'], req['weights_dir'], req['out_dir'], hold, int(req['context']))
    return out


def op_probe(req):
    hold = _holdout(req)
    return {'probe': _probe(req['model'], req['weights_dir'], req.get('adapter_dir'), hold, int(req['context']))}


# ── smoke ───────────────────────────────────────────────────────────────────

def op_smoke(req):
    import torch
    rng = np.random.default_rng(0)
    series = np.cumsum(rng.normal(0, 0.2, 4096)).astype(float)
    windows = [series[:-k] if k else series for k in range(4)]
    out = {}
    for model, wdir in req['weights'].items():
        h = load(model, wdir, None, pick_device(req.get('device', 'auto')))
        d, _ = deciles(model, h, windows, 2048, 4)
        out[model] = {'deciles': [round(float(v), 6) for v in d[0]],
                      'increasing': bool(np.all(np.diff(d, axis=1) > 0))}
        if model == 'timesfm25':
            import timesfm
            m = timesfm.TimesFM_2p5_200M_torch(torch_compile=False)
            m.load_checkpoint(os.path.join(wdir, 'model.safetensors'), torch_compile=False)
            m.compile(timesfm.ForecastConfig(max_context=2048, max_horizon=128, normalize_inputs=True,
                                             per_core_batch_size=4, use_continuous_quantile_head=True,
                                             force_flip_invariance=True, infer_is_positive=False,
                                             fix_quantile_crossing=True))
            _, qf = m.forecast(horizon=1, inputs=[w[-2048:] for w in windows])
            ref = qf[:, 0, 1:10]
            out[model]['max_abs_diff_vs_package'] = float(np.max(np.abs(ref - d)))
            out[model]['scale'] = float(np.mean(np.abs(ref)))
        del h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    out['torch'] = torch.__version__
    out['cuda'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return out


OPS = {'forecast': op_forecast, 'rolling': op_rolling, 'finetune': op_finetune, 'probe': op_probe,
       'smoke': op_smoke}


def main():
    """The request comes on stdin, or from `--request <file>` (Windows
    PowerShell 5.1 does not pipe text to a native process reliably)."""
    if len(sys.argv) == 3 and sys.argv[1] == '--request':
        with open(sys.argv[2], encoding='utf-8-sig') as fh:
            raw = fh.read()
    else:
        raw = sys.stdin.read()
    import faulthandler
    faulthandler.enable()
    t0 = time.time()
    try:
        req = json.loads(raw)
        # Libraries print to stdout (transformers' Trainer prints its metrics):
        # stdout carries the reply and nothing else.
        import contextlib
        with contextlib.redirect_stdout(sys.stderr):
            reply = OPS[req['op']](req)
        reply.update(ok=True, seconds=round(time.time() - t0, 2))
    except Exception as e:                       # noqa: BLE001 -- every failure goes back as data
        import traceback
        log(traceback.format_exc())
        reply = {'ok': False, 'error': f'{type(e).__name__}: {e}'}
    log(f'reply ready: ok={reply.get("ok")}')
    sys.stdout.write(json.dumps(reply))
    sys.stdout.flush()
    return 0 if reply.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())

# =====================================================================================================
# Samprity - fine-tuned multilingual-e5-small, configuration "E_balanced"  (FT family "ft")
# ONE self-contained Kaggle cell. Paste into the team notebook and run (GPU T4, internet ON).
#
# Writes to /kaggle/working/:
#   oof_ft_A.npy  oof_ft_B.npy  oof_ft_C.npy     (86, 6)      rows = sample.csv order
#   test_ft_A.npy test_ft_B.npy test_ft_C.npy    (500000, 6)  rows = test.csv order
#   ft_manifest.json                             config, data hashes, fold check, timings, OOF scores
# Columns: A = classes 0..5, B = techniques T1..T6, C = actions 0..5.
# A and C are softmax probabilities (rows sum to 1). B is six independent sigmoids.
#
# Rulebook v1.3: training data = sample.csv + embedded synthetic rows (Qwen2.5-7B-Instruct, official class
# definitions only) + embedded CC BY 4.0 external rows. NO pseudo-labels, NO test text in any fitting step
# (no preprocessing fit, calibration, threshold or augmentation uses test). test.csv is read only after
# every model is trained, and only for the final inference.
#
# Folds: the team's shared split, RepeatedStratifiedKFold(5 folds x 5 repeats, random_state=42) on label
# (= folds_5x5.npy). If /kaggle/working/folds_5x5.npy exists it is used and checked against the rebuild.
# OOF aggregation: each row is predicted once per repeat (by the model that did not see it);
# the saved OOF is the mean over the repeats used -> (86, 6). Same as the embedding family (oo.mean(0)).
# FT_REPEATS = 1 uses only repeat 0 (5 fits), like the xlmr family.
# =====================================================================================================
import os, io, re, gc, json, math, time, glob, base64, gzip, hashlib, random
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F

FT_REPEATS    = int(os.environ.get('FT_REPEATS', 5))       # 5 = all 25 fits (~25 min); 1 = 5 fits (~8 min)
SEED          = 42
MODEL_NAME    = 'intfloat/multilingual-e5-small'          # MIT licence
OOF_MAX_LEN   = 128                                        # as in the AutoTrain CV screen
FINAL_MAX_LEN = 256                                        # as in the AutoTrain final fit
SMOKE         = os.environ.get('FT_SMOKE') == '1'          # local pipeline check only (tiny random encoder)
CFG = dict(model_name=MODEL_NAME, prefix='query: ', epochs=6, batch_size=32, lr_encoder=5e-5, lr_head=1e-3,
           weight_decay=0.01, warmup_ratio=0.1, dropout=0.1, freeze_embeddings=True, loss_weights=(1.0, 0.5, 0.5),
           grad_clip=1.0, eval_batch_size=64, fp16=True, real_repeat=6, external_max=600,
           augment='corrupt', augment_copies=1)            # = AutoTrain candidate E_balanced (no pseudo_path)
if SMOKE:
    CFG.update(epochs=1, batch_size=8, real_repeat=1, external_max=40)
    OOF_MAX_LEN = FINAL_MAX_LEN = 32

D = os.environ.get('FT_DATA_DIR') or next(p for p in ['/kaggle/input/competitions/nsucec-neurahack-2026-preliminary/',
                                                    '/kaggle/input/nsucec-neurahack-2026-preliminary/']
                                          if os.path.exists(p + 'sample.csv'))
OUT = os.environ.get('FT_OUT_DIR', '/kaggle/working/')
os.makedirs(OUT, exist_ok=True)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
T0 = time.time()
TECH = [f'T{i}' for i in range(1, 7)]

def set_seed(s):
    os.environ['PYTHONHASHSEED'] = str(s); random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s); torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

# ---------------------------------------------------------------- embedded training-only data (gzip+base64)
# synth_filtered.csv : output of models/finetune/llm.py synth (Qwen/Qwen2.5-7B-Instruct, Apache-2.0, greedy,
#                      seeded spec grid, official definitions only; rows near sample.csv removed).
#                      Produced by notebook samprityhaque/scamshield-autotrain v2. Contains no test text.
# external_v2.csv    : 4 CC BY 4.0 Mendeley datasets (BTTC 10.17632/5wrm959d6f.1, Labeled Bangla SMS Spam/Ham
#                      10.17632/hhd43jbv9p.1, Financial scams 10.17632/znsk27yk3h.2, SMS Phishing Mishra & Soni
#                      10.17632/f45bkkt8pr.1). Bangalabarta (CC BY-NC-SA) excluded. Partial labels.
SYNTH_B64 = "__SYNTH_B64__"
SYNTH_SHA = "__SYNTH_SHA__"
EXT_B64   = "__EXT_B64__"
EXT_SHA   = "__EXT_SHA__"

def unpack(b64, sha, name):
    raw = gzip.decompress(base64.b64decode(b64))
    got = hashlib.sha256(raw).hexdigest()[:16]
    assert got == sha, f'{name}: embedded data corrupted ({got} != {sha})'
    return pd.read_csv(io.BytesIO(raw), keep_default_na=False, dtype={'id': str, 'text': str}), got

# ---------------------------------------------------------------- text handling (same as repo common/)
_SPK = re.compile(r'^([^\w]*?)\s*([A-Za-z]{1,12})\s*:\s*(.*)$', re.S)
def split_turns(text, sep='|||'):
    text = '' if text is None or (isinstance(text, float) and text != text) else str(text)
    parts = [p.strip() for p in text.split(sep)] if sep in text else [text.strip()]
    out = []
    for p in parts:
        m = _SPK.match(p)
        if m and len(parts) > 1 and len(m.group(2)) <= 2:
            out.append((m.group(2).upper(), (m.group(1).strip() + ' ' + m.group(3).strip()).strip()))
        else:
            out.append((None, p))
    return out

_HOMO = {'a': 'а', 'e': 'е', 'o': 'о', 'p': 'р', 'c': 'с', 'x': 'х', 'i': 'і', 'y': 'у'}
_LEET = {'i': '1', 'o': '0', 'e': '3', 'a': '4', 's': '5', 'l': '1'}
_EMOJI = ['\U0001F381', '\U0001F525', '\U0001F4B0', '✅', '⚠️', '\U0001F449']
_ZWC = ['​', '‌', '‍']
def corrupt(text, seed, strength=0.15):          # training-fold augmentation only (augmentation/corrupt.py)
    rng = random.Random(f'{seed}:{text}')
    out = []
    for ch in text:
        low = ch.lower(); r = rng.random()
        if ch.isalpha() and ch.isascii() and r < strength:
            op = rng.randint(0, 5)
            if op == 0 and low in _HOMO: out.append(_HOMO[low]); continue
            if op == 1 and low in _LEET: out.append(_LEET[low]); continue
            if op == 2: out.append(ch * rng.randint(2, 4)); continue
            if op == 3: continue
            if op == 4: out.append(ch.swapcase()); continue
            out.append(ch + rng.choice(_ZWC)); continue
        out.append(ch)
    toks = ''.join(out).split(' ')
    if rng.random() < 0.6:
        toks.insert(rng.randint(0, len(toks)), ''.join(rng.choice('abcdefghijklmnopqrstuvwxyz') for _ in range(rng.randint(5, 10))))
    if rng.random() < 0.3:
        toks.insert(rng.randint(0, len(toks)), rng.choice(_EMOJI))
    return ' '.join(toks)

def tech_matrix(vals):
    M = np.zeros((len(vals), 6), np.int8)
    for i, v in enumerate(vals):
        s = str(v).strip().upper()
        if s in ('', 'NONE'):
            continue
        for c in s.split('|'):
            assert c.strip() in TECH, f'bad technique {v!r}'
            M[i, TECH.index(c.strip())] = 1
    return M

def frame(ids, texts, label, action, tech):
    df = pd.DataFrame({'id': np.asarray(ids).astype(str), 'raw_text': [str(t) for t in texts]})
    df['turns'] = [split_turns(t) for t in df['raw_text']]
    df['label'] = np.asarray(label, int); df['action'] = np.asarray(action, int)
    TB = tech_matrix(list(tech))
    for j, t in enumerate(TECH):
        df[t] = TB[:, j]
    return df

def drop_near_sample(df, sample_texts, thr=0.8):   # fitted on training-side texts only (never test)
    from sklearn.feature_extraction.text import TfidfVectorizer
    nz = lambda x: pd.Series(list(x)).astype(str).str.lower().str.replace(r'\s+', ' ', regex=True).str.strip()
    e, s = nz(df['text']), nz(sample_texts)
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), sublinear_tf=True).fit(pd.concat([e, s]))
    mx = (vec.transform(e) @ vec.transform(s).T).max(axis=1).toarray().ravel()
    keep = (mx < thr) & ~e.isin(set(s)).to_numpy()
    return df[keep].reset_index(drop=True), int((~keep).sum())

# ---------------------------------------------------------------- data
raw_tr = pd.read_csv(D + 'sample.csv', keep_default_na=False, dtype={'text': str})
TEXT = next(c for c in ['text', 'message', 'conversation', 'content'] if c in raw_tr.columns)
train = frame(raw_tr['id'], raw_tr[TEXT], raw_tr['label'], raw_tr['action'], raw_tr['techniques'])
n = len(train)
yA, yC, YB = train['label'].to_numpy(), train['action'].to_numpy(), train[TECH].to_numpy()
assert set(np.unique(yA)) <= set(range(6)) and set(np.unique(yC)) <= set(range(6))
assert (pd.read_csv(D + 'sample.csv')['id'].astype(str).to_numpy() == train['id'].to_numpy()).all()  # team row order

syn_raw, syn_sha = unpack(SYNTH_B64, SYNTH_SHA, 'synthetic')
syn_raw = syn_raw[syn_raw['text'].str.len() > 0]
syn_raw, syn_drop = drop_near_sample(syn_raw, raw_tr[TEXT])
SYN = frame(syn_raw['id'], syn_raw['text'], syn_raw['label'], syn_raw['action'], syn_raw['techniques'])
assert SYN['label'].between(0, 5).all() and SYN['action'].between(0, 5).all()

ext_raw, ext_sha = unpack(EXT_B64, EXT_SHA, 'external')
ext_raw = ext_raw[ext_raw['text'].str.len() > 0]
ext_raw, ext_drop = drop_near_sample(ext_raw, raw_tr[TEXT])
lab = ext_raw['label'].astype(int).to_numpy()
tech = ext_raw['techniques'].fillna('')
EXT = frame(ext_raw['id'], ext_raw['text'], lab, ext_raw['action'].astype(int).clip(lower=0), tech.where(tech != '', 'NONE'))
masks = []
for y_, allowed in zip(lab, ext_raw['allowed'].fillna('')):
    mk = np.zeros(6, np.float32)
    if y_ >= 0: mk[y_] = 1.0
    else:
        for a in str(allowed).split('|'): mk[int(a)] = 1.0
    masks.append(mk)
EXT['a_mask'] = masks
EXT['b_known'] = (tech != '').to_numpy()
EXT['c_known'] = (ext_raw['action'].astype(int) >= 0).to_numpy()
print(f'train {n} | synthetic {len(SYN)} (-{syn_drop} near sample) | external {len(EXT)} (-{ext_drop} near sample)')

# ---------------------------------------------------------------- team folds (folds_5x5)
from sklearn.model_selection import RepeatedStratifiedKFold
splits = list(RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED).split(np.zeros(n), yA))
folds = np.zeros((5, n), int)
for i, (_, va) in enumerate(splits):
    folds[i // 5, va] = i % 5
team_path = OUT + 'folds_5x5.npy'
if os.path.exists(team_path):
    team = np.load(team_path)
    assert team.shape == folds.shape and (team == folds).all(), 'folds_5x5.npy differs from the rebuilt team split'
    fold_src = 'loaded folds_5x5.npy (matches rebuild)'
else:
    fold_src = 'rebuilt RepeatedStratifiedKFold(5,5,42) on label (= folds_5x5.npy)'
fold_fp = hashlib.sha256(folds.tobytes()).hexdigest()[:16]
print('folds:', fold_src, '| fingerprint', fold_fp)

# ---------------------------------------------------------------- model (same as models/finetune/multitask.py)
def load_backbone(max_len):
    from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if SMOKE:                                           # random tiny encoder, same tokenizer
        enc = BertModel(BertConfig(vocab_size=len(tok), hidden_size=32, num_hidden_layers=1, num_attention_heads=2,
                                   intermediate_size=64, max_position_embeddings=max(64, max_len + 2)))
    else:
        enc = AutoModel.from_pretrained(MODEL_NAME)
    return tok, enc

class Net(nn.Module):
    def __init__(self, enc, p):
        super().__init__(); self.encoder = enc; h = enc.config.hidden_size
        self.drop = nn.Dropout(p); self.head_a, self.head_b, self.head_c = nn.Linear(h, 6), nn.Linear(h, 6), nn.Linear(h, 6)
    def forward(self, ids, am):
        x = self.encoder(input_ids=ids, attention_mask=am).last_hidden_state
        m = am.unsqueeze(-1).to(x.dtype); z = self.drop((x * m).sum(1) / m.sum(1).clamp(min=1e-6))
        return self.head_a(z), self.head_b(z), self.head_c(z)

def class_w(y, k=6):
    c = np.bincount(y, minlength=k).astype(float)
    w = np.where(c > 0, (len(y) / (k * np.maximum(c, 1))) ** 0.5, 0.0)
    return torch.tensor(w / w[w > 0].mean(), dtype=torch.float32)

def build_text(turns, raw, sep):
    body = f' {sep} '.join(f'{s}: {t}' if s else t for s, t in turns) if len(turns) > 1 else raw
    return CFG['prefix'] + body

def training_frame(real, seed):
    """real rows x real_repeat + synthetic + external sample, then + 1 corrupted copy. No test rows."""
    ex = EXT.sample(CFG['external_max'], random_state=seed) if len(EXT) > CFG['external_max'] else EXT
    df = pd.concat([real] * CFG['real_repeat'] + [SYN, ex], ignore_index=True)
    cp = df.copy(); cp['raw_text'] = [corrupt(t, 1000) for t in df['raw_text']]
    df = pd.concat([df, cp], ignore_index=True)
    df['turns'] = [split_turns(t) for t in df['raw_text']]
    return df

class FT:
    def __init__(self, max_len, seed=SEED):
        self.max_len, self.seed = max_len, seed
    def texts(self, df):
        sep = getattr(self.tok, 'sep_token', None) or '|'
        return [build_text(t, r, sep) for t, r in zip(df['turns'], df['raw_text'])]
    def enc(self, texts):
        return self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors='pt')
    def fit(self, real):
        c = CFG; set_seed(self.seed); t0 = time.time()
        self.tok, e = load_backbone(self.max_len); self.net = Net(e, c['dropout']).to(DEVICE)
        df = training_frame(real, self.seed); X = self.texts(df); N = len(X)
        ya, yc = df['label'].to_numpy(int), df['action'].to_numpy(int); yb = df[TECH].to_numpy(np.float32)
        am = np.zeros((N, 6), np.float32); col = df['a_mask'] if 'a_mask' in df else [None] * N
        for i, (y_, mk) in enumerate(zip(ya, col)):
            if isinstance(mk, (list, tuple, np.ndarray)): am[i] = mk
            else: am[i, y_] = 1.0
        bk = df['b_known'].fillna(True).astype(bool).to_numpy() if 'b_known' in df else np.ones(N, bool)
        ck = df['c_known'].fillna(True).astype(bool).to_numpy() if 'c_known' in df else np.ones(N, bool)
        wA = class_w(ya[ya >= 0]).to(DEVICE); wC = class_w(yc[ck]).to(DEVICE)
        pos = yb[bk].sum(0); pw = torch.tensor(np.clip(np.sqrt((bk.sum() - pos) / np.maximum(pos, 1)), 1, 5), dtype=torch.float32).to(DEVICE)
        if c['freeze_embeddings']:
            for p in self.net.encoder.embeddings.parameters(): p.requires_grad = False
        opt = torch.optim.AdamW([{'params': [p for p in self.net.encoder.parameters() if p.requires_grad], 'lr': c['lr_encoder']},
                                 {'params': [p for k, p in self.net.named_parameters() if not k.startswith('encoder.')], 'lr': c['lr_head']}],
                                weight_decay=c['weight_decay'])
        total = math.ceil(N / c['batch_size']) * c['epochs']; warm = max(1, int(c['warmup_ratio'] * total))
        sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total - s) / max(1, total - warm)))
        amp = c['fp16'] and DEVICE.type == 'cuda'; scaler = torch.amp.GradScaler('cuda', enabled=amp)
        gen = torch.Generator().manual_seed(self.seed); wl = c['loss_weights']; self.net.train()
        for _ in range(c['epochs']):
            order = torch.randperm(N, generator=gen).numpy()
            for i in range(0, N, c['batch_size']):
                idx = order[i:i + c['batch_size']]; b = self.enc([X[j] for j in idx])
                ta = torch.tensor(ya[idx], device=DEVICE); tb = torch.tensor(yb[idx], device=DEVICE); tc = torch.tensor(yc[idx], device=DEVICE)
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp):
                    la, lb, lc = self.net(b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE))
                la, lb, lc = la.float(), lb.float(), lc.float()
                # Task A: -log P(true class in allowed set) (= weighted CE on fully labeled rows); B/C only where known
                m = torch.as_tensor(am[idx], device=DEVICE)
                li = -torch.logsumexp(F.log_softmax(la, -1) + torch.log(m.clamp_min(1e-12)), -1)
                wi = torch.where(ta >= 0, wA[ta.clamp(min=0)], torch.ones_like(li))
                loss = wl[0] * (wi * li).sum() / wi.sum()
                bkt = torch.as_tensor(bk[idx], device=DEVICE); ckt = torch.as_tensor(ck[idx], device=DEVICE)
                if bkt.any():
                    loss = loss + wl[1] * F.binary_cross_entropy_with_logits(lb, tb, pos_weight=pw, reduction='none').mean(1)[bkt].mean()
                if ckt.any():
                    loss = loss + wl[2] * F.cross_entropy(lc[ckt], tc[ckt].clamp(min=0), weight=wC)
                opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), c['grad_clip']); scaler.step(opt); scaler.update(); sch.step()
        self.net.eval(); self.fit_s = round(time.time() - t0, 1); self.n_rows = N
        return self
    @torch.no_grad()
    def predict(self, df):
        X = self.texts(df); order = np.argsort([len(t) for t in X], kind='stable')
        outs = {'A': [], 'B': [], 'C': []}; amp = CFG['fp16'] and DEVICE.type == 'cuda'
        bs, i, t0, nxt = CFG['eval_batch_size'], 0, time.time(), 100_000
        while i < len(X):
            idx = order[i:i + bs]
            try:
                b = self.enc([X[j] for j in idx])
                with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=amp):
                    la, lb, lc = self.net(b['input_ids'].to(DEVICE), b['attention_mask'].to(DEVICE))
            except torch.cuda.OutOfMemoryError:
                assert bs > 1; bs //= 2; torch.cuda.empty_cache(); continue
            outs['A'].append(la.float().cpu()); outs['B'].append(lb.float().cpu()); outs['C'].append(lc.float().cpu())
            i += len(idx)
            if i >= nxt:
                print(f'  inference {i}/{len(X)} ({i / (time.time() - t0):.0f} rows/s)', flush=True); nxt += 100_000
        inv = np.empty_like(order); inv[order] = np.arange(len(order))
        L = {k: torch.cat(v)[torch.from_numpy(inv)] for k, v in outs.items()}
        return {'A': F.softmax(L['A'], -1).double().numpy(), 'B': torch.sigmoid(L['B']).double().numpy(),
                'C': F.softmax(L['C'], -1).double().numpy()}
    def release(self):
        self.peak_mb = round(torch.cuda.max_memory_allocated() / 2**20) if DEVICE.type == 'cuda' else None
        self.net = None; gc.collect()
        if DEVICE.type == 'cuda': torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()

def check_probs(P, rows, name, oof=False):
    for t in 'ABC':
        a = P[t]
        assert a.shape == (rows, 6), f'{name}_{t} shape {a.shape}'
        assert np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1, f'{name}_{t} outside [0,1]'
        if t != 'B':
            assert np.allclose(a.sum(1), 1, atol=1e-4), f'{name}_{t} rows do not sum to 1'

# ---------------------------------------------------------------- 1) cross-validation on the team folds
oof_sum = {t: np.zeros((n, 6)) for t in 'ABC'}; cover = np.zeros(n, int); fit_secs, peaks = [], []
for r in range(FT_REPEATS):
    for k in range(5):
        tr_idx, va_idx = np.where(folds[r] != k)[0], np.where(folds[r] == k)[0]
        m = FT(OOF_MAX_LEN).fit(train.iloc[tr_idx].reset_index(drop=True))
        p = m.predict(train.iloc[va_idx].reset_index(drop=True)); m.release()
        for t in 'ABC':
            oof_sum[t][va_idx] += p[t]
        cover[va_idx] += 1; fit_secs.append(m.fit_s); peaks.append(m.peak_mb or 0)
        print(f'repeat {r} fold {k}: fit {m.fit_s}s on {m.n_rows} rows | elapsed {(time.time() - T0) / 60:.1f} min', flush=True)
assert (cover == FT_REPEATS).all(), 'every row must be predicted exactly once per repeat'
OOF = {t: oof_sum[t] / FT_REPEATS for t in 'ABC'}
check_probs(OOF, n, 'oof_ft')

from sklearn.metrics import f1_score
pa = OOF['A'].argmax(1); fraud = ~np.isin(pa, [0, 5])
pc = np.where(fraud, OOF['C'][:, 1:].argmax(1) + 1, 0)
pb = (OOF['B'] >= 0.5).astype(int); pb[~fraud] = 0
oof_scores = {'F_A': round(f1_score(yA, pa, average='macro', labels=range(6), zero_division=0), 4),
              'accuracy_A': round(float((pa == yA).mean()), 4),
              'F_B': round(f1_score(YB, pb, average='macro', zero_division=0), 4),
              'F_C': round(f1_score(yC, pc, average='macro', labels=range(6), zero_division=0), 4)}
print('OOF (mean over', FT_REPEATS, 'repeat(s)):', oof_scores)
for t in 'ABC':
    np.save(f'{OUT}oof_ft_{t}.npy', OOF[t].astype(np.float32))

# ---------------------------------------------------------------- 2) final fit on all 86 rows -> test inference
final = FT(FINAL_MAX_LEN).fit(train)
print(f'final fit {final.fit_s}s on {final.n_rows} rows', flush=True)
# test.csv is opened only now, after all training. It is used for inference only.
raw_te = pd.read_csv(D + 'test.csv', keep_default_na=False, dtype={'text': str})
TT = next(c for c in ['text', 'message', 'conversation', 'content'] if c in raw_te.columns)
NT = len(raw_te)
assert (pd.read_csv(D + 'test.csv', usecols=['id'])['id'].astype(str).to_numpy() == raw_te['id'].astype(str).to_numpy()).all()
assert raw_te['id'].astype(str).is_unique
uniq, inv = np.unique(raw_te[TT].astype(str).to_numpy(), return_inverse=True)   # identical texts -> one forward pass
te = pd.DataFrame({'raw_text': uniq}); te['turns'] = [split_turns(t) for t in te['raw_text']]
t_inf = time.time(); pu = final.predict(te); inf_s = round(time.time() - t_inf, 1); final.release()
TEST = {t: pu[t][inv] for t in 'ABC'}
check_probs(TEST, NT, 'test_ft')
for t in 'ABC':
    np.save(f'{OUT}test_ft_{t}.npy', TEST[t].astype(np.float32))

# ---------------------------------------------------------------- 3) validation of the saved files + manifest
for split, rows in (('oof', n), ('test', NT)):
    for t in 'ABC':
        a = np.load(f'{OUT}{split}_ft_{t}.npy')
        assert a.shape == (rows, 6) and a.dtype == np.float32 and np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
manifest = {
    'family': 'ft', 'author': 'Samprity', 'config_name': 'E_balanced', 'model': MODEL_NAME, 'cfg': {k: (list(v) if isinstance(v, tuple) else v) for k, v in CFG.items()},
    'oof_max_len': OOF_MAX_LEN, 'final_max_len': FINAL_MAX_LEN, 'seed': SEED, 'smoke': SMOKE,
    'folds': fold_src, 'fold_fingerprint': fold_fp, 'repeats_used': FT_REPEATS,
    'oof_aggregation': 'mean over repeats of out-of-fold predictions (each row once per repeat)',
    'oof_scores_uncalibrated': oof_scores,
    'data': {'train_rows': n, 'synthetic_rows': len(SYN), 'synthetic_sha16': syn_sha, 'external_rows': len(EXT),
             'external_sha16': ext_sha, 'test_rows': NT, 'test_unique_texts': int(len(uniq))},
    'test_use': 'inference only, after training; no pseudo-labels; no test text in any fit, calibration, threshold or augmentation',
    'fit_seconds': fit_secs, 'final_fit_seconds': final.fit_s, 'inference_seconds': inf_s,
    'peak_gpu_mb': max(peaks + [final.peak_mb or 0]), 'device': torch.cuda.get_device_name(0) if DEVICE.type == 'cuda' else 'cpu',
    'versions': {'torch': torch.__version__, 'transformers': __import__('transformers').__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
    'total_minutes': round((time.time() - T0) / 60, 1)}
json.dump(manifest, open(OUT + 'ft_manifest.json', 'w'), indent=1)
print(json.dumps({k: manifest[k] for k in ('oof_scores_uncalibrated', 'peak_gpu_mb', 'inference_seconds', 'total_minutes')}))
print('saved oof_ft_{A,B,C}.npy', (n, 6), 'and test_ft_{A,B,C}.npy', (NT, 6), 'to', OUT)

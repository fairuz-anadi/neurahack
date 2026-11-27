# ---- candidates: shrunk ensemble vs. the TF-IDF + XLM-R blend from Part A (v2ens); keep the better CV proxy ----
CANDS = {'ens_' + '_'.join(NAMES): (mix(w, O), lambda: mix(w, TS), thr)}
if all(os.path.exists(f'{OUT}{s}_v2ens_{t}.npy') for s in ['oof', 'test'] for t in 'ABC'):
    Ov = load('v2ens', 'oof')
    thr_v = max(np.round(np.arange(0.3, 0.71, 0.05), 2), key=lambda t: proxy(*Ov, t)[0])
    CANDS['v2ens'] = (Ov, lambda: load('v2ens', 'test'), thr_v)
SCORES = {k: proxy(*v[0], v[2]) for k, v in CANDS.items()}
for k, s in SCORES.items(): print(f'{k:22s} thr {CANDS[k][2]} | CV proxy/A/B/C', s.round(3))
WIN = max(SCORES, key=lambda k: SCORES[k][0]); print('chosen for submission.csv:', WIN)

# ---- confidence: temperature by Brier, then test -> csv (one file per candidate) ----
def conf_of(P, tau):
    L = np.log(np.clip(P, 1e-9, 1)) / tau; L -= L.max(1, keepdims=True)
    S = np.exp(L); S /= S.sum(1, keepdims=True); return S.max(1)
for k, (Oc, Tc, th) in CANDS.items():
    PA, PB, PC = Oc; PA = PA / PA.sum(1, keepdims=True)
    correct = (post(PA, PB, PC, th)[0] == yA).astype(float)
    briers = {tau: np.mean((conf_of(PA, tau) - correct) ** 2) for tau in np.round(np.arange(0.05, 3.01, 0.05), 2)}
    tau = min(briers, key=briers.get); print(f'{k}: Brier raw {briers[1.0]:.4f} -> tau {tau}: {briers[tau]:.4f}')
    PA, PB, PC = Tc(); PA = PA / PA.sum(1, keepdims=True)
    a, B, act = post(PA, PB, PC, th)
    sub = pd.DataFrame({'id': test['id'].values, 'label': a.astype(int),
                        'techniques': ['|'.join(TC[j] for j in np.flatnonzero(r)) or 'NONE' for r in B],
                        'action': act.astype(int), 'confidence': np.clip(conf_of(PA, tau), 0.02, 0.98).round(4)})[ss_cols]
    assert list(sub.columns) == ['id', 'label', 'techniques', 'action', 'confidence']
    assert len(sub) == NT and sub['id'].is_unique and (sub['id'].values == test['id'].values).all()
    assert sub['label'].between(0, 5).all() and sub['action'].between(0, 5).all()
    assert sub['techniques'].str.fullmatch(r'NONE|T[1-6](\|T[1-6])*').all() and not sub.isna().any().any()
    sub.to_csv(OUT + f'sub_{k}.csv', index=False); print('wrote', f'sub_{k}.csv')
    if k == WIN:
        sub.to_csv(OUT + 'submission.csv', index=False)
        print('submission.csv =', k, '| test label share:', sub['label'].value_counts(normalize=True).sort_index().round(3).to_dict())

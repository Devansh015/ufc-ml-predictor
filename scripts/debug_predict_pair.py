import sys
import importlib.util
from pathlib import Path

if len(sys.argv) < 3:
    print('Usage: python scripts/debug_predict_pair.py "Fighter A" "Fighter B"')
    sys.exit(1)

name_a = sys.argv[1]
name_b = sys.argv[2]

spec = importlib.util.spec_from_file_location("webapp_app", Path('src/webapp/app.py').resolve())
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

normalize = mod.normalize
find_direct_matchup = mod.find_direct_matchup
build_synthetic_row = mod.build_synthetic_row
clf = mod.clf
imputer = mod.imputer
saved_features = mod.saved_features

df = mod.df

import pandas as pd

print('Testing matchup:', name_a, 'vs', name_b)

def get_probas(a_raw, b_raw):
    a = normalize(a_raw)
    b = normalize(b_raw)
    match_row, swapped = find_direct_matchup(df, a, b)
    if match_row is not None:
        X = match_row.to_frame().T
        X = X.reindex(columns=saved_features)
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0.0)
        if imputer is not None:
            X = pd.DataFrame(imputer.transform(X), columns=X.columns)
        proba_red = clf.predict_proba(X)[:, 1][0]
        proba_a = 1.0 - proba_red if swapped else proba_red
        proba_b = 1.0 - proba_a
        return {'method': 'historical_row', 'p_a': float(proba_a), 'p_b': float(proba_b)}
    Xsyn = build_synthetic_row(df, a, b, saved_features, use_recency=True, decay=0.9, last_n=5)
    Xsyn = Xsyn.reindex(columns=saved_features)
    Xsyn = Xsyn.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if imputer is not None:
        Xsyn = pd.DataFrame(imputer.transform(Xsyn), columns=Xsyn.columns)
    proba_red = clf.predict_proba(Xsyn)[:, 1][0]
    proba_a = proba_red
    proba_b = 1.0 - proba_a
    return {'method': 'synthetic_aggregates', 'p_a': float(proba_a), 'p_b': float(proba_b)}

res_ab = get_probas(name_a, name_b)
res_ba = get_probas(name_b, name_a)

print('\nOrder A,B ->', res_ab)
print('Order B,A ->', res_ba)

winner_ab = name_a if res_ab['p_a'] > res_ab['p_b'] else name_b
winner_ba = name_b if res_ba['p_a'] > res_ba['p_b'] else name_a
print('\nWinner A,B ->', winner_ab)
print('Winner B,A ->', winner_ba)

if winner_ab == winner_ba:
    print('\nNote: same real fighter predicted in both orders. Probabilities:')
    print('A,B p_a,p_b:', res_ab['p_a'], res_ab['p_b'])
    print('B,A p_a,p_b:', res_ba['p_a'], res_ba['p_b'])

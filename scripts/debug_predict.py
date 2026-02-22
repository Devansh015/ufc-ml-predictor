import importlib.util
from pathlib import Path

# load the app module without running the server
spec = importlib.util.spec_from_file_location("webapp_app", str(Path(__file__).resolve().parents[1] / "src" / "webapp" / "app.py"))
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# helpers
normalize = mod.normalize
find_direct_matchup = mod.find_direct_matchup
build_synthetic_row = mod.build_synthetic_row
clf = mod.clf
imputer = mod.imputer
saved_features = mod.saved_features

df = mod.df

# pick a sample matchup from the data
sample = df.iloc[0]
name_a = sample.get('red_fighter')
name_b = sample.get('blue_fighter')
print('Sample taken from first row:', name_a, 'vs', name_b)

# function to get probabilities for given order
import pandas as pd
import numpy as np

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
        return {'method': 'historical_row', 'p_a': proba_a, 'p_b': proba_b}
    Xsyn = build_synthetic_row(df, a, b, saved_features, use_recency=True, decay=0.9, last_n=5)
    Xsyn = Xsyn.reindex(columns=saved_features)
    Xsyn = Xsyn.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    if imputer is not None:
        Xsyn = pd.DataFrame(imputer.transform(Xsyn), columns=Xsyn.columns)
    proba_red = clf.predict_proba(Xsyn)[:, 1][0]
    proba_a = proba_red
    proba_b = 1.0 - proba_a
    return {'method': 'synthetic_aggregates', 'p_a': proba_a, 'p_b': proba_b}

# test both orders
print('\nOrder A,B:')
res_ab = get_probas(name_a, name_b)
print(res_ab)

print('\nOrder B,A:')
res_ba = get_probas(name_b, name_a)
print(res_ba)

# show whether winner flips
winner_ab = name_a if res_ab['p_a'] > res_ab['p_b'] else name_b
winner_ba = name_b if res_ba['p_a'] > res_ba['p_b'] else name_a
print('\nWinner A,B ->', winner_ab)
print('Winner B,A ->', winner_ba)

# If winners are the same fighter label (e.g., always fighter_a), show probabilities
if winner_ab == winner_ba:
    print('\nNote: winner is same when switching order. Probabilities:')
    print('A,B p_a,p_b:', res_ab['p_a'], res_ab['p_b'])
    print('B,A p_a,p_b:', res_ba['p_a'], res_ba['p_b'])

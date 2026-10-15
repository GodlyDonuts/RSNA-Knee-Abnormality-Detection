"""Pure index arithmetic. No images, clinical metadata, models, or network calls."""
from itertools import combinations
from math import comb
import ast, csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
source=Path('/workspace/shared/rsna-anatomical-audit-current/rsna/competitive/data.py').read_text()
node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='_indices')
module=ast.Module(body=[node], type_ignores=[])
ns={'np':np,'Tensor':object}
exec(compile(module,'native_indices_source','exec'),ns)
indices=ns['_indices']

def make(N):
    q=indices(N,16,False).tolist()
    sel=[tuple(q[j-1:j+2]) for j in range(1,len(q)-1)]
    loc=[(q[j]-1,q[j],q[j]+1) for j in range(1,len(q)-1)]
    assert [w[1] for w in sel]==[w[1] for w in loc]
    assert all(len(set(w))==3 and min(w)>=0 and max(w)<N for w in sel+loc)
    if N<=16: assert sel==loc
    return q,sel,loc

out=[]; rows=[]
for N in [3,8,16,17,24,31,32,46,61,96,128]:
    q,sel,loc=make(N)
    M=len(sel); K=min(2,M); bags=list(combinations(range(M),K))
    r={'N':N,'selected_indices':q,'centers':[w[1] for w in sel], 'candidate_windows':M,'encodes_per_series':K,'paired_bags':len(bags)}
    for label,windows in [('selected',sel),('local',loc)]:
        sets=[set().union(*(windows[j] for j in bag)) for bag in bags]
        union=set().union(*map(set,windows))
        spans=[w[2]-w[0] for w in windows]
        r[label]={'all_candidate_unique_originals':len(union), 'all_candidate_omitted_indices':[j for j in range(N) if j not in union], 'mean_unique_originals_per_bag':sum(map(len,sets))/len(bags), 'index_window_span_min':min(spans),'index_window_span_max':max(spans),'full_original_adjacency_window_fraction':sum(w[1]-w[0]==1 and w[2]-w[1]==1 for w in windows)/M}
        for k in range(1,N+1):
            contact=adj_contact=0
            for a in range(N-k+1):
                b=a+k-1
                for bag,U in zip(bags,sets):
                    contact += any(a<=x<=b for x in U)
                    adj_contact += any(any(w[t+1]==w[t]+1 and a<=w[t] and w[t+1]<=b for t in (0,1)) for w in [windows[j] for j in bag])
            den=(N-k+1)*len(bags)
            rows.append({'N':N,'k_acquired_indices':k,'arm':label,'numerator_any_hit':contact,'numerator_adjacent_pair_in_one_window':adj_contact,'denominator':den,'prob_any_hit_uniform_start':contact/den,'prob_adjacent_pair_uniform_start':adj_contact/den})
    out.append(r)
for N in range(3,1001): make(N)
(ROOT/'synthetic_sampling_results.json').write_text(json.dumps({'scope':'Pure index enumeration; uniform interval-start assumption is not a clinical distribution; no lesion size asserted.','index_invariants_N_3_through_1000':'passed','summary':out},indent=2))
with (ROOT/'synthetic_interval_curves.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader();w.writerows(rows)
print(json.dumps(out,indent=2))
print('Illustrative exact curves:')
for row in rows:
    if row['N'] in (16,31,46,61) and row['k_acquired_indices'] in (1,2,3,6,12): print(row)

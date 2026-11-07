#!/usr/bin/env python3
"""Audit 20: full objective Parquet statistics, resumable per-file summaries.

Requires: pandas, numpy, pyarrow. Run from the SoccerMon repository root.
"""
import argparse
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

SENSORS = ['lat','lon','speed','heart_rate','hacc','hdop','signal_quality','num_satellites','inst_acc_impulse','accl_x','accl_y','accl_z','gyro_x','gyro_y','gyro_z']
GPS = {'lat','lon','speed','heart_rate','hacc','hdop','signal_quality','num_satellites'}
FIELDS = ['rows','valid','missing','zero','negative','invalid','positive','sum','mean','M2','min','max','positive_sum','positive_mean','positive_M2','positive_min','positive_max']


def blank():
    return dict(rows=0,valid=0,missing=0,zero=0,negative=0,invalid=0,positive=0,sum=0.,mean=0.,M2=0.,min=None,max=None,positive_sum=0.,positive_mean=0.,positive_M2=0.,positive_min=None,positive_max=None)


def merge_moments(a_n,a_mean,a_m2,b_n,b_mean,b_m2):
    if not b_n: return a_n,a_mean,a_m2
    if not a_n: return b_n,b_mean,b_m2
    n=a_n+b_n; delta=b_mean-a_mean
    return n,a_mean+delta*b_n/n,a_m2+b_m2+delta*delta*a_n*b_n/n


def update(s,values,col):
    v=pd.to_numeric(pd.Series(values),errors='coerce').to_numpy(dtype=np.float64)
    s['rows']+=len(v)
    finite=np.isfinite(v)
    s['missing']+=int(np.isnan(v).sum())
    s['invalid']+=int(np.isinf(v).sum())
    x=v[finite]
    s['zero']+=int((x==0).sum()); s['negative']+=int((x<0).sum())
    if len(x):
        n,m,m2=merge_moments(s['valid'],s['mean'],s['M2'],len(x),float(x.mean()),float(np.square(x-x.mean()).sum()))
        s['valid'],s['mean'],s['M2']=n,m,m2
        s['sum']+=float(x.sum()); lo=float(x.min()); hi=float(x.max())
        s['min']=lo if s['min'] is None else min(s['min'],lo)
        s['max']=hi if s['max'] is None else max(s['max'],hi)
    p=x[x>0]
    if len(p):
        n,m,m2=merge_moments(s['positive'],s['positive_mean'],s['positive_M2'],len(p),float(p.mean()),float(np.square(p-p.mean()).sum()))
        s['positive'],s['positive_mean'],s['positive_M2']=n,m,m2
        s['positive_sum']+=float(p.sum()); lo=float(p.min()); hi=float(p.max())
        s['positive_min']=lo if s['positive_min'] is None else min(s['positive_min'],lo)
        s['positive_max']=hi if s['positive_max'] is None else max(s['positive_max'],hi)


def combine(a,b):
    out=blank()
    for k in ['rows','missing','zero','negative','invalid']:
        out[k]=a[k]+b[k]
    for prefix in ['', 'positive_']:
        nk='valid' if not prefix else 'positive'
        mk='mean' if not prefix else 'positive_mean'
        m2k='M2' if not prefix else 'positive_M2'
        out[nk],out[mk],out[m2k]=merge_moments(a[nk],a[mk],a[m2k],b[nk],b[mk],b[m2k])
        sk='sum' if not prefix else 'positive_sum'
        out[sk]=a[sk]+b[sk]
        for kind in ['min','max']:
            key=prefix+kind; vals=[v[key] for v in (a,b) if v[key] is not None]
            out[key]=(min(vals) if kind=='min' else max(vals)) if vals else None
    return out


def finish(s):
    n=s['rows']; v=s['valid']
    return dict(rows=n,valid=v,missing=s['missing'],missing_pct=100*s['missing']/n if n else None,
        infinite=s['invalid'],zero=s['zero'],zero_pct=100*s['zero']/n if n else None,
        negative=s['negative'],mean=s['mean'] if v else None,
        std=(s['M2']/(v-1))**.5 if v>1 else None,
        minimum=s['min'],maximum=s['max'],positive_count=s['positive'],
        positive_mean=s['positive_mean'] if s['positive'] else None,
        positive_std=(s['positive_M2']/(s['positive']-1))**.5 if s['positive']>1 else None,
        positive_min=s['positive_min'],positive_max=s['positive_max'])


def atomic_json(path,obj):
    temp=path.with_suffix('.tmp')
    with open(temp,'w') as f: json.dump(obj,f,allow_nan=False)
    os.replace(temp,path)


def process_file(path,batch_size):
    parquet=pq.ParquetFile(path)
    names=set(parquet.schema_arrow.names)
    cols=[c for c in SENSORS if c in names]
    has_time='time' in names
    read_cols=cols+(['time'] if has_time else [])
    stats={c:blank() for c in cols}; timestamp_stats={c:blank() for c in cols if c in GPS}
    # GPS/HR repeat across IMU rows; deduplicate consecutive equal timestamps.
    # A timestamp may span batch boundaries; retain the previous timestamp.
    prev_time=None
    time_missing=0; total_rows=0; repeated=0
    for batch in parquet.iter_batches(batch_size=batch_size,columns=read_cols):
        frame=batch.to_pandas()
        total_rows+=len(frame)
        for col in cols: update(stats[col],frame[col].to_numpy(),col)
        if has_time:
            times=frame['time'].astype('string')
            time_missing+=int(times.isna().sum())
            distinct=times.ne(times.shift()).fillna(True).to_numpy(dtype=bool)
            if len(times) and prev_time is not None and pd.notna(times.iloc[0]):
                distinct[0]=(times.iloc[0]!=prev_time)
            if len(times) and pd.notna(times.iloc[-1]): prev_time=times.iloc[-1]
            repeated+=int((~distinct).sum())
            for col in timestamp_stats:
                update(timestamp_stats[col],frame.loc[distinct,col].to_numpy(),col)
    return {'raw':stats,'timestamp':timestamp_stats,'time_missing':time_missing,'total_rows':total_rows,'repeated_timestamp_rows':repeated,'timestamp_method':'consecutive-first; assumes timestamps ordered within file'}


def aggregates(records,group_cols,mode):
    groups={}
    for rec in records:
        key=tuple(rec[k] for k in group_cols)
        target=groups.setdefault(key,{})
        for col,s in rec['stats'][mode].items():
            target[col]=combine(target[col],s) if col in target else s
    rows=[]
    for key,features in groups.items():
        for col,s in features.items():
            rows.append(dict(zip(group_cols,key),feature=col,measurement_mode=mode,**finish(s)))
    return pd.DataFrame(rows)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--manifest',default='data/processed/audit/objective_file_manifest.csv')
    ap.add_argument('--output',default='data/processed/audit/objective_statistics')
    ap.add_argument('--team',choices=['TeamA','TeamB'])
    ap.add_argument('--batch-size',type=int,default=100000)
    ap.add_argument('--limit',type=int,default=0,help='For initial smoke testing only')
    args=ap.parse_args()
    output=Path(args.output); checkpoint=output/'checkpoint'; checkpoint.mkdir(parents=True,exist_ok=True)
    manifest=pd.read_csv(args.manifest)
    required={'path','team','year','player_id'}
    if required-set(manifest.columns): raise ValueError('Manifest missing '+str(required-set(manifest.columns)))
    if args.team: manifest=manifest.loc[manifest.team==args.team]
    manifest=manifest.sort_values('path').drop_duplicates('path')
    if args.limit: manifest=manifest.head(args.limit)
    records=[]; failures=[]
    for i,row in enumerate(manifest.itertuples(index=False),1):
        path=Path(row.path)
        import hashlib
        identifier=hashlib.sha256(str(path.resolve()).encode()).hexdigest()
        cp=checkpoint/(identifier+'.json')
        try:
            if cp.exists():
                rec=json.loads(cp.read_text())
                if rec['path']!=str(path) or rec['size']!=path.stat().st_size or rec['mtime_ns']!=path.stat().st_mtime_ns:
                    raise ValueError('Source changed since checkpoint; remove its checkpoint to reprocess')
            else:
                rec=dict(path=str(path),size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns,
                         team=str(row.team),year=int(row.year),player_id=str(row.player_id),
                         stats=process_file(path,args.batch_size))
                atomic_json(cp,rec)
            records.append(rec)
        except Exception as exc:
            failures.append({'path':str(path),'error':repr(exc)})
        if i%50==0 or i==len(manifest): print(f'{i:,}/{len(manifest):,} files; completed={len(records):,}; failures={len(failures)}',flush=True)
    for mode in ['raw','timestamp']:
        for group,name in [([], 'global'),(['team'],'team'),(['year'],'year'),(['team','year'],'team_year'),(['team','player_id'],'player')]:
            frame=aggregates(records,group,mode)
            frame.to_csv(output/f'objective_{name}_{mode}_statistics.csv',index=False)
    # Per-file statistics for detailed inspection.
    rows=[]
    for rec in records:
        for mode in ['raw','timestamp']:
            for col,s in rec['stats'][mode].items():
                rows.append(dict(path=rec['path'],team=rec['team'],year=rec['year'],player_id=rec['player_id'],feature=col,measurement_mode=mode,**finish(s)))
    pd.DataFrame(rows).to_csv(output/'objective_file_statistics.csv',index=False)
    pd.DataFrame(failures,columns=['path','error']).to_csv(output/'objective_processing_errors.csv',index=False)
    summary=[f'Files selected: {len(manifest):,}',f'Files completed: {len(records):,}',f'Errors: {len(failures):,}',
             f'Rows read: {sum(r["stats"]["total_rows"] for r in records):,}',
             f'Rows with missing time: {sum(r["stats"]["time_missing"] for r in records):,}',
             'Nulls, infinities and zeros are counted separately.',
             'Timestamp statistics use first row of each consecutive timestamp group.',
             'Percentiles/outlier counts are NOT computed by this version; no approximate quantiles are claimed.',
             'Full-dataset coverage only when all selected files complete without errors and --limit is not used.']
    (output/'objective_audit20_summary.txt').write_text('\n'.join(summary)+'\n')
    print('\n'.join(summary))

if __name__=='__main__': main()

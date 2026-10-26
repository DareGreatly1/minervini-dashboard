import pandas as pd, yfinance as yf, numpy as np, warnings; warnings.filterwarnings('ignore')
tp=pd.read_csv('minervini_screen.csv'); tk=list(tp.ticker)
d=yf.download(tk,period='2y',auto_adjust=True,progress=False)
def zig(h,l,th=0.03):
  # alternating swing highs/lows starting from a high
  piv=[('H',0,h[0])]; 
  for i in range(1,len(h)):
    t,j,v=piv[-1]
    if t=='H':
      if h[i]>v: piv[-1]=('H',i,h[i])
      elif l[i]<v*(1-th): piv.append(('L',i,l[i]))
    else:
      if l[i]<v: piv[-1]=('L',i,l[i])
      elif h[i]>v*(1+th): piv.append(('H',i,h[i]))
  return piv
def fund(t):
  try:
    q=yf.Ticker(t).quarterly_income_stmt
    out={}
    for k,row in [('eps','Diluted EPS'),('rev','Total Revenue')]:
      if row not in q.index: out[k]=None;continue
      s=q.loc[row].sort_index().dropna()
      g=[]
      for dt in s.index[-2:]:
        prior=s[(s.index<dt-pd.Timedelta(days=330))&(s.index>dt-pd.Timedelta(days=400))]
        if len(prior) and prior.iloc[-1]!=0: g.append((s[dt]/prior.iloc[-1]-1)*100*(1 if prior.iloc[-1]>0 else -1))
        else: g.append(None)
      out[k]=g; out[k+'_q']=str(s.index[-1].date()) if len(s) else None
    return out
  except Exception as e: return {'eps':None,'rev':None,'err':str(e)}
rows=[]
for t in tk:
  H=d['High'][t].dropna();L=d['Low'][t].dropna();C=d['Close'][t].dropna();V=d['Volume'][t].dropna()
  n=len(C); base_end=n-3  # exclude last 3 days for base definition
  w=H.iloc[max(0,base_end-325):base_end]
  start=w.values.argmax()+max(0,base_end-325)
  blen=(base_end-start)/5
  if blen<3:  # just made highs; look further back for prior base: use highest high before a 3wk window
    w2=H.iloc[max(0,base_end-325):base_end-15]; start=w2.values.argmax()+max(0,base_end-325); blen=(base_end-start)/5
  piv=zig(H.values[start:base_end],L.values[start:base_end])
  cons=[];pivot=None
  for a,b in zip(piv,piv[1:]):
    if a[0]=='H' and b[0]=='L': cons.append(round((1-b[2]/a[2])*100,1)); pivot=a[2]
  if piv[-1][0]=='H': # final rising leg: current contraction from last high to subsequent low
    j=piv[-1][1]; lo=L.values[start+j:base_end].min(); c=(1-lo/piv[-1][2])*100
    if c>0.5: cons.append(round(c,1)); 
    pivot=piv[-1][2]
  # longest decreasing tail
  k=1
  while k<len(cons) and cons[-k-1]>cons[-k]: k+=1
  seq=cons[-min(k,6):]
  a_ok= 3<=blen<=65 and len(seq)>=2 and seq[-1]<=12
  v50=V.iloc[-53:-3].mean(); dry=V.iloc[-13:-3].mean()/v50
  b_ok=dry<1
  p=C.iloc[-1]; pfp=(p/pivot-1)*100
  bo=[i for i in range(n-3,n) if C.iloc[i]>pivot and C.iloc[i-1]<=pivot]
  bov=(V.iloc[bo[0]]/v50) if bo else None
  c_ok=(-5<=pfp<=0) or (bo and bov>=1.4 and pfp<=5)
  f=fund(t); e=f.get('eps') or [None]; r=f.get('rev') or [None]
  eg,rg=e[-1],r[-1]
  if eg is None or rg is None: d_ok=None
  else: d_ok= eg>=20 and rg>0
  rows.append(dict(ticker=t,price=round(p,2),base_weeks=round(blen,1),contractions='>'.join(f'{x:g}%' for x in seq),all_pullbacks=cons,pivot=round(pivot,2),pct_from_pivot=round(pfp,1),breakout_vol_ratio=None if bov is None else round(bov,2),vol_dryup_ratio=round(dry,2),eps_yoy=None if eg is None else round(eg,1),eps_yoy_prev_q=None if e[0] is None or len(e)<2 else round(e[0],1),rev_yoy=None if rg is None else round(rg,1),latest_q=f.get('eps_q'),a_vcp=a_ok,b_volume=b_ok,c_pivot=bool(c_ok),d_fund=d_ok,rs_rank=tp.set_index('ticker').rs_rank[t],sector=tp.set_index('ticker').sector[t]))
df=pd.DataFrame(rows)
cr=['a_vcp','b_volume','c_pivot','d_fund']
df['fails']=df.apply(lambda r:[c for c in cr if not (r[c] is not None and bool(r[c]))],axis=1)
df['n_fail']=df.fails.str.len()
df=df.sort_values(['n_fail','rs_rank'],ascending=[True,False])
df.to_csv('minervini_setups.csv',index=False)
pd.set_option('display.width',300);pd.set_option('display.max_columns',30);pd.set_option('display.max_rows',100)
print(df[df.n_fail<=1].drop(columns=['all_pullbacks','sector']).to_string(index=False))
print(df.n_fail.value_counts()); print(df[[*cr]].apply(lambda s:(s==True).sum()))
print(df[df.d_fund.isna()].ticker.tolist())

import pandas as pd, yfinance as yf, requests, io, numpy as np
H={'User-Agent':'Mozilla/5.0'}
def wt(u): return pd.read_html(io.StringIO(requests.get(u,headers=H).text))
sp=[t for t in wt('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies') if 'Symbol' in t.columns][0]
nd=[t for t in wt('https://en.wikipedia.org/wiki/List_of_NASDAQ-100_companies') if 'Ticker' in t.columns][0]
info={}
for _,r in sp.iterrows(): info[r.Symbol.replace('.','-')]={'name':r.Security,'idx':{'SP500'},'sector':r['GICS Sector']}
sc=[c for c in nd.columns if 'Industry' in c]
for _,r in nd.iterrows():
  t=r.Ticker.replace('.','-'); d=info.setdefault(t,{'name':r.get('Company',r.iloc[0]),'idx':set(),'sector':r[sc[0]] if sc else ''}); d['idx'].add('NDX100')
tk=sorted(info); print('universe',len(tk),len(sp),len(nd))
px=yf.download(tk,period='2y',auto_adjust=True,progress=False,threads=True)['Close']
px=px.dropna(how='all')
fail=[t for t in tk if t not in px or px[t].dropna().shape[0]<260]
print('asof',px.index[-1].date(),'fail',fail)
rows=[]
for t in tk:
  if t in fail: continue
  s=px[t].dropna(); p=s.iloc[-1]
  m50,m150,m200=[s.rolling(n).mean() for n in(50,150,200)]
  y=s.iloc[-252:]; hi,lo=y.max(),y.min()
  r=lambda a,b: s.iloc[-1-a]/s.iloc[-1-b]-1
  rs=.4*r(0,63)+.2*r(63,126)+.2*r(126,189)+.2*r(189,252)
  c=[p>m150.iloc[-1] and p>m200.iloc[-1],m150.iloc[-1]>m200.iloc[-1],m200.iloc[-1]>m200.iloc[-22],m50.iloc[-1]>m150.iloc[-1] and m50.iloc[-1]>m200.iloc[-1],p>m50.iloc[-1],p>=1.3*lo,p>=.75*hi]
  rows.append(dict(ticker=t,name=info[t]['name'],index='/'.join(sorted(info[t]['idx'])),price=round(p,2),pct_off_52w_high=round((1-p/hi)*100,1),pct_above_52w_low=round((p/lo-1)*100,1),rs_raw=rs,sector=info[t]['sector'],ok=all(c),n=sum(c)))
df=pd.DataFrame(rows); df['rs_rank']=(df.rs_raw.rank(pct=True)*99).round(0).astype(int)
df.to_csv('universe_rs.csv',index=False)
ps=df[df.ok&(df.rs_rank>=70)].sort_values('rs_rank',ascending=False)
ps['rs_12m_weighted_pct']=(ps.rs_raw*100).round(1)
ps[['ticker','name','index','price','pct_off_52w_high','pct_above_52w_low','rs_rank','rs_12m_weighted_pct','sector']].to_csv('minervini_screen.csv',index=False)
print('pass',len(ps),'tech-only(7)',df.ok.sum())
pd.set_option('display.width',250);pd.set_option('display.max_rows',500)
print(ps[['ticker','name','index','price','pct_off_52w_high','pct_above_52w_low','rs_rank','sector']].to_string(index=False))

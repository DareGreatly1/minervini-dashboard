#!/usr/bin/env python3
"""Daily Minervini pipeline: screen -> VCP scan -> score -> signals -> dashboard."""
import os, sys, json, subprocess, datetime as dt, html, shutil, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, yfinance as yf
D=os.path.dirname(os.path.abspath(__file__)); os.chdir(D)
INTRADAY=os.environ.get('INTRADAY')=='1'
STOP_PCT=0.08; PARTIAL=0.20; BOVOL=1.4
# A+ grading (added 2026-11-18 from backtest in backtest/): only A+ breakouts become BUYs, sized bigger.
APLUS_RISK=2.0; APLUS_CAP=0.35   # % of equity risked per A+ trade; max position as fraction of equity
LEGACY_RISK=1.0; LEGACY_CAP=0.25 # sizing shown for older (pre-A+) system positions
APLUS_RS=90; APLUS_RS_LINE=80; MAX_DIST_DAYS=5
def run(s):
    r=subprocess.run([sys.executable,s],capture_output=True,text=True)
    if r.returncode: print(r.stderr); sys.exit(f'{s} failed')
    return r.stdout
out1=run('scr.py'); out2=run('vcp.py')
tt=pd.read_csv('minervini_screen.csv'); su=pd.read_csv('minervini_setups.csv')
fail_dl=[l for l in out1.splitlines() if l.startswith('asof')]
# ---- score
def clip(x,a=0,b=1): return float(min(max(x,a),b)) if pd.notna(x) else 0.0
def score(r):
    fin=float(str(r.contractions).split('>')[-1].rstrip('%')) if isinstance(r.contractions,str) and r.contractions else 12
    s=10*(4-r.n_fail)                                  # 40 criteria
    s+=10*clip(r.base_weeks/30)                        # 10 base length (cap 30w)
    s+=10*clip((12-fin)/12)                            # 10 final contraction tightness
    s+=10*clip((1-r.vol_dryup_ratio)/0.4)              # 10 dry-up (0.6 or lower = full)
    s+=10*clip(1-abs(r.pct_from_pivot)/5)              # 10 proximity to pivot
    s+=10*clip((r.rs_rank-70)/29)                      # 10 RS
    s+=5*clip(r.eps_yoy/100) + 5*clip(r.rev_yoy/50)    # 10 EPS/rev growth (capped)
    return round(s,1)
su['score']=su.apply(score,axis=1); su=su.sort_values('score',ascending=False)
su.to_csv('minervini_setups_scored.csv',index=False)
full=su[su.n_fail==0]
# ---- state
SP='state.json'
st=json.load(open(SP)) if os.path.exists(SP) else {'positions':{},'history':[],'last_signals':{},'runs':[]}
pos=st['positions']
MP=json.load(open('positions.json')) if os.path.exists('positions.json') else {}
myopen={t:v for t,v in MP.items() if v.get('open',True)}
need=sorted(set(full.ticker)|set(pos)|set(su.head(10).ticker)|set(myopen))
px=yf.download(need,period='1y',auto_adjust=True,progress=False)
asof=str(px.index[-1].date())
def series(t): return px['Close'][t].dropna(),px['Volume'][t].dropna()
# ---- A+ grading inputs: market regime, RS line, earnings (never breaks the run)
MKT={'ok':None,'txt':'market data unavailable','spy':{},'qqq':{},'dist':None}
try:
    mk=yf.download(['SPY','QQQ'],period='2y',auto_adjust=True,progress=False)
    def _reg(sym):
        c=mk['Close'][sym].dropna(); a50=c.rolling(50).mean(); a200=c.rolling(200).mean()
        chk={'above 50d':bool(c.iloc[-1]>a50.iloc[-1]),'50d rising':bool(a50.iloc[-1]>a50.iloc[-11]),'above 200d':bool(c.iloc[-1]>a200.iloc[-1]),'200d rising':bool(a200.iloc[-1]>a200.iloc[-21])}
        return dict(px=round(float(c.iloc[-1]),2),m50=round(float(a50.iloc[-1]),2),m200=round(float(a200.iloc[-1]),2),chk=chk,ok=all(chk.values()))
    _sc,_sv=mk['Close']['SPY'].dropna(),mk['Volume']['SPY'].dropna()
    _dd=((_sc.pct_change()<=-0.002)&(_sv>_sv.shift(1))).astype(int)
    MKT['spy'],MKT['qqq'],MKT['dist']=_reg('SPY'),_reg('QQQ'),int(_dd.iloc[-25:].sum())
    MKT['ok']=MKT['spy']['ok'] and MKT['qqq']['ok'] and MKT['dist']<=MAX_DIST_DAYS
    _fl={'above 50d':'below 50d','50d rising':'50d falling','above 200d':'below 200d','200d rising':'200d falling'}
    bad=[f"{k} {_fl[n]}" for k,d in (('SPY',MKT['spy']),('QQQ',MKT['qqq'])) for n,v in d['chk'].items() if not v]
    if MKT['dist']>MAX_DIST_DAYS: bad.append(f"{MKT['dist']} distribution days in 25 sessions (max {MAX_DIST_DAYS})")
    MKT['txt']='uptrend: SPY & QQQ above rising 50d and 200d, %d distribution days'%MKT['dist'] if MKT['ok'] else 'NOT OK: '+'; '.join(bad)
    _spy=mk['Close']['SPY'].dropna()
except Exception as e:
    print('market regime failed:',e); _spy=None
def rsl_off(t):
    try:
        c,_=series(t); r=(c/_spy.reindex(c.index)).dropna().iloc[-252:]; return float(r.iloc[-1]/r.max()-1)
    except Exception: return None
from concurrent.futures import ThreadPoolExecutor
def _earn(t):
    try:
        e=yf.Ticker(t).get_earnings_dates(limit=12); ds=sorted({x.date() for x in e.index}) if e is not None else []
        return t,ds
    except Exception: return t,None
_today=pd.Timestamp(asof).date()
_cand=[r.ticker for r in full.itertuples() if -5<=r.pct_from_pivot<=5]
try:
    with ThreadPoolExecutor(6) as _ex: EARN=dict(_ex.map(_earn,_cand))
except Exception: EARN={}
def earn_info(t):
    ds=EARN.get(t)
    if not ds: return None,None
    prv=[d for d in ds if d<=_today]; nxt=[d for d in ds if d>_today]
    return ((_today-prv[-1]).days if prv else None),((nxt[0]-_today).days if nxt else None)
GR={}
def grade(r):
    t=r.ticker; ok=[]; miss=[]
    if MKT['ok']: ok.append('market uptrend')
    else: miss.append('market not in uptrend' if MKT['ok'] is False else 'market data unavailable')
    ro=rsl_off(t); rs=int(r.rs_rank); rl=ro is not None and ro>=-0.01
    if rs>=APLUS_RS: ok.append(f'RS {rs}')
    elif rs>=APLUS_RS_LINE and rl: ok.append(f'RS {rs} + RS line at new high')
    else: miss.append(f'leadership (RS {rs} < {APLUS_RS}' + ('' if rs<APLUS_RS_LINE else ', RS line %.1f%% below high'%(-ro*100 if ro is not None else 0)) + ')')
    se,ne=earn_info(t)
    if se is not None and se<=1: miss.append(f'earnings-day move (reported {se}d ago)')
    else: ok.append('not an earnings-day gap')
    info_e=(f'next earnings in {ne}d' if ne is not None else 'next earnings date unknown')
    g=dict(aplus=not miss,ok=ok,miss=miss,next_earn=ne,since_earn=se,earn_txt=info_e,rsl=ro)
    g['label']=('A+' if g['aplus'] else 'not A+')+(' — ' + ('; '.join(ok)) if g['aplus'] else ' — missing: '+'; '.join(miss))
    GR[t]=g; return g
sig={}; notes={}
# existing positions -> HOLD/SELL
ttset=set(tt.ticker)
for t,p in list(pos.items()):
    c,v=series(t); price=c.iloc[-1]; s50=c.rolling(50).mean().iloc[-1]; v50=v.iloc[-51:-1].mean()
    p['last']=round(float(price),2); gain=price/p['entry']-1; p['gain_pct']=round(gain*100,1)
    why=[]
    if price<p['stop']: why.append(f"close {price:.2f} < stop {p['stop']}")
    if price<s50 and v.iloc[-1]>=BOVOL*v50: why.append('closed below 50d SMA on heavy volume')
    if t not in ttset: why.append('failed Trend Template')
    if why:
        sig[t]='SELL'; notes[t]='; '.join(why); st['history'].append({'date':asof,'ticker':t,'signal':'SELL','price':p['last'],'entry':p['entry'],'reason':notes[t]}); del pos[t]
    else:
        sig[t]='HOLD'; notes[t]=f"{gain*100:+.1f}% vs entry {p['entry']}, stop {p['stop']}"+(' | TAKE PARTIAL PROFIT (>= +20%)' if gain>=PARTIAL else '')
# new BUY / WATCH from full setups
for _,r in full.iterrows():
    t=r.ticker
    if t in pos or t in sig: continue
    brk=pd.notna(r.breakout_vol_ratio) and r.breakout_vol_ratio>=BOVOL and 0<r.pct_from_pivot<=5
    if not (brk or -5<=r.pct_from_pivot<=0): continue
    g=grade(r)
    if brk and g['aplus']:
        sig[t]='BUY'; stop=round(r.pivot*(1-STOP_PCT),2)
        pos[t]={'entry':float(r.pivot),'stop':stop,'since':asof,'last':float(r.price),'gain_pct':round(r.pct_from_pivot,1),'grade':'A+'}
        notes[t]=f"A+ breakout {r.pct_from_pivot:+.1f}% past pivot {r.pivot} on {r.breakout_vol_ratio}x vol; stop {stop} | {'; '.join(g['ok'])}; {g['earn_txt']}"
        st['history'].append({'date':asof,'ticker':t,'signal':'BUY','grade':'A+','price':float(r.price),'entry':float(r.pivot),'reason':notes[t]})
    elif brk:
        sig[t]='WATCH'; notes[t]=f"broke out {r.pct_from_pivot:+.1f}% past pivot {r.pivot} on {r.breakout_vol_ratio}x vol but NOT A+, no buy. Missing: {'; '.join(g['miss'])}"
    else:
        sig[t]='WATCH'; notes[t]=f"{r.pct_from_pivot:+.1f}% below pivot {r.pivot}; buy only if A+ and close > pivot on vol >= {BOVOL}x 50d. "+(f"A+ ready ({'; '.join(g['ok'])})" if g['aplus'] else f"Missing for A+: {'; '.join(g['miss'])}")+f"; {g['earn_txt']}"
# ---- changes vs prev
prev=st.get('last_signals',{})
changes=[f"- {t}: {prev.get(t,'—')} → {sig.get(t,'—')}" for t in sorted(set(prev)|set(sig)) if prev.get(t)!=sig.get(t)]
st['last_signals']=sig; st['runs'].append({'run':dt.datetime.now().isoformat(timespec='minutes'),'asof':asof,'signals':sig})
if not INTRADAY: json.dump(st,open(SP,'w'),indent=1)
# ---- charts
def svg(t,pivot=None,stop=None,w=320,h=130):
    c,_=series(t); full_c=c
    s=[full_c.rolling(n).mean().iloc[-120:] for n in (50,150,200)]; c=c.iloc[-120:]
    vals=pd.concat([c,*s]).dropna().tolist()+[x for x in (pivot,stop) if x]
    lo,hi=min(vals),max(vals); X=lambda i:i*w/(len(c)-1); Y=lambda y:h-(y-lo)/(hi-lo)*h
    def pl(ser,col,sw=1):
        pts=' '.join(f'{X(i):.1f},{Y(y):.1f}' for i,y in enumerate(ser.values) if pd.notna(y))
        return f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="{pts}"/>'
    g=pl(c,'#111',1.5)+pl(s[0],'#1f77b4')+pl(s[1],'#ff7f0e')+pl(s[2],'#d62728')
    if pivot: g+=f'<line x1="0" x2="{w}" y1="{Y(pivot):.1f}" y2="{Y(pivot):.1f}" stroke="green" stroke-dasharray="4"/>'
    if stop: g+=f'<line x1="0" x2="{w}" y1="{Y(stop):.1f}" y2="{Y(stop):.1f}" stroke="crimson" stroke-dasharray="2"/>'
    return f'<svg width="{w}" height="{h}" style="background:#fafafa;border:1px solid #ddd">{g}</svg>'
uni=pd.read_csv('universe_rs.csv').set_index('ticker')
px2=yf.download(list(myopen),period='2y',auto_adjust=True,progress=False) if myopen else None
mycards=[]; mysum=[]
for t,p in myopen.items():
    c=px2['Close'][t].dropna() if len(myopen)>1 else px2['Close'].squeeze().dropna()
    v=px2['Volume'][t].dropna() if len(myopen)>1 else px2['Volume'].squeeze().dropna()
    price=float(c.iloc[-1]); m50,m150,m200=[float(c.rolling(n).mean().iloc[-1]) for n in (50,150,200)]
    m200p=float(c.rolling(200).mean().iloc[-22]); y=c.iloc[-252:]; hi,lo=float(y.max()),float(y.min())
    v50=float(v.iloc[-51:-1].mean()); vr=float(v.iloc[-1])/v50
    rs=int(uni.rs_rank[t]) if t in uni.index else None
    tt_rules={'price>150&200d':price>m150 and price>m200,'150d>200d':m150>m200,'200d rising 1mo':m200>m200p,'50d>150&200d':m50>m150 and m50>m200,'price>50d':price>m50,'>=30% above 52w low':price>=1.3*lo,'within 25% of 52w high':price>=0.75*hi,'RS>=70':(rs is not None and rs>=70)}
    failed=[k for k,ok in tt_rules.items() if not ok]
    pl=(price-p['entry'])*p['shares']; plp=(price/p['entry']-1)*100
    status='OK'; why=[]
    if price<=p['stop']: status='SELL'; why.append(f"close {price:.2f} <= stop {p['stop']}")
    if price<m50 and vr>=BOVOL: status='SELL'; why.append(f'below 50d SMA {m50:.2f} on {vr:.2f}x volume')
    if failed:
        why.append('WARNING fails Trend Template: '+', '.join(failed))
        if status=='OK': status='WARNING'
    if plp>=20: why.append('TAKE PARTIAL PROFIT (+20-25%)')
    elif plp>=10: why.append(f"up >=10%: move stop to breakeven {p['entry']}")
    if price<m50 and vr<BOVOL: why.append(f'below 50d SMA {m50:.2f} (volume {vr:.2f}x, not heavy)')
    p.update(last=round(price,2),last_status=status,last_run=asof)
    d=dict(t=t,shares=p['shares'],entry=p['entry'],price=round(price,2),pl=round(pl,2),plp=round(plp,2),stop=p['stop'],to_stop=round((price/p['stop']-1)*100,1),sma50=round(m50,2),volr=round(vr,2),rs=rs,status=status,why='; '.join(why) or 'within rules')
    mysum.append(d)
if not INTRADAY: json.dump(MP,open('positions.json','w'),indent=1)
# ---- account & sizing (never breaks the run)
try: AC=json.load(open('account.json'))
except Exception: AC=None
openval=sum(d['shares']*d['price'] for d in mysum)
if AC:
    cash=float(AC.get('cash',0)); start=float(AC.get('starting_balance',15000)); rp=float(AC.get('risk_pct',1.0)); eq=cash+openval
def size(t):
    try:
        if not AC: return None
        p=pos.get(t,{}); r=info.loc[t] if t in info.index else None
        e=p.get('entry', float(r.pivot) if r is not None else None)
        if not e: return None
        sp=p.get('stop', round(e*(1-STOP_PCT),2)); ps=e-sp
        if ps<=0: return None
        ap=(p.get('grade')=='A+') if p else GR.get(t,{}).get('aplus',False)
        if not ap and not p: return dict(no=True)
        rpx,cap=(APLUS_RISK,APLUS_CAP) if ap else (LEGACY_RISK,LEGACY_CAP)
        rd=eq*rpx/100; n=int(rd//ps); n=min(n,int(cash//e),int(cap*eq//e))
        return dict(sh=max(n,0),cost=round(max(n,0)*e,2),risk=round(max(n,0)*ps,2),entry=round(e,2),stop=sp,rp=rpx,cap=cap)
    except Exception: return None
def sztxt(t,k):
    z=size(t)
    if not z: return ''
    if z.get('no'): return 'Not A+: no buy, even on a breakout'
    return f"{'If it breaks out (A+ sizing): ' if k=='WATCH' else ''}buy {z['sh']} sh ≈ ${z['cost']:,.0f} @ {z['entry']}, risk ${z['risk']:,.0f} = {z['rp']:g}% of equity (stop {z['stop']})"
if AC:
    pl=eq-start
    acsec=f'<h2>Account</h2><table><tr><th>starting balance</th><th>cash</th><th>open positions</th><th>total equity</th><th>P&amp;L vs start</th><th>risk/trade</th></tr><tr><td>${start:,.2f}</td><td>${cash:,.2f}</td><td>${openval:,.2f}</td><td><b>${eq:,.2f}</b></td><td>{pl:+,.2f} ({pl/start*100:+.2f}%)</td><td>A+: {APLUS_RISK:g}% = ${eq*APLUS_RISK/100:,.2f}</td></tr></table><p class="cav">Cash as of {html.escape(str(AC.get("as_of","")))} (update account.json after trades). Suggested shares for <b>A+</b> buys = floor({APLUS_RISK:g}% of equity / (entry−stop)), capped at available cash and {APLUS_CAP*100:.0f}% of equity. Older non-A+ system positions keep {LEGACY_RISK:g}% / {LEGACY_CAP*100:.0f}%.</p>'
else: acsec='<h2>Account</h2><p><i>account.json missing</i></p>'
info=su.set_index('ticker'); names=tt.set_index('ticker')['name'] if 'name' in tt else {}
def gbadge(t):
    g=GR.get(t); gp=pos.get(t,{}).get('grade')
    if gp=='A+' or (g and g['aplus']): return '<span style="background:#2ca02c;color:#fff;padding:1px 6px;border-radius:4px;font-weight:bold">'+('A+' if gp=='A+' else 'A+ ready')+'</span> '
    if g: return '<span style="background:#999;color:#fff;padding:1px 6px;border-radius:4px">not A+</span> <small style="color:#b45309">missing: '+html.escape('; '.join(g['miss']))+'</small><br>'
    if t in pos: return '<span style="background:#bbb;color:#fff;padding:1px 6px;border-radius:4px">pre-A+ signal</span> '
    return ''
def card(t,kind):
    r=info.loc[t] if t in info.index else None; p=pos.get(t,{})
    pv=p.get('entry', r.pivot if r is not None else None); sp=p.get('stop', round(pv*(1-STOP_PCT),2) if pv else None)
    nm=html.escape(str(names.get(t,''))) if t in names else ''
    sc=f" · score {r.score}" if r is not None else ''
    return f'<div class="card {kind}"><b>{t}</b> {nm}{sc}<br>{gbadge(t)}<small>{html.escape(notes.get(t,""))}</small><br><b style="font-size:12px">{html.escape(sztxt(t,kind)) if kind in ('BUY','WATCH') else ''}</b><br>{svg(t,pv,sp)}</div>'

# ---- BUY (new): tracked positions whose BUY came on the last completed session before today (or today)
def _newset():
    today=(dt.datetime.utcnow()-dt.timedelta(hours=7)).date()
    done=[d for d in px.index.date if d<today]
    if not done: return set(),None
    lc=str(max(done))
    return {t for t,p in pos.items() if p.get('since','')>=lc and sig.get(t) in ('BUY','HOLD')},lc
newset,lastcomp=_newset()
def newcard(t):
    p=pos[t]; c,_v=series(t); price=float(c.iloc[-1]); e=p['entry']; sd=p.get('since','')
    h=[x for x in st['history'] if x['ticker']==t and x['signal']=='BUY' and x['date']==sd]
    if h: sigpx=h[-1]['price']
    else:
        try: sigpx=float(c[c.index.date==dt.date.fromisoformat(sd)].iloc[-1])
        except Exception: sigpx=None
    past=(price/e-1)*100; chase=e*1.05; ext=past>5
    txt=''
    if AC:
        ap=p.get('grade')=='A+'; rpx,cap=(APLUS_RISK,APLUS_CAP) if ap else (LEGACY_RISK,LEGACY_CAP)
        spx=p['stop']; ps=price-spx; n=int(eq*rpx/100//ps) if ps>0 else 0; n=max(min(n,int(cash//price),int(cap*eq//price)),0)
        txt=f"buy {n} sh ≈ ${n*price:,.0f} @ {price:.2f} now, stop {spx:.2f} (system stop, pivot −8%), risk ${n*max(ps,0):,.0f} ({rpx:g}% of equity, max {cap*100:.0f}% position)"
    lab='<b style="color:#d62728">EXTENDED, DON\'T CHASE</b> — ' if ext else '<b style="color:#2ca02c">Within buy range</b> — '
    r=info.loc[t] if t in info.index else None
    nm=html.escape(str(names.get(t,''))) if t in names else ''
    sc=f" · score {r.score}" if r is not None else ''
    rows=(f'<table><tr><th>signal date</th><td>{sd}</td></tr><tr><th>signal close</th><td>{"%.2f"%sigpx if sigpx else "n/a"}</td></tr><tr><th>pivot</th><td>{e:.2f}</td></tr>'
      f'<tr><th>current price</th><td><b>{price:.2f}</b></td></tr><tr><th>past pivot</th><td>{past:+.1f}%</td></tr><tr><th>don\'t chase above (pivot×1.05)</th><td>{chase:.2f}</td></tr>'
      f'<tr><th>system stop</th><td>{p["stop"]:.2f}</td></tr></table>')
    return f'<div class="card BUY"><b>{t}</b> {nm}{sc}<br>{gbadge(t)}<small>{html.escape(notes.get(t,""))}</small><br>{lab}{"price is %.1f%% past pivot"%past}<br>{rows}<b style="font-size:12px">{html.escape(txt)}</b><br>{svg(t,e,p["stop"])}</div>'
def newsection():
    ts=sorted(newset,key=lambda t:-info.score.get(t,0))
    body=''.join(newcard(t) for t in ts) or '<p><i>None</i></p>'
    return f'<h2>BUY (new) ({len(ts)}) <small style="font-weight:normal">— BUY signal on {lastcomp or "n/a"} close or later; shown through the next trading day, then moves to HOLD</small></h2><div class="grid">{body}</div>'
def section(k,title):
    ts=[t for t,s in sig.items() if s==k and t not in newset]; ts.sort(key=lambda t:-info.score.get(t,0))
    body=''.join(card(t,k) for t in ts) or '<p><i>None</i></p>'
    return f'<h2>{title} ({len(ts)})</h2><div class="grid">{body}</div>'
cols=['ticker','score','n_fail','fails','rs_rank','price','pivot','pct_from_pivot','contractions','base_weeks','vol_dryup_ratio','breakout_vol_ratio','eps_yoy','rev_yoy','sector']
tbl=su[cols].copy(); tbl.insert(1,'signal',tbl.ticker.map(sig).fillna('')); tbl.insert(2,'grade',tbl.ticker.map(lambda t:('A+' if GR[t]['aplus'] else 'missing: '+'; '.join(GR[t]['miss'])) if t in GR else ''))
thead=''.join(f'<th onclick="srt({i})">{c}</th>' for i,c in enumerate(tbl.columns))
rows=''.join('<tr>'+''.join(f'<td>{html.escape("" if pd.isna(v) else str(v))}</td>' for v in r)+'</tr>' for r in tbl.itertuples(index=False))
def mysvg(t,entry,stop):
    c=(px2['Close'][t] if len(myopen)>1 else px2['Close'].squeeze()).dropna(); w,h=320,130
    ss=[c.rolling(n).mean().iloc[-120:] for n in (50,150,200)]; c=c.iloc[-120:]
    vals=pd.concat([c,*ss]).dropna().tolist()+[entry,stop]; lo,hi=min(vals),max(vals)
    X=lambda i:i*w/(len(c)-1); Y=lambda y:h-(y-lo)/(hi-lo)*h
    pl=lambda ser,col,sw=1:f'<polyline fill="none" stroke="{col}" stroke-width="{sw}" points="'+' '.join(f'{X(i):.1f},{Y(y):.1f}' for i,y in enumerate(ser.values) if pd.notna(y))+'"/>'
    g=pl(c,'#111',1.5)+pl(ss[0],'#1f77b4')+pl(ss[1],'#ff7f0e')+pl(ss[2],'#d62728')
    g+=f'<line x1="0" x2="{w}" y1="{Y(entry):.1f}" y2="{Y(entry):.1f}" stroke="green" stroke-dasharray="4"/><line x1="0" x2="{w}" y1="{Y(stop):.1f}" y2="{Y(stop):.1f}" stroke="crimson" stroke-dasharray="2"/>'
    return f'<svg width="{w}" height="{h}" style="background:#fafafa;border:1px solid #ddd">{g}</svg>'
col={'OK':'#2ca02c','WARNING':'#e6b800','SELL':'#d62728'}
mysec='<h2>My Positions (%d) <small style="font-weight:normal">— prices as of %s, intraday if run during market hours</small></h2><div class="grid">'%(len(mysum),asof)+''.join(
 f'<div class="card" style="border-left-color:{col[d["status"]]};width:auto"><b>{d["t"]}</b> {html.escape(myopen[d["t"]].get("name",""))} — <b style="color:{col[d["status"]]}">{d["status"]}</b><br>'
 f'<table><tr><th>shares</th><th>entry</th><th>price</th><th>P/L $</th><th>P/L %</th><th>stop</th><th>to stop</th><th>50d SMA</th><th>vol/50d</th><th>RS</th></tr>'
 f'<tr><td>{d["shares"]}</td><td>{d["entry"]}</td><td>{d["price"]}</td><td>{d["pl"]:+.2f}</td><td>{d["plp"]:+.2f}%</td><td>{d["stop"]}</td><td>{d["to_stop"]}%</td><td>{d["sma50"]}</td><td>{d["volr"]}</td><td>{d["rs"]}</td></tr></table>'
 f'<small>{html.escape(d["why"])}</small><br>{mysvg(d["t"],d["entry"],d["stop"])}</div>' for d in mysum)+('</div>' if mysum else '<p><i>None</i></p></div>')
now_mst=dt.datetime.utcnow()-dt.timedelta(hours=7)
_mc,_mb=('#e8f5e9','#2ca02c') if MKT['ok'] else ('#fdecea','#d62728')
def _mrow(n,d): return f"<td>{n}</td><td>{d.get('px','')}</td><td>{d.get('m50','')}</td><td>{d.get('m200','')}</td><td>{', '.join({'above 50d':'below 50d','50d rising':'50d falling','above 200d':'below 200d','200d rising':'200d falling'}[k] for k,v in d.get('chk',{}).items() if not v) or 'all OK'}</td>"
mktbanner=(f'<div style="background:{_mc};border:2px solid {_mb};padding:8px 12px;border-radius:6px;margin:8px 0"><b style="font-size:17px;color:{_mb}">'
  +('Market OK: A+ breakouts can be bought' if MKT['ok'] else 'Market NOT OK: take no new buys')+f'</b><br><small>{html.escape(MKT["txt"])}</small>'
  +(f'<table style="margin-top:4px"><tr><th>index</th><th>close</th><th>50d</th><th>200d</th><th>problems</th></tr><tr>{_mrow("SPY",MKT["spy"])}</tr><tr>{_mrow("QQQ",MKT["qqq"])}</tr></table><small>Distribution days (SPY down ≥0.2% on higher volume) in last 25 sessions: <b>{MKT["dist"]}</b> (max {MAX_DIST_DAYS})</small>' if MKT['spy'] else '')+'</div>')
banner=mktbanner+(f'<p style="background:#fff3cd;border:1px solid #e6b800;padding:8px;border-radius:6px"><b>Intraday refresh as of {now_mst:%H:%M} MST, provisional until close.</b> Prices are live; signals are not recorded and the daily close run is authoritative.</p>' if INTRADAY else '')
page=f'''<!doctype html><html><head><meta charset="utf-8"><title>Minervini Dashboard {asof}</title><style>
body{{font-family:system-ui,sans-serif;margin:20px;color:#222}}.grid{{display:flex;flex-wrap:wrap;gap:10px}}
.card{{border:1px solid #ccc;border-radius:6px;padding:8px;width:330px;border-left:6px solid #999}}
.BUY{{border-left-color:#2ca02c}}.WATCH{{border-left-color:#e6b800}}.HOLD{{border-left-color:#1f77b4}}.SELL{{border-left-color:#d62728}}
table{{border-collapse:collapse;font-size:12px}}td,th{{border:1px solid #ddd;padding:3px 6px}}th{{cursor:pointer;background:#eee}}
.legend span{{margin-right:12px}}.cav{{font-size:12px;color:#555}}</style></head><body>
<div style="display:flex;align-items:center;gap:16px;flex-wrap:wrap"><h1 style="margin:0">Minervini Daily Dashboard</h1>
<a href="https://github.com/DareGreatly1/minervini-dashboard/actions/workflows/refresh.yml" target="_blank" rel="noopener" style="background:#2563eb;color:#fff;padding:10px 18px;border-radius:6px;font-weight:bold;text-decoration:none;font-size:16px">&#x21bb; Refresh now</a>
<span class="cav">Click 'Run workflow' on GitHub; page updates in ~3-5 min.<br>Last updated: <b>{now_mst:%Y-%m-%d %H:%M} MST</b></span></div>
{banner}<p>Data as of <b>{asof}</b> (yfinance daily; intraday if run during market hours) · generated {now_mst:%Y-%m-%d %H:%M} MST ·
universe S&amp;P 500 + NASDAQ 100 · Trend Template passers: {len(tt)} · full setups: {len(full)}</p>
<p class="legend"><span style="color:#111">■ close</span><span style="color:#1f77b4">■ 50d</span><span style="color:#ff7f0e">■ 150d</span><span style="color:#d62728">■ 200d</span><span style="color:green">-- pivot/entry</span><span style="color:crimson">-- stop (-8%)</span> (last 120 sessions)</p>
{newsection()}{acsec}{mysec}{section('WATCH','WATCH (full setup, ≤5% below pivot)')}{section('HOLD','HOLD')}{section('SELL','SELL')}
<h2>Cleanest setups (all {len(su)} Trend Template passers, click header to sort)</h2><table id="t"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table>
<h2>Rules &amp; caveats</h2><div class="cav"><p>Score (0-100): 10/criterion passed (VCP, volume dry-up, pivot, fundamentals) + base length (≤30w) 10 + final contraction tightness 10 + dry-up 10 + pivot proximity 10 + RS 70→99 10 + EPS YoY (cap 100%) 5 + revenue YoY (cap 50%) 5.</p>
<p><b>A+ grade (since 2026-11-18, from a 3.7-year backtest):</b> a breakout is only a BUY if all of these hold: (1) market uptrend: SPY and QQQ both above a rising 50-day and rising 200-day, with no more than {MAX_DIST_DAYS} distribution days in the last 25 sessions; (2) leadership: RS rank ≥{APLUS_RS}, or RS ≥{APLUS_RS_LINE} with the RS line (stock ÷ SPY) at a 52-week high; (3) not an earnings-day gap (no earnings report today or yesterday). Breakouts that miss any of these stay on WATCH, labeled with what's missing. A+ sizing: {APLUS_RISK:g}% of equity at risk, max {APLUS_CAP*100:.0f}% of equity per position.</p>
<p>BUY: A+ grade + full setup, close &gt; pivot within last 3 sessions on ≥1.4× 50d volume, ≤5% past pivot; entry = pivot, stop = pivot −8%. WATCH: full setup 0–5% below pivot. HOLD: tracked position above stop and not below 50d SMA on heavy volume, still in Trend Template. SELL: close &lt; stop, or close &lt; 50d SMA on ≥1.4× volume, or fails Trend Template. Partial-profit flag at +20%.</p>
<p>VCP detection is algorithmic (3% zigzag swings) and is not a substitute for reviewing charts. EPS acceleration not checked (yfinance gives ~5 quarters). Large EPS % can come from depressed prior-year bases. RS rank is relative to this universe only. Not investment advice.</p>
<p>{html.escape(' '.join(fail_dl))}</p></div>
<script>function srt(c){{const tb=document.querySelector('#t tbody');const r=[...tb.rows];const d=tb.dataset.c==c&&tb.dataset.d!='-1'?-1:1;tb.dataset.c=c;tb.dataset.d=d;
r.sort((a,b)=>{{let x=a.cells[c].innerText,y=b.cells[c].innerText,nx=parseFloat(x),ny=parseFloat(y);return(isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)*d}});r.forEach(e=>tb.appendChild(e))}}</script></body></html>'''
open('dashboard.html','w').write(page)
if not INTRADAY: shutil.copy('dashboard.html',f'archive/dashboard_{asof}.html')
md=[f'# Minervini summary — data as of {asof}','',('MARKET OK for A+ buys: ' if MKT['ok'] else 'MARKET NOT OK, no new buys: ')+MKT['txt'],'',(f'Account: cash ${cash:,.2f}, open ${openval:,.2f}, equity ${eq:,.2f}, P&L {eq-start:+,.2f} ({(eq-start)/start*100:+.2f}%)' if AC else 'Account: account.json missing'),'','## My positions']+[f"- {d['t']}: {d['status']} — {d['shares']} sh @ {d['entry']}, now {d['price']}, P/L {d['pl']:+.2f} ({d['plp']:+.2f}%), stop {d['stop']} ({d['to_stop']}% away); {d['why']}" for d in mysum]+['',f'Trend Template passers: {len(tt)} · full setups: {len(full)}','']
for k in ('BUY','WATCH','HOLD','SELL'):
    ts=[t for t,s in sig.items() if s==k]; md.append(f'## {k} ({len(ts)})'); md+= [f'- {t}: {notes[t]}'+(f' | {sztxt(t,k)}' if k in ('BUY','WATCH') and sztxt(t,k) else '') for t in ts] or ['- none']; md.append('')
md+=['## Changes vs previous run']+(changes or ['- none'])+['','Top 10 cleanest: '+', '.join(f'{r.ticker} ({r.score})' for r in su.head(10).itertuples())]
if not INTRADAY: open('latest_summary.md','w').write('\n'.join(md)); print('\n'.join(md))
# ---- publish (never breaks the run)
if INTRADAY: sys.exit(0)
try:
    r=subprocess.run(['bash',os.path.join(D,'publish.sh')],capture_output=True,text=True,timeout=300); print(r.stdout.strip() or r.stderr.strip())
except Exception as e: print('publish failed:',e)

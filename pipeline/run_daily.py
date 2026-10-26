#!/usr/bin/env python3
"""Daily Minervini pipeline: screen -> VCP scan -> score -> signals -> dashboard."""
import os, sys, json, subprocess, datetime as dt, html, shutil, warnings
warnings.filterwarnings('ignore')
import pandas as pd, numpy as np, yfinance as yf
D=os.path.dirname(os.path.abspath(__file__)); os.chdir(D)
INTRADAY=os.environ.get('INTRADAY')=='1'
STOP_PCT=0.08; PARTIAL=0.20; BOVOL=1.4
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
    if pd.notna(r.breakout_vol_ratio) and r.breakout_vol_ratio>=BOVOL and 0<r.pct_from_pivot<=5:
        sig[t]='BUY'; stop=round(r.pivot*(1-STOP_PCT),2)
        pos[t]={'entry':float(r.pivot),'stop':stop,'since':asof,'last':float(r.price),'gain_pct':round(r.pct_from_pivot,1)}
        notes[t]=f"breakout {r.pct_from_pivot:+.1f}% past pivot {r.pivot} on {r.breakout_vol_ratio}x vol; stop {stop}"
        st['history'].append({'date':asof,'ticker':t,'signal':'BUY','price':float(r.price),'entry':float(r.pivot),'reason':notes[t]})
    elif -5<=r.pct_from_pivot<=0:
        sig[t]='WATCH'; notes[t]=f"{r.pct_from_pivot:+.1f}% below pivot {r.pivot}; buy on close > pivot with vol >= {BOVOL}x 50d"
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
        rd=eq*rp/100; n=int(rd//ps); n=min(n,int(cash//e),int(0.25*eq//e))
        return dict(sh=max(n,0),cost=round(max(n,0)*e,2),risk=round(max(n,0)*ps,2),entry=round(e,2),stop=sp)
    except Exception: return None
def sztxt(t,k):
    z=size(t)
    if not z: return ''
    return f"{'If it breaks out: ' if k=='WATCH' else ''}buy {z['sh']} sh ≈ ${z['cost']:,.0f} @ {z['entry']}, risk ${z['risk']:,.0f} (stop {z['stop']})"
if AC:
    pl=eq-start
    acsec=f'<h2>Account</h2><table><tr><th>starting balance</th><th>cash</th><th>open positions</th><th>total equity</th><th>P&amp;L vs start</th><th>risk/trade</th></tr><tr><td>${start:,.2f}</td><td>${cash:,.2f}</td><td>${openval:,.2f}</td><td><b>${eq:,.2f}</b></td><td>{pl:+,.2f} ({pl/start*100:+.2f}%)</td><td>{rp}% = ${eq*rp/100:,.2f}</td></tr></table><p class="cav">Cash as of {html.escape(str(AC.get("as_of","")))} (update account.json after trades). Suggested shares = floor(risk$ / (entry−stop)), capped at available cash and 25% of equity.</p>'
else: acsec='<h2>Account</h2><p><i>account.json missing</i></p>'
info=su.set_index('ticker'); names=tt.set_index('ticker')['name'] if 'name' in tt else {}
def card(t,kind):
    r=info.loc[t] if t in info.index else None; p=pos.get(t,{})
    pv=p.get('entry', r.pivot if r is not None else None); sp=p.get('stop', round(pv*(1-STOP_PCT),2) if pv else None)
    nm=html.escape(str(names.get(t,''))) if t in names else ''
    sc=f" · score {r.score}" if r is not None else ''
    return f'<div class="card {kind}"><b>{t}</b> {nm}{sc}<br><small>{html.escape(notes.get(t,""))}</small><br><b style="font-size:12px">{html.escape(sztxt(t,kind))}</b><br>{svg(t,pv,sp)}</div>'
def section(k,title):
    ts=[t for t,s in sig.items() if s==k]; ts.sort(key=lambda t:-info.score.get(t,0))
    body=''.join(card(t,k) for t in ts) or '<p><i>None</i></p>'
    return f'<h2>{title} ({len(ts)})</h2><div class="grid">{body}</div>'
cols=['ticker','score','n_fail','fails','rs_rank','price','pivot','pct_from_pivot','contractions','base_weeks','vol_dryup_ratio','breakout_vol_ratio','eps_yoy','rev_yoy','sector']
tbl=su[cols].copy(); tbl.insert(1,'signal',tbl.ticker.map(sig).fillna(''))
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
banner=(f'<p style="background:#fff3cd;border:1px solid #e6b800;padding:8px;border-radius:6px"><b>Intraday refresh as of {now_mst:%H:%M} MST, provisional until close.</b> Prices are live; signals are not recorded and the daily close run is authoritative.</p>' if INTRADAY else '')
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
{acsec}{mysec}{section('BUY',"Today's BUY")}{section('WATCH','WATCH (full setup, ≤5% below pivot)')}{section('HOLD','HOLD')}{section('SELL','SELL')}
<h2>Cleanest setups (all {len(su)} Trend Template passers, click header to sort)</h2><table id="t"><thead><tr>{thead}</tr></thead><tbody>{rows}</tbody></table>
<h2>Rules &amp; caveats</h2><div class="cav"><p>Score (0-100): 10/criterion passed (VCP, volume dry-up, pivot, fundamentals) + base length (≤30w) 10 + final contraction tightness 10 + dry-up 10 + pivot proximity 10 + RS 70→99 10 + EPS YoY (cap 100%) 5 + revenue YoY (cap 50%) 5.</p>
<p>BUY: full setup, close &gt; pivot within last 3 sessions on ≥1.4× 50d volume, ≤5% past pivot; entry = pivot, stop = pivot −8%. WATCH: full setup 0–5% below pivot. HOLD: tracked position above stop and not below 50d SMA on heavy volume, still in Trend Template. SELL: close &lt; stop, or close &lt; 50d SMA on ≥1.4× volume, or fails Trend Template. Partial-profit flag at +20%.</p>
<p>VCP detection is algorithmic (3% zigzag swings) and is not a substitute for reviewing charts. EPS acceleration not checked (yfinance gives ~5 quarters). Large EPS % can come from depressed prior-year bases. RS rank is relative to this universe only. Not investment advice.</p>
<p>{html.escape(' '.join(fail_dl))}</p></div>
<script>function srt(c){{const tb=document.querySelector('#t tbody');const r=[...tb.rows];const d=tb.dataset.c==c&&tb.dataset.d!='-1'?-1:1;tb.dataset.c=c;tb.dataset.d=d;
r.sort((a,b)=>{{let x=a.cells[c].innerText,y=b.cells[c].innerText,nx=parseFloat(x),ny=parseFloat(y);return(isNaN(nx)||isNaN(ny)?x.localeCompare(y):nx-ny)*d}});r.forEach(e=>tb.appendChild(e))}}</script></body></html>'''
open('dashboard.html','w').write(page)
if not INTRADAY: shutil.copy('dashboard.html',f'archive/dashboard_{asof}.html')
md=[f'# Minervini summary — data as of {asof}','',(f'Account: cash ${cash:,.2f}, open ${openval:,.2f}, equity ${eq:,.2f}, P&L {eq-start:+,.2f} ({(eq-start)/start*100:+.2f}%)' if AC else 'Account: account.json missing'),'','## My positions']+[f"- {d['t']}: {d['status']} — {d['shares']} sh @ {d['entry']}, now {d['price']}, P/L {d['pl']:+.2f} ({d['plp']:+.2f}%), stop {d['stop']} ({d['to_stop']}% away); {d['why']}" for d in mysum]+['',f'Trend Template passers: {len(tt)} · full setups: {len(full)}','']
for k in ('BUY','WATCH','HOLD','SELL'):
    ts=[t for t,s in sig.items() if s==k]; md.append(f'## {k} ({len(ts)})'); md+= [f'- {t}: {notes[t]}'+(f' | {sztxt(t,k)}' if k in ('BUY','WATCH') and sztxt(t,k) else '') for t in ts] or ['- none']; md.append('')
md+=['## Changes vs previous run']+(changes or ['- none'])+['','Top 10 cleanest: '+', '.join(f'{r.ticker} ({r.score})' for r in su.head(10).itertuples())]
if not INTRADAY: open('latest_summary.md','w').write('\n'.join(md)); print('\n'.join(md))
# ---- publish (never breaks the run)
if INTRADAY: sys.exit(0)
try:
    r=subprocess.run(['bash',os.path.join(D,'publish.sh')],capture_output=True,text=True,timeout=300); print(r.stdout.strip() or r.stderr.strip())
except Exception as e: print('publish failed:',e)

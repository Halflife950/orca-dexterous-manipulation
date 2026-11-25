"""Plot the prespecified 25-second endpoints on the fresh Z evaluation bank."""
from pathlib import Path
import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
with (ROOT/'evidence/z_trials.csv').open() as stream:
    rows=[r for r in csv.DictReader(stream) if r['cohort']=='fresh' and r['horizon']=='25']
fields=['retained','supported45','repeated','joint']
labels=['Retained\nat 25 s','Supported\n45°','Repeated\nexchange','All three']
fig,(a,b)=plt.subplots(1,2,figsize=(9,3.55),gridspec_kw={'width_ratios':[1.7,1]},layout='constrained')
colors=['#728493','#08786f']
for j,(arm,legend) in enumerate([('baseline','Baseline'),('slip','Outward-slip cost')]):
    group=[r for r in rows if r['arm']==arm]
    values=[sum(int(r[f]) for r in group)/len(group)*100 for f in fields]
    bars=a.bar(np.arange(4)+(j-.5)*.38,values,.37,color=colors[j],label=legend)
    a.bar_label(bars,fmt='%.0f',padding=2,fontsize=8)
a.set_xticks(range(4),labels)
a.set_title('A   Fresh starts, all three seeds',loc='left',fontsize=12)
a.legend(frameon=False,fontsize=8,loc='upper right')
for seed in sorted({r['seed'] for r in rows}):
    values=[]
    for arm in ('baseline','slip'):
        group=[r for r in rows if r['arm']==arm and r['seed']==seed]
        values.append(sum(int(r['joint']) for r in group)/len(group)*100)
    b.plot([0,1],values,marker='o',color='#40566a',linewidth=1.5)
    b.text(1.07,values[1]+{'990011':2,'990023':-2}.get(seed,0),seed,fontsize=8,va='center')
b.set_xlim(-.16,1.68)
b.set_xticks([0,1],['Baseline','Slip cost'])
b.set_title('B   Joint success by seed',loc='left',fontsize=12)
for ax in (a,b):
    ax.set_ylim(0,100)
    ax.set_ylabel('Episodes (%)')
    ax.spines[['right','top']].set_visible(False)
    ax.grid(axis='y',alpha=.19)
    ax.set_axisbelow(True)
fig.savefig(ROOT/'figures/z_outcomes.png',dpi=215)

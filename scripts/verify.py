"""Verify the paper's headline counts from per-start evidence, using only Python."""
from collections import Counter
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]

def rows(name):
    with (ROOT / 'evidence' / name).open(newline='') as stream:
        return list(csv.DictReader(stream))

def measure(records, filters, fields):
    subset = [row for row in records if all(row[k] == v for k, v in filters.items())]
    return [len(subset)] + [sum(int(row[field]) for row in subset) for field in fields]

def verify():
    x = rows('x_confirmation.csv')
    for arm, expected in [('dwell',[192,192,189,187,0]),('reset',[192,174,71,39,10])]:
        measured = measure(x, dict(backend='cuda0',arm=arm), ['entered','final61','final301','dropped'])
        assert measured == expected, (arm, measured)
        print('X GPU', arm, 'N / entered / final 1 s / final 5 s / dropped', *measured)
    y = rows('forearm_trials.csv')
    for axis, mode, expected in [('Y','fixed',[576,0,0]),('Y','free',[576,355,252]),('Z','fixed',[576,40,14]),('Z','free',[576,110,85])]:
        measured = measure(y, dict(axis=axis,mode=mode), ['held','final16'])
        assert measured == expected, (axis,mode,measured)
        print('World', axis, mode, 'N / any 16 / final 16', *measured)
    z = rows('z_trials.csv')
    key_fields = ('cohort','seed','horizon','bank','slot')
    keys = {arm:Counter(tuple(r[k] for k in key_fields) for r in z if r['arm']==arm) for arm in ('baseline','slip')}
    assert keys['baseline']==keys['slip'] and set(keys['slip'].values())=={1}
    fields = ['retained','supported45','repeated','joint']
    for cohort,arm,expected in [('original','baseline',[576,202,206,200,61]),('original','slip',[576,436,364,242,190]),('fresh','baseline',[576,202,211,210,58]),('fresh','slip',[576,431,389,263,208])]:
        filters = dict(cohort=cohort,arm=arm,horizon='25')
        measured = measure(z,filters,fields)
        assert measured == expected, (filters,measured)
        print('In-hand Z',cohort,arm,'N / retained / supported 45 / repeated / joint',*measured)
    assert measure(z, dict(cohort='fresh',arm='slip',horizon='25',seed='990037'),fields)==[192,166,164,152,133]
    reversal=Counter(r['outcome'] for r in rows('reversal_trials.csv'))
    assert sorted(reversal.values())==[6,25,225]
    print('Reversal',dict(reversal))
    for line in (ROOT/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split('  ',1)
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
            raise ValueError(f'Checksum mismatch: {name}')
    print('Release checksums verified.')

if __name__=='__main__':
    verify()

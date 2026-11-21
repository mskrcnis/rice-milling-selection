"""Extract annual ARPT candidate means; keep all milling answers quarantined."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
import subprocess
import numpy as np
import pandas as pd
import pdfplumber
import argparse

ROOT=Path(__file__).resolve().parents[1]
RAW=ROOT/'raw/arkansas'
OUT=ROOT/'derived/arkansas'
AUDIT=ROOT/'audit'
SITE_CODES={'RREC','PTRS','NEREC','NERREC','CLAY','DESHA','CHICOT','GRE','JAC','LAW','FAU','ARK','PHI','POI1','POI2','LON','NEC','NBES','PTBS'}
SITE_ALIAS={'PTBS':'PTRS','NBES':'NEC'}
KG_HA_PER_BU_AC=45*.45359237/.40468564224
COMPROMISED={(2022,'CLAY'):'weedy location',(2022,'LAW'):'weedy location',
 (2022,'PTRS'):'harvested at low moisture',(2024,'NEREC'):'significant bird damage',
 (2024,'ARK'):'volunteer rice throughout',(2025,'RREC'):'delayed harvest'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm(s):return re.sub('[^A-Z0-9]','',s.upper())
def center(w):return (w['x0']+w['x1'])/2

def body_rows(page,grain_template=None):
    words=page.extract_words(extra_attrs=['size'],x_tolerance=1,y_tolerance=2)
    grain_headers=[w for w in words if w['text']=='Grain' and w['top']<170]
    if not grain_headers and grain_template is None:return [],words
    grain=min(grain_headers,key=lambda w:w['x0']) if grain_template is None else grain_template
    anchors=[w for w in words if w['text'] in {'L','M','LA','S'} and abs(center(w)-center(grain))<18 and w['top']>grain['top']+8]
    rows=[]
    for anchor in anchors:
        line=[w for w in words if abs(w['top']-anchor['top'])<.65 and w['size']>=anchor['size']*.85]
        name=' '.join(w['text'] for w in line if w['x1']<anchor['x0']-8)
        if not name:continue
        rows.append((name,anchor['text'],anchor,line))
    return rows,words

def cell(words,x,spacing,anchor):
    chosen=[w for w in words if abs(w['top']-anchor['top'])<1.5 and abs(center(w)-x)<spacing*.48 and w['size']>=anchor['size']*.85]
    if not chosen:return '',None
    chosen=sorted(chosen,key=lambda w:w['x0'])
    value=''.join(w['text'] for w in chosen)
    supers=[w for w in words if abs(w['top']-anchor['top'])<3 and abs(center(w)-x)<spacing*.48 and w['size']<anchor['size']*.85 and w['text'].isdigit()]
    return value, ''.join(w['text'] for w in supers) or None

def value(s,kind):
    s=s.replace('*','').strip()
    if s in {'','--','-','—','–','.'}:return None
    if kind=='milling':
        m=re.fullmatch(r'(\d{1,2})[-:]([0-9]{1,2})',s)
        if not m:raise ValueError(('milling cell',s))
        hr,tr=map(int,m.groups());assert 0<=hr<=tr<=100
        return (hr,tr)
    if not re.fullmatch(r'\d{1,3}(?:\.\d+)?',s):raise ValueError(('grain cell',s))
    return float(s)

def select_record(records):
    annual=[r for r in records if r['source_type']=='contemporaneous_annual_matrix']
    candidates=annual or [r for r in records if r['value'] is not None] or records
    return min(candidates,key=lambda r:(r['report_year'],r['source_page']))

def annual_matrix(page,report_year,pageno,kind,template=None):
    rows,words=body_rows(page,None if template is None else template[0])
    first=min(a['top'] for _,_,a,_ in rows)
    headers=[w for w in words if w['text'].strip('*') in SITE_CODES and w['top']<first] if template is None else template[1]
    headers=sorted(headers,key=lambda w:w['x0'])
    assert len(headers)>=3,(report_year,pageno,'site headers',headers)
    centers=[center(w) for w in headers]
    spacing=float(np.median(np.diff(centers)))
    records=[]
    for name,grain,anchor,line in rows:
        for h,x in zip(headers,centers):
            raw,lodging=cell(words,x,spacing,anchor)
            v=value(raw,kind)
            site=h['text'].strip('*');site=SITE_ALIAS.get(site,site)
            records.append({'report_year':report_year,'year':report_year,'site':site,
                'cultivar_original':name,'candidate_id':norm(name),'grain_class':grain,'kind':kind,'value':v,
                'source_page':pageno,'source_table':f'{report_year} annual {kind} summary by location',
                'source_header':h['text'],'raw_cell':raw,'lodging_percent':lodging if kind=='grain' else None,
                'source_type':'contemporaneous_annual_matrix'})
    return records

def historical_site(page,report_year,pageno,title):
    if (report_year,pageno)==(2019,4):return []
    rows,words=body_rows(page)
    if not rows:return []
    if 'NERREC' in title:site='NERREC'
    elif 'NEREC' in title:site='NEREC'
    elif 'RREC' in title:site='RREC'
    elif 'PTRS' in title:site='PTRS'
    elif 'Clay County' in title:site='CLAY'
    elif 'Desha County' in title:site='DESHA'
    elif 'Chicot County' in title:site='CHICOT'
    else:raise ValueError(('unknown site',title))
    first=min(a['top'] for _,_,a,_ in rows)
    headers=sorted([dict(w,text=w['text'].strip('*')) for w in words if re.fullmatch(r'20\d{2}\*?',w['text']) and 55<w['top']<first],key=lambda w:w['x0'])
    years=sorted(set(int(w['text']) for w in headers))
    if not years:return []
    byyear={y:[w for w in headers if int(w['text'])==y] for y in years}
    assert all(len(v)==2 for v in byyear.values()),(report_year,pageno,[(y,len(v)) for y,v in byyear.items()])
    spacing=float(np.median(np.diff([center(w) for w in headers])))
    records=[]
    for name,grain,anchor,line in rows:
        for year,hs in byyear.items():
            if site=='NERREC' and year<=2021:continue
            actual_site=site
            if (report_year,pageno,year)==(2017,7,2017):actual_site='CHICOT'
            if (report_year,pageno,year)==(2018,7,2015):actual_site='DESHA'
            for kind,h in zip(['milling','grain'],hs):
                raw,lodging=cell(words,center(h),spacing,anchor)
                v=value(raw,kind)
                records.append({'report_year':report_year,'year':year,'site':actual_site,
                    'cultivar_original':name,'candidate_id':norm(name),'grain_class':grain,'kind':kind,'value':v,
                    'source_page':pageno,'source_table':title,'source_header':str(year),
                    'raw_cell':raw,'lodging_percent':None,'source_type':'individual_year_in_site_table'})
    return records

def site_metadata(year,pages):
    result={}
    if year<2021:return result
    for line in pages[1].splitlines():
        chunks=re.split(r'\s{2,}',line.strip())
        if len(chunks)>=6 and chunks[0].split(',')[0] in SITE_CODES:
            result[chunks[0].split(',')[0]]={'site_description':chunks[0],'planting_date_original':chunks[1],
                'emergence_date_original':chunks[2],'harvest_date_original':chunks[3],'soil_original':chunks[4],
                'location_type':chunks[5]}
    return result

def verify_text_geometry():
    """Reconcile Poppler tokens with PDF word geometry and visually checked blanks."""
    cells=pd.read_csv(OUT/'private_all_extracted_cells.csv',keep_default_na=False)
    comparisons=[];row_checks=[]
    visible_blank_slots={(2011,5,'JUPITER'):1,(2018,5,'PVL01'):4,(2020,3,'RT7801'):5,
                         (2021,7,'CLL17'):4,(2021,7,'RT7401'):4}
    for (report,page),group in cells.groupby(['report_year','source_page']):
        text=(OUT/f'report_{report}.txt').read_text().split('\f')[page-1]
        rows={}
        for line in text.splitlines():
            m=re.match(r'^\s*(.*?)\s{2,}(L|M|LA|S)\s{2,}(.*)$',line)
            if m:
                key=norm(m[1]);assert key not in rows,(report,page,'duplicate text row',key)
                rows[key]=[s.replace(' ','') for s in re.split(r'\s{2,}',m[3].strip())]
        row_checks.append({'report_year':int(report),'source_page':int(page),'text_rows':len(rows),
            'geometry_rows':group.candidate_id.nunique(),'equal_row_ids':set(rows)==set(group.candidate_id)})
        for candidate,records in group.groupby('candidate_id',sort=False):
            tokens=rows[candidate];blank=visible_blank_slots.get((report,page,candidate))
            if records.source_type.iloc[0]=='contemporaneous_annual_matrix':
                sites=list(records.site.unique());n=len(sites)+1
                tail=tokens[-(n-(blank is not None)):]
                if blank is not None:tail.insert(blank,'')
                assert len(tail)==n,(report,page,candidate,'column count')
                independently=dict(zip(sites,tail[:-1]))
                for _,r in records.iterrows():
                    expected=str(r.raw_cell)+str(r.lodging_percent);observed=independently[r.site]
                    comparisons.append({'report_year':report,'source_page':page,'candidate_id':candidate,
                        'year':r.year,'site':r.site,'kind':r.kind,'geometry_raw':expected,'text_raw':observed,'equal':expected==observed})
            else:
                headers=[]
                for line in text.splitlines():
                    years=re.findall(r'\b20\d{2}\b',line)
                    if len(years)>=4 and len(set(years))>=2:headers=[int(x) for x in years];break
                assert len(headers)==6,(report,page,'historical header count',headers)
                years=headers[:3];tail=tokens[-(8-(blank is not None)):]
                if blank is not None:tail.insert(blank,'')
                for _,r in records.iterrows():
                    j=years.index(int(r.year))+(4 if r.kind=='grain' else 0)
                    observed=tail[j];expected=str(r.raw_cell)
                    comparisons.append({'report_year':report,'source_page':page,'candidate_id':candidate,
                        'year':r.year,'site':r.site,'kind':r.kind,'geometry_raw':expected,'text_raw':observed,'equal':expected==observed})
    comparison=pd.DataFrame(comparisons);checks=pd.DataFrame(row_checks)
    comparison.to_csv(OUT/'private_independent_text_geometry.csv',index=False)
    checks.to_csv(AUDIT/'arkansas_independent_table_row_checks.csv',index=False)
    summary={'method':'Poppler fixed-layout tokens versus pdfplumber word bounding boxes and font sizes; five genuinely empty cells confirmed visually and retained as empty tokens',
        'tables':len(checks),'candidate_table_rows':int(checks.text_rows.sum()),'measurement_cells':len(comparison),
        'row_identity_checks_passed':bool(checks.equal_row_ids.all()),'cell_checks_passed':bool(comparison.equal.all()),
        'mismatched_cells':int((~comparison.equal).sum())}
    (AUDIT/'arkansas_independent_extraction_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    assert checks.equal_row_ids.all(),checks[~checks.equal_row_ids].to_dict('records')
    assert comparison.equal.all(),f'{int((~comparison.equal).sum())} mismatched cells; inspect private_independent_text_geometry.csv'
    visual=OUT/'private_visual_checks/manual_checks.json'
    if visual.exists():
        manual=json.loads(visual.read_text())
        pub=pd.read_csv(OUT/'public_features.csv').set_index('row_id')
        prv=pd.read_csv(OUT/'private_outcomes.csv').set_index('row_id')
        for r in manual['row_checks']:
            for col,v in r['expected'].items():
                observed=(pub if col in pub else prv).loc[r['row_id'],col]
                assert observed==v,('manual visual mismatch',r['row_id'],col)
        for r in manual['blank_cell_checks']:
            sel=cells
            for col,v in r.items():sel=sel[sel[col]==v]
            assert len(sel)==1 and sel.raw_cell.iloc[0]=='',('manual blank mismatch',r)
        summary.update(manual_candidate_rows=len(manual['row_checks']),manual_blank_cells=len(manual['blank_cell_checks']),manual_checks_passed=True)
        (AUDIT/'arkansas_independent_extraction_check.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

def main():
    OUT.mkdir(parents=True,exist_ok=True);AUDIT.mkdir(parents=True,exist_ok=True)
    source_manifest=json.loads((RAW/'source_manifest.json').read_text())
    sources={Path(r['file']).name:r for r in source_manifest if r.get('downloaded')}
    extracted=[];meta={};screen=[{'report_year':y,'downloaded_final':False,'extracted_measurement_cells':0,'grain_cells':0,'milling_cells':0,'eligible_table_structure':False,
        'reason':'No standalone FINAL annual report recovered from official archive and targeted official-site search; retrospective pooled columns do not establish a site-year candidate universe.'} for y in [2008,2009,2010]];exclusions=[]
    for year in range(2011,2026):
        source=RAW/f'report_{year}.pdf'
        assert sha(source)==sources[source.name]['sha256']
        textfile=OUT/f'report_{year}.txt'
        subprocess.run(['pdftotext','-layout',str(source),str(textfile)],check=True,capture_output=True)
        pages=textfile.read_text().split('\f');meta[year]=site_metadata(year,pages)
        before=len(extracted)
        with pdfplumber.open(source) as doc:
            for i,page in enumerate(doc.pages):
                lines=pages[i].strip().splitlines();title=' '.join(l.strip() for l in lines[:2])
                if year>=2021 and i in [2,3]:
                    extracted+=annual_matrix(page,year,i+1,'grain' if i==2 else 'milling')
                elif year<=2015 and title.startswith('Table 2.'):
                    extracted+=annual_matrix(page,year,i+1,'grain')
                elif 2016<=year<=2020 and re.search(rf'{year} Summary\s*$',title):
                    extracted+=annual_matrix(page,year,i+1,'grain')
                    if year==2019:
                        _,ws=body_rows(page)
                        grain=min([w for w in ws if w['text']=='Grain' and w['top']<170],key=lambda w:w['x0'])
                        grain=dict(grain,top=0)
                        headers=[w for w in ws if w['text'].strip('*') in SITE_CODES and w['top']<150]
                        extracted+=annual_matrix(doc.pages[i+1],year,i+2,'grain',(grain,headers))
                elif year>=2016 and 'Summary' in title and re.search(r'(County|RREC|NEREC|PTRS)',title) and not title.startswith('DD50'):
                    extracted+=historical_site(page,year,i+1,title)
                elif any(s in title for s in ['Planting Date','seeding date','All Locations','Permanent Locations','Producer Rice','Cultivar Rice Testing']):
                    exclusions.append({'report_year':year,'page':i+1,'title':title,'reason':'outside site-year ARPT candidate pool: planting-date, aggregate, or distinct on-farm PREP/CRT design'})
        new=extracted[before:]
        screen.append({'report_year':year,'downloaded_final':True,'source_file':str(source.relative_to(ROOT)),
            'extracted_measurement_cells':len(new),'grain_cells':sum(r['kind']=='grain' for r in new),
            'milling_cells':sum(r['kind']=='milling' for r in new),
            'eligible_table_structure':any(r['kind']=='milling' for r in new),
            'reason':'individual site-year ARPT grain and milling cells available' if any(r['kind']=='milling' for r in new) else 'Contemporaneous site grain establishes candidate universe; milling in this annual report is pooled across sites and is not paired. Later reports may supply explicitly labelled individual-year cells.'})
        print(year,'extracted cells',len(extracted)-before,flush=True)
    screen.append({'report_year':2026,'downloaded_final':False,'extracted_measurement_cells':0,'grain_cells':0,'milling_cells':0,'eligible_table_structure':False,'reason':'Only preliminary results found; final URL failed and official archive stops at 2025 as accessed'})
    exclusions.extend([
        {'report_year':2019,'page':4,'title':'RREC site summary','reason':'Title says 2017–2019 but column headers say 2016–2018; all cells from this page excluded without inferring intended years.'},
        {'report_year':2022,'page':'site summaries','title':'NERREC retrospective 2021 cells','reason':'2021 annual trial distinguishes POI1 and POI2; later NERREC aggregate cannot identify either site. Do not merge these cells.'}])
    allcells=[]
    for r in extracted:
        r['source_file']=f'raw/arkansas/report_{r["report_year"]}.pdf'
        r['source_sha256']=sources[Path(r['source_file']).name]['sha256']
        r['source_url']=sources[Path(r['source_file']).name]['url']
        allcells.append({**r,'value':json.dumps(r['value'])})
    pd.DataFrame(allcells).to_csv(OUT/'private_all_extracted_cells.csv',index=False)
    # Prespecified precedence: same-year annual matrix, then earliest report containing that year.
    grouped={}
    for r in extracted:
        key=(r['year'],r['site'],r['candidate_id'],r['kind'])
        grouped.setdefault(key,[]).append(r)
    chosen={};duplicates=[]
    for key,rs in grouped.items():
        selected=select_record(rs)
        chosen[key]=selected
        for r in rs:
            if r is not selected:
                duplicates.append({'year':key[0],'site':key[1],'candidate_id':key[2],'kind':key[3],
                    'chosen_report':selected['report_year'],'other_report':r['report_year'],
                    'chosen_page':selected['source_page'],'other_page':r['source_page'],
                    'conflict':r['value'] is not None and selected['value']!=r['value'],
                    'chosen_raw':selected['raw_cell'],'other_raw':r['raw_cell']})
    public=[];private=[]
    annual_keys={k[:3] for k,rs in grouped.items() if k[3]=='grain' and any(r['source_type']=='contemporaneous_annual_matrix' for r in rs)}
    for year,site,candidate in sorted(annual_keys):
        g=chosen.get((year,site,candidate,'grain'));m=chosen.get((year,site,candidate,'milling'))
        base=g or m;observed_yield=g is not None and g['value'] is not None
        paired=observed_yield and m is not None and m['value'] is not None
        pool_id=f'ARPT__{year}__{site}';row_id=f'{pool_id}__{candidate}'
        damage=COMPROMISED.get((year,site),'')
        public.append({'row_id':row_id,'pool_id':pool_id,'candidate_id':candidate,'cultivar_original':base['cultivar_original'],
            'site':site,'year':year,'program':'Arkansas_ARPT','trial_stage':'ARPT',
            'field_yield':g['value']*KG_HA_PER_BU_AC if observed_yield else np.nan,
            'field_yield_bu_ac':g['value'] if observed_yield else np.nan,'yield_unit':'kg/ha at 12% rough-rice moisture',
            'grain_class':base['grain_class'],'replicate_count':4 if year>=2013 else 3,'replicate_data_available':False,
            'moisture_basis_percent':12,'bushel_mass_lb':45,
            'paired_outcome_available':paired,'pool_compromised':bool(damage),'pool_compromise_reason':damage,
            'harvest_condition_compromised':(year,site) in {(2022,'PTRS'),(2025,'RREC')},
            'eligible':paired and not damage,'sensitivity_include_compromised':paired,'previously_seen':year==2025 and site=='NERREC',
            'exposure':'previously_seen' if year==2025 and site=='NERREC' else 'unseen_by_policy_development',
            'lodging_percent':g['lodging_percent'] if g else None,
            'grain_source_report':g['report_year'] if g else None,'grain_source_page':g['source_page'] if g else None,
            'milling_source_report':m['report_year'] if m else None,'milling_source_page':m['source_page'] if m else None,
            'source_selection':'contemporaneous annual grain candidate universe; same-year annual cell including missing preferred, otherwise earliest observed individual-year cell; no inferred cultivar aliases',
            **meta.get(year,{}).get(site,{})})
        private.append({'row_id':row_id,'pool_id':pool_id,'candidate_id':candidate,
            'hr_percent':m['value'][0] if m and m['value'] is not None else np.nan,
            'tr_percent':m['value'][1] if m and m['value'] is not None else np.nan,
            'head_output_kg_ha':g['value']*KG_HA_PER_BU_AC*m['value'][0]/100 if paired else np.nan,
            'milling_denominator':'original rough-rice mass','milling_unit':'percent',
            'endpoint_definition':'product of published cultivar mean paddy yield and mean head recovery; not mean plotwise product',
            'source_report':m['report_year'] if m else None,'source_page':m['source_page'] if m else None,
            'source_table':m['source_table'] if m else None,'raw_milling_cell':m['raw_cell'] if m else None})
    pub=pd.DataFrame(public);prv=pd.DataFrame(private)
    assert pub.row_id.is_unique and prv.row_id.is_unique and set(pub.row_id)==set(prv.row_id)
    pub.to_csv(OUT/'public_features.csv',index=False);prv.to_csv(OUT/'private_outcomes.csv',index=False)
    pd.DataFrame(duplicates).to_csv(OUT/'private_duplicate_audit.csv',index=False)
    pd.DataFrame(exclusions).to_csv(AUDIT/'arkansas_excluded_tables.csv',index=False)
    pd.DataFrame(screen).to_csv(AUDIT/'arkansas_year_screen.csv',index=False)
    coverage=pub.groupby(['year','site','pool_id'],dropna=False).agg(candidates=('row_id','size'),
        yield_observed=('field_yield','count'),paired=('paired_outcome_available','sum'),eligible=('eligible','sum'),
        grain_classes=('grain_class',lambda x:','.join(sorted(x.unique()))),previously_seen=('previously_seen','max'),
        compromised=('pool_compromised','max'),reason=('pool_compromise_reason','first')).reset_index()
    coverage['decision_pool_eligible_n10']=coverage.eligible>=10
    coverage['unseen_transfer_pool_2017_2025']=coverage.decision_pool_eligible_n10 & coverage.year.between(2017,2025) & ~coverage.previously_seen
    coverage.to_csv(AUDIT/'arkansas_coverage.csv',index=False)
    manifest={'prepared_utc':datetime.now(timezone.utc).isoformat(),'input_hashes':{r['file']:r['sha256'] for r in sources.values()},
        'output_hashes':{str(p.relative_to(ROOT)):sha(p) for p in [OUT/'public_features.csv',OUT/'private_outcomes.csv',OUT/'private_all_extracted_cells.csv',OUT/'private_duplicate_audit.csv',AUDIT/'arkansas_coverage.csv',AUDIT/'arkansas_year_screen.csv']},
        'row_count':len(pub),'pool_count':pub.pool_id.nunique(),'eligible_rows':int(pub.eligible.sum()),
        'eligible_pools':pub.loc[pub.eligible,'pool_id'].nunique(),'previously_seen_pools':pub.loc[pub.previously_seen,'pool_id'].nunique(),
        'source_precision':'Integer bushels/acre and integer percent recoveries, retained exactly; superscript lodging separately extracted.',
        'conversion_factor_kg_ha_per_bu_ac':KG_HA_PER_BU_AC}
    (AUDIT/'arkansas_preparation_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(coverage.groupby('year')[['candidates','paired','eligible']].sum().to_string())
    print({k:v for k,v in manifest.items() if k in ['row_count','pool_count','eligible_rows','eligible_pools','previously_seen_pools']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify-only',action='store_true')
    if parser.parse_args().verify_only:print(verify_text_geometry())
    else:main()

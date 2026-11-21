"""Acquire immutable official Arkansas rice reports and methods evidence."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen
import argparse
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
BASE='https://www.uaex.uada.edu/farm-ranch/crops-commercial-horticulture/rice/'
SOURCES={
 'publications_archive.html':'https://aaes.uada.edu/communications/publications/',
 'rice_research_2014.pdf':'https://aaes.uada.edu/files/2023/07/626.pdf',
 'rice_research_2015.pdf':'https://aaes.uada.edu/files/2023/07/635.pdf',
 'rice_research_2016.pdf':'https://aaes.uada.edu/files/2023/07/640.pdf',
 'publication_archives.html':'https://aaes.uada.edu/communications/publications/publication-archives/',
 'rice_research_2013.pdf':'https://bpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/3/599/files/2023/07/617.pdf',
 'rice_research_2017.pdf':'https://bpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/3/599/files/2023/07/651_BR_Wells_Arkansas_Rice_Research_Studies_2017.pdf',
 'rice_research_2019.pdf':'https://bpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/3/599/files/2023/07/667_BR_Wells_Arkansas_Rice_Research_Studies_2019.pdf',
 'rice_research_2020.pdf':'https://aaes.uada.edu/files/2025/10/676_BR_Wells_Arkansas_Rice_Research_Studies_2020.pdf',
 'report_2011.pdf':BASE+'RIS%20170%20ARPT%202011.pdf',
 'laboratory_milling_chapter14.pdf':'https://www.uaex.uada.edu/publications/pdf/mp192/chapter-14.pdf',
 'rice_glossary_chapter19.pdf':'https://www.uaex.uada.edu/publications/pdf/mp192/chapter-19.pdf',
 'rice_research_2023.pdf':'https://wordpressua.uark.edu/aaes/files/2024/08/705_BR_Wells_Arkansas_Rice_Research_Studies_2023.pdf',
 'rice_research_2024.pdf':'https://bpb-us-e1.wpmucdn.com/wordpressua.uark.edu/dist/3/599/files/2025/08/714_BR_Wells_Arkansas_Rice_Research_Studies_2024.pdf',
 'archive.html':'https://uaex.uada.edu/farm-ranch/crops-commercial-horticulture/rice/',
 'variety_testing.html':'https://aaes.uada.edu/variety-testing/',
 'terms.html':'https://uaex.uada.edu/disclaimer.aspx',
 'rice_drying.html':'https://www.uaex.uada.edu/farm-ranch/crops-commercial-horticulture/Grain_drying_and_storage/rice_drying_and_storage.aspx',
 'milling_definition.html':'https://uaex.uada.edu/media-resources/news/2021/december2021/12-17-2021-Ark-YEAREND-rice.aspx',
 'report_2012.pdf':BASE+'RIS%20171%20ARPT%202012.pdf',
 'report_2013.pdf':BASE+'RIS%20172%20ARPT%202013.pdf',
 'report_2014.pdf':BASE+'RIS%20173%20ARPT%202014.pdf',
 'report_2015.pdf':BASE+'RIS%20175%20ARPT%202015-2.pdf',
 **{f'report_{y}.pdf':BASE+f'RIS%20{y-1840}%20AR%20Rice%20Cultivar%20Testing%20{y}.pdf' for y in range(2016,2021)},
 'report_2021.pdf':BASE+'RIS%20181%20AR%20Rice%20Performance%20Trials%202021.pdf',
 **{f'report_{y}.pdf':f'https://arkansascrops.uada.edu/posts/crops/rice/{y}_Arkansas_Rice_Performance_Trials_Final.pdf' for y in range(2022,2026)},
 'report_2026_unavailable_probe.pdf':'https://arkansascrops.uada.edu/posts/crops/rice/2026_Arkansas_Rice_Performance_Trials_Final.pdf',
 'preliminary_2026.html':'https://arkansascrops.uada.edu/posts/crops/rice/arkansas-rice-update-9-18-26.aspx',
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def acquire_one(item):
 name,url=item;path=ROOT/'raw/arkansas'/name
 if path.exists():return None
 rec={'file':str(path.relative_to(ROOT)),'url':url,'accessed_utc':datetime.now(timezone.utc).isoformat(),
      'source':'University of Arkansas System Division of Agriculture official public site',
      'redistribution':'Public access does not establish an open license; preserve attribution and source disclaimer. No redistribution permission inferred.'}
 try:
  with urlopen(Request(url,headers={'User-Agent':'Mozilla/5.0 research reproducibility audit'}),timeout=50) as response:
   body=response.read();rec.update(status=response.status,final_url=response.url,content_type=response.headers.get('content-type'),last_modified=response.headers.get('last-modified'))
  if name.endswith('.pdf') and not body.startswith(b'%PDF'):raise ValueError('Response is not a PDF')
  path.write_bytes(body);rec.update(sha256=sha(path),bytes=len(body),downloaded=True)
 except Exception as exc:
  rec.update(downloaded=False,error=repr(exc));rec.pop('file',None)
 print(name,rec['downloaded'],flush=True)
 return rec

def main(offline=False):
 directory=ROOT/'raw/arkansas';directory.mkdir(parents=True,exist_ok=True)
 manifest=directory/'source_manifest.json'
 existing=json.loads(manifest.read_text()) if manifest.exists() else []
 tracked={r.get('file') for r in existing if r.get('downloaded')}
 for name,url in SOURCES.items():
  p=directory/name
  if p.exists() and str(p.relative_to(ROOT)) not in tracked:
   existing.append({'file':str(p.relative_to(ROOT)),'url':url,'accessed_utc':datetime.now(timezone.utc).isoformat(),
    'sha256':sha(p),'bytes':p.stat().st_size,'downloaded':True,'source':'Official public source; manually acquired earlier in this same extraction session',
    'redistribution':'No open license inferred; source attribution retained.'})
 for r in existing:
  if r.get('downloaded'):assert sha(ROOT/r['file'])==r['sha256'],r['file']
 if offline:return
 with ThreadPoolExecutor(max_workers=4) as executor:
  new=[x for x in executor.map(acquire_one,SOURCES.items()) if x is not None]
 manifest.write_text(json.dumps(existing+new,indent=2)+'\n')

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--offline',action='store_true')
 main(parser.parse_args().offline)

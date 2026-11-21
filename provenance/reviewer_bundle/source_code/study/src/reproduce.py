import argparse
import json
from pathlib import Path
from .common import ROOT,digest,write_json,verify_freeze


def verified_run(run):
    run=Path(run).resolve()
    frozen_count=verify_freeze(run/'evaluation_freeze.json')
    documents={}
    for name in ['verification_summary.json','table_verification.json','all_tables_event_regeneration.json']:
        path=run/'verification'/name
        if not path.is_file():raise ValueError('Missing independent verification: '+str(path))
        document=json.loads(path.read_text())
        if document.get('status')!='passed':raise ValueError('Verification did not pass: '+str(path))
        documents[name]=document
    events=documents['verification_summary.json']
    bindings=events.get('input_sha256',{})
    required=['evaluation_events.jsonl.gz','per_pool_results.csv','frozen_initial_predictions.csv',
              'frozen_initial_batch_plans.json','selected_configuration.json','evaluation_freeze.json']
    for name in required:
        path=run/name
        if bindings.get(str(path))!=digest(path):
            raise ValueError('Event verification does not bind current run artifact: '+str(path))
    for name,expected in bindings.items():
        if digest(name)!=expected:raise ValueError('Event-verification input changed: '+name)
    regenerated=run/'verification/regenerated_per_pool_results.csv'
    regenerated_hash=digest(regenerated)
    if events.get('regenerated_results_sha256')!=regenerated_hash:
        raise ValueError('Independent per-pool results changed: '+str(regenerated))
    core=documents['table_verification.json']
    if core.get('event_verification_summary_sha256')!=digest(run/'verification/verification_summary.json'):
        raise ValueError('Core-table verification binds a different event-verification summary')
    if core.get('regenerated_per_pool_sha256')!=regenerated_hash:
        raise ValueError('Core-table verification binds different independent per-pool results')
    config=json.loads((run/'selected_configuration.json').read_text())
    core_names={'pool_seed_averages.csv','policy_summary.csv','paired_pool_differences.csv',
                'paired_summary.csv','paired_by_year.csv'}
    if 'secondary_adaptive' in config:core_names.add('secondary_adaptive_paired.csv')
    if set(core.get('tables',{}))!=core_names or core.get('table_count')!=len(core_names):
        raise ValueError('Incomplete independently reconstructed core-table inventory')
    for name,record in core['tables'].items():
        if record.get('actual_sha256')!=digest(run/'tables'/name):
            raise ValueError('Verified core table changed: '+str(run/'tables'/name))
        if record.get('regenerated_sha256')!=digest(run/'verification/regenerated_tables'/name):
            raise ValueError('Independent aggregate reconstruction changed: '+name)
    all_tables=documents['all_tables_event_regeneration.json']
    table_names={p.name for p in (run/'tables').glob('*.csv')}
    if set(all_tables.get('tables',{}))!=table_names or all_tables.get('table_count')!=len(table_names):
        raise ValueError('All-table verification inventory differs from actual tables')
    if all_tables.get('event_metrics_sha256')!=regenerated_hash:
        raise ValueError('All-table regeneration binds different independent per-pool results')
    for name,record in all_tables['tables'].items():
        if record.get('source_sha256')!=digest(run/'tables'/name):
            raise ValueError('Event-regenerated aggregate table changed: '+name)
    return {'frozen_files_checked':frozen_count,'event_count':events['event_count'],
            'core_tables_checked':len(core_names),'all_tables_checked':len(table_names),
            'verification_sha256':{name:digest(run/'verification'/name) for name in documents}}


def artifact_paths(run):
    paths={'evaluation_events.jsonl.gz','per_pool_results.csv','frozen_initial_predictions.csv',
           'frozen_initial_batch_plans.json','prediction_diagnostics.csv','selected_configuration.json',
           'frozen_models.joblib','model_attempts.json'}
    paths.update(str(p.relative_to(run)) for directory in ['tables','figures','diagnostics']
                 for p in (run/directory).glob('*') if p.suffix in ('.csv','.png','.svg'))
    paths.update(str(p.relative_to(run)) for p in (run/'sensitivity').glob('*.csv'))
    return paths


def compare_runs(first,second):
    first=Path(first).resolve();second=Path(second).resolve()
    result={'status':'fail',
            'scope':'Two fresh historical-model fits and identical-policy replays; reproducibility, not independent scientific replication.',
            'first_run':str(first),'second_run':str(second),'artifacts':[]}
    destination=ROOT/'audit/clean_repeat_verification.json'
    try:
        result['verification']={'first':verified_run(first),'second':verified_run(second)}
    except (ValueError,KeyError,OSError,TypeError) as error:
        result['preflight_error']=str(error)
        write_json(destination,result)
        raise ValueError('Clean-repeat verification preflight failed; see '+str(destination)) from error
    first_paths=artifact_paths(first);second_paths=artifact_paths(second)
    result['inventory_difference']={'first_only':sorted(first_paths-second_paths),
                                    'second_only':sorted(second_paths-first_paths)}
    comparisons=[]
    for name in sorted(first_paths|second_paths):
        a=first/name;b=second/name
        a_hash=digest(a) if a.is_file() else None
        b_hash=digest(b) if b.is_file() else None
        same=a_hash is not None and a_hash==b_hash
        comparisons.append({'path':name,'identical':same,'first_sha256':a_hash,'second_sha256':b_hash})
    result['artifacts']=comparisons
    result['status']='pass' if first_paths==second_paths and all(r['identical'] for r in comparisons) else 'fail'
    write_json(destination,result)
    if result['status']!='pass':raise ValueError('Clean repeat differs; see audit/clean_repeat_verification.json')
    return result


def verify_preservation():
    source=ROOT/'audit/awd_preservation_manifest.json'
    manifest=json.loads(source.read_text())
    failures=[p for p,h in manifest['files'].items() if not (ROOT.parent/p).exists() or digest(ROOT.parent/p)!=h]
    result={'status':'pass' if not failures else 'fail','files_checked':len(manifest['files']),'changed_or_missing':failures}
    write_json(ROOT/'audit/awd_preservation_verification.json',result)
    if failures:raise ValueError('Previously preserved AWD files changed')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('first');parser.add_argument('second')
    args=parser.parse_args()
    result=compare_runs(args.first,args.second)
    preservation=verify_preservation()
    print(json.dumps({'repeat':result['status'],'identical_artifacts':len(result['artifacts']),'awd_preservation':preservation},indent=2))

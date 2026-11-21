import argparse
import json
import shutil
from pathlib import Path
import joblib
import pandas as pd
from .common import ROOT,now,stage,freeze,verify_freeze,write_json,digest
from .data import harmonize_public,partition_keys
from .experiment import read_public,load_key,develop,fit_menu,initial_predictions,evaluate
from .models import MODEL_MENU
from .analysis import plot_development,aggregate,plot_final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--offline',action='store_true')
    parser.add_argument('--stage',choices=['prepare','development','freeze','all'],default='all')
    parser.add_argument('--output',default='outputs/run')
    args=parser.parse_args()
    if args.stage=='prepare':
        summary=harmonize_public()
        labels=partition_keys()
        stage('cohort_preparation','complete',command='python -m src.run --stage prepare',
              inputs=[ROOT/f'derived/{source}/{name}.csv' for source in ['uruguay','arkansas'] for name in ['public_features','private_outcomes']],
              outputs=[ROOT/'protocol/split_manifest.csv',ROOT/'derived/cohort/public_candidates.csv',ROOT/'private/development_labels.csv',ROOT/'private/evaluation_key.csv'],
              next_action='Hash source coverage and split specification before development and final evaluation.')
        print(json.dumps({'coverage':summary,'private_partition_counts':labels},indent=2))
        return
    development=read_public(ROOT/'derived/cohort/development_public.csv')
    if args.stage=='development':
        verify_freeze(ROOT/'protocol/development_input_lock.json')
        out=ROOT/args.output
        if out.exists() and any(out.iterdir()):raise FileExistsError('Development output must be clean: '+str(out))
        out.mkdir(parents=True,exist_ok=True)
        key=load_key(ROOT/'private/development_labels.csv',development.row_id)
        config=develop(development,key,[2012,2013,2014,2015,2016],out)
        plot_development(out)
        stage('development','complete',command='python -m src.run --offline --stage development --output '+args.output,
              inputs=[ROOT/'protocol/development_input_lock.json',ROOT/'private/development_labels.csv'],
              outputs=[out/'selected_configuration.json',out/'development_per_pool.csv',out/'development_events.jsonl.gz',out/'figures/development_baseline_curve.png'],
              next_action='Lock selected configuration, code and final prediction plans before opening evaluation keys.')
        print(json.dumps(config,indent=2))
        return
    verify_freeze(ROOT/'protocol/cohort_lock.json')
    verify_freeze(ROOT/'protocol/analysis_lock.json')
    out=ROOT/args.output
    if out.exists() and any(out.iterdir()):
        stamp=now().replace('-','').replace(':','').replace('.','').replace('+','_')
        out=out.with_name(out.name+'_'+stamp)
    out.mkdir(parents=True,exist_ok=True)
    print('Run directory:',out,flush=True)
    started=now()
    config=json.loads((ROOT/'protocol/selected_configuration.json').read_text())
    write_json(out/'selected_configuration.json',config)
    key=load_key(ROOT/'private/development_labels.csv',development.row_id)
    models=fit_menu(development,key,MODEL_MENU,out)
    joblib.dump(models,out/'frozen_models.joblib')
    public=read_public(ROOT/'derived/cohort/evaluation_public.csv')
    initial_predictions(public,models,config,out)
    shutil.copyfile(ROOT/'derived/cohort/evaluation_public.csv',out/'frozen_public_candidates.csv')
    source=out/'frozen_source';source.mkdir()
    for path in sorted((ROOT/'src').glob('*.py')):shutil.copyfile(path,source/path.name)
    frozen_paths=[out/'selected_configuration.json',out/'frozen_models.joblib',out/'frozen_initial_predictions.csv',
                  out/'frozen_initial_batch_plans.json',out/'frozen_public_candidates.csv',
                  ROOT/'protocol/analysis_lock.json',ROOT/'protocol/cohort_lock.json',
                  ROOT/'derived/cohort/evaluation_public.csv',ROOT/'private/evaluation_key.csv',
                  ROOT/'derived/cohort/all_public_metadata.csv',
                  ROOT/'derived/cohort/line_history_status.csv',
                  ROOT/'derived/uruguay/private_outcomes.csv',ROOT/'derived/arkansas/private_outcomes.csv',
                  ROOT/'derived/arkansas/public_features.csv']+sorted(source.glob('*.py'))
    freeze(frozen_paths,out/'evaluation_freeze.json',{
        'meaning':'Code, configurations, model parameters, all initial forecasts and initial batch plans frozen before evaluation answers are opened.',
        'model_training_max_year':2016,'source_program':'uruguay','policy_selection_years':[2012,2013,2014,2015,2016],
        'within_pool_feedback':'Only paid batch-one answers may update batch-two rankings.',
        'outcome_exposure':'Source extraction reviewers saw labels. First modeling run remained outcome-separated until this freeze; clean repeats are reproducibility only.'})
    stage('prediction_policy_freeze','complete',command='python -m src.run --offline --output '+args.output,
          inputs=[ROOT/'protocol/analysis_lock.json'],outputs=[out/'evaluation_freeze.json'],
          next_action='Evaluate each fixed policy through paid query gate, then independent scoring.')
    if args.stage=='freeze':return
    # The evaluation answer key is first opened only after the complete immutable freeze above.
    evaluation_key=load_key(ROOT/'private/evaluation_key.csv',public.row_id)
    print('Running fixed-policy evaluation after verified prediction freeze.',flush=True)
    result=evaluate(public,evaluation_key,models,config,out)
    print('Scored',len(result),'events; independently reconstructing actions and tables.',flush=True)
    from .verify import verify_run
    verify_run(out)
    aggregate(result,config,out)
    from .verify_tables import verify_tables
    verify_tables(out)
    from .regenerate_tables import verify_all_tables
    verify_all_tables(out)
    plot_final(out,config)
    from .diagnostics import run_diagnostics
    run_diagnostics(out)
    from .availability_report import run_availability_report
    run_availability_report(out)
    print('Decision and table verification passed; running fixed-query source sensitivities.',flush=True)
    from .sensitivity import run_sensitivities
    run_sensitivities(out/'evaluation_events.jsonl.gz',ROOT/'derived/cohort/evaluation_public.csv',
                     ROOT/'derived/cohort/all_public_metadata.csv',ROOT/'derived/uruguay/private_outcomes.csv',
                     ROOT/'derived/arkansas/private_outcomes.csv',ROOT/'derived/arkansas/public_features.csv',
                     out/'evaluation_freeze.json',out/'sensitivity')
    print('Generating report and manuscript from verified tables.',flush=True)
    from .report import build_report
    build_report(out)
    write_json(out/'runtime.json',{'started_utc':started,'scored_finished_utc':now(),
                                  'training_mode':'fresh_fit','output_directory':str(out),
                                  'evaluation_events_sha256':digest(out/'evaluation_events.jsonl.gz'),
                                  'evaluation_freeze_sha256':digest(out/'evaluation_freeze.json')})
    stage('scoring','complete',command='python -m src.run --offline --output '+args.output,
          inputs=[out/'evaluation_freeze.json'],outputs=[out/'evaluation_events.jsonl.gz',out/'per_pool_results.csv',out/'tables/policy_summary.csv'],
          next_action='Compare this completed replay with another clean run using src.reproduce; review the generated manuscript.')


if __name__=='__main__':main()

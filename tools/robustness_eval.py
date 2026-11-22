#!/usr/bin/env python3
"""List, preview, or explicitly run independent nuScenes robustness evaluations."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'configs/robustness/scenarios.json'


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def command_for(scenario, checkpoint, output, gpu):
    return [
        sys.executable, '-B',
        str(ROOT / 'tools/test.py'),
        str(ROOT / scenario['config']),
        str(checkpoint), '--gpu-id',
        str(gpu), '--seed', '0', '--eval', 'bbox', '--out',
        str(output / 'outputs.pkl'), '--eval-options',
        'jsonfile_prefix=' + str(output / 'predictions')
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--list',
                        action='store_true',
                        help='List all scenarios; never evaluate')
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Print commands without creating outputs or using GPU')
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--scenario',
                           '--scenarios',
                           nargs='+',
                           dest='scenarios')
    selection.add_argument('--suite', choices=['paper', 'toolkit'])
    selection.add_argument('--all',
                           action='store_true',
                           help='Explicitly select all 52 scenarios')
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output-root',
                        type=Path,
                        default=ROOT / 'work_dirs/robustness')
    args = parser.parse_args()
    with MANIFEST.open() as stream:
        manifest = json.load(stream)
    rows = manifest['scenarios']
    if args.list:
        for row in rows:
            print('{:<38} {:<8} {} {}'.format(
                row['name'], row['suite'], row['corruption'],
                json.dumps(row['parameters'], sort_keys=True)))
        return 0
    if args.scenarios:
        unknown = set(args.scenarios) - {r['name'] for r in rows}
        if unknown:
            parser.error('Unknown scenarios: ' + ', '.join(sorted(unknown)))
        selected = [r for r in rows if r['name'] in args.scenarios]
    elif args.all:
        selected = rows
    else:
        selected = [r for r in rows if r['suite'] == (args.suite or 'paper')]
    checkpoint = (args.checkpoint or ROOT / manifest['checkpoint']).resolve()
    if not checkpoint.is_file():
        parser.error('Checkpoint not found: ' + str(checkpoint))
    for row in selected:
        if not (ROOT / row['config']).is_file():
            parser.error('Config not found: ' + row['config'])
        for dependency in row['metadata_dependencies']:
            actual = dependency
            if dependency == manifest['noise_metadata']:
                actual = os.environ.get('NUSCENES_R_METADATA', dependency)
            if dependency == manifest['mask_dir']:
                actual = os.environ.get('NUSCENES_R_MASK_DIR', dependency)
            if not Path(actual).exists():
                parser.error('Missing dependency: ' + actual)
    if args.dry_run:
        for row in selected:
            output = args.output_root.resolve() / '<unique-run>' / row['name']
            print(shlex.join(command_for(row, checkpoint, output,
                                         args.gpu_id)))
        return 0

    # Only this explicit execution branch imports the model environment.
    sys.path.insert(0, str(ROOT))
    import torch
    import mmcv
    import mmdet
    from mmcv import Config
    from plugin.futr3d.robustness_scenarios import load_manifest
    assert load_manifest() == manifest
    args.output_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(
        tempfile.mkdtemp(prefix='run-', dir=str(args.output_root.resolve())))
    checkpoint_hash = digest(checkpoint)
    noise_path = Path(
        os.environ.get('NUSCENES_R_METADATA', manifest['noise_metadata']))
    mask_dir = Path(os.environ.get('NUSCENES_R_MASK_DIR',
                                   manifest['mask_dir']))
    dependencies = {}
    if any(r['metadata_dependencies'] for r in selected):
        dependencies[str(noise_path)] = digest(noise_path)
    if any(r['corruption'] == 'occlusion' for r in selected):
        for mask in sorted(mask_dir.glob('mask_*.jpg')):
            dependencies[str(mask)] = digest(mask)
    git_revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'],
                                           cwd=str(ROOT),
                                           text=True).strip()
    git_status = subprocess.check_output(['git', 'status', '--short'],
                                         cwd=str(ROOT),
                                         text=True)
    write_json(
        run_dir / 'run.json',
        dict(started_utc=datetime.datetime.now(
            datetime.timezone.utc).isoformat(),
             interpreter=sys.executable,
             torch=torch.__version__,
             mmcv=mmcv.__version__,
             mmdet=mmdet.__version__,
             cuda=torch.version.cuda,
             seed=0,
             gpu_id=args.gpu_id,
             checkpoint=str(checkpoint),
             checkpoint_sha256=checkpoint_hash,
             manifest_sha256=digest(MANIFEST),
             git_revision=git_revision,
             git_status=git_status,
             dependency_sha256=dependencies,
             scenarios=selected))
    results = []
    exit_code = 0
    for row in selected:
        output = run_dir / row['name']
        output.mkdir()
        cmd = command_for(row, checkpoint, output, args.gpu_id)
        config = Config.fromfile(str(ROOT / row['config']))
        config.dump(str(output / 'config.py'))
        write_json(output / 'scenario.json', dict(scenario=row, command=cmd))
        print('Running {} -> {}'.format(row['name'], output), flush=True)
        started = time.monotonic()
        with (output / 'evaluation.log').open('x') as log:
            completed = subprocess.run(cmd,
                                       cwd=str(ROOT),
                                       stdout=log,
                                       stderr=subprocess.STDOUT)
        record = dict(name=row['name'],
                      corruption=row['corruption'],
                      parameters=row['parameters'],
                      checkpoint=str(checkpoint),
                      checkpoint_sha256=checkpoint_hash,
                      returncode=completed.returncode,
                      runtime_seconds=time.monotonic() - started,
                      output_directory=str(output),
                      mAP=None,
                      NDS=None)
        metrics = list((output / 'predictions').rglob('metrics_summary.json'))
        if completed.returncode == 0 and len(metrics) == 1:
            with metrics[0].open() as stream:
                summary = json.load(stream)
            record.update(mAP=summary['mean_ap'],
                          NDS=summary['nd_score'],
                          metrics_file=str(metrics[0]))
        else:
            record[
                'error'] = 'Evaluation failed or exactly one metrics_summary.json was not produced; inspect evaluation.log'
            exit_code = completed.returncode or 1
        write_json(output / 'result.json', record)
        results.append(record)
        print('{}: mAP={} NDS={} elapsed={:.1f}s'.format(
            row['name'], record['mAP'], record['NDS'],
            record['runtime_seconds']),
              flush=True)
        if exit_code:
            break
    clean = next(
        (r for r in results if r['name'] == 'clean' and r['mAP'] is not None),
        None)
    if clean:
        for record in results:
            if record['mAP'] is None:
                continue
            for metric in ['mAP', 'NDS']:
                record[metric +
                       '_drop_pp'] = 100 * (clean[metric] - record[metric])
                record[metric + '_relative_drop_percent'] = (
                    100 * (clean[metric] - record[metric]) /
                    clean[metric] if clean[metric] else None)
    write_json(run_dir / 'results.json', results)
    print('Results:', run_dir / 'results.json')
    return exit_code


if __name__ == '__main__':
    sys.exit(main())

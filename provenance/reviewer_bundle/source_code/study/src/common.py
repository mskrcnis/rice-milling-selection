import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False, default=str) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def stage(name, status, command=None, inputs=(), outputs=(), issues=(), next_action=''):
    path = ROOT / 'stage_ledger.json'
    ledger = json.loads(path.read_text())
    ledger['stages'][name] = {
        'status': status, 'updated_utc': now(), 'command': command,
        'inputs': {str(Path(p).resolve()): digest(p) for p in inputs},
        'outputs': [str(Path(p).resolve()) for p in outputs],
        'validation_outcome': status, 'unresolved_issues': list(issues),
        'next_action': next_action,
    }
    write_json(path, ledger)


def freeze(paths, destination, metadata):
    if Path(destination).exists():
        raise FileExistsError('An immutable freeze already exists: ' + str(destination))
    write_json(destination, {'created_utc': now(), 'metadata': metadata,
                            'files': {str(Path(p).resolve()): digest(p) for p in paths}})


def verify_freeze(path):
    content = json.loads(Path(path).read_text())
    for name, expected in content['files'].items():
        if digest(name) != expected:
            raise ValueError('Frozen input changed: ' + name)
    return len(content['files'])

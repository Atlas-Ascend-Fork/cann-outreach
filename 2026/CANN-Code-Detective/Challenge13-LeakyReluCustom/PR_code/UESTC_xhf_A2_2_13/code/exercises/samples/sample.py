import argparse
import json
from pathlib import Path

import numpy as np
import reference


ROOT = Path(__file__).resolve().parent

CASES = [
    json.loads((ROOT / name).read_text(encoding='utf-8'))
    for name in ('case.json', 'case-tail.json')
]
OUTPUTS = [case['outputs'][0]['tensors'][0] for case in CASES]


def generate(directory):
    rng = np.random.default_rng(42)
    for name in ('input', 'output', 'golden'):
        (directory / name).mkdir(parents=True, exist_ok=True)
    for case, output in zip(CASES, OUTPUTS):
        name = output['file']
        (directory / 'output' / name).unlink(missing_ok=True)
        (directory / 'golden' / name).unlink(missing_ok=True)
        x = case['inputs'][0]['tensors'][0]
        values = rng.uniform(*x['range'], x['shape']).astype(x['dtype'])
        (directory / 'input' / x['file']).write_bytes(values.tobytes())
        result = reference.impl(values, **case['attributes'])
        if result.dtype != np.dtype(output['dtype']) or list(result.shape) != output['shape']:
            raise ValueError('reference dtype/shape mismatch: ' + name)
        (directory / 'golden' / name).write_bytes(result.tobytes())
    print('Sample data generated.')


def verify(directory):
    passed = True
    expected = {item['file'] for item in OUTPUTS}
    for folder in ('output', 'golden'):
        actual = {p.name for p in (directory / folder).iterdir()}
        if actual != expected:
            print('VERIFY ERROR: unexpected or missing files in ' + folder)
            return 1
    for item in OUTPUTS:
        name = item['file']
        dtype = np.dtype(item['dtype'])
        size, rtol, atol, tol = (item[key] for key in ('bytes', 'rtol', 'atol', 'tol'))
        try:
            raw = (directory / 'output' / name).read_bytes()
            golden = (directory / 'golden' / name).read_bytes()
            if len(raw) != size or len(golden) != size:
                raise ValueError('exact byte size mismatch')
            a, b = [np.frombuffer(v, dtype=dtype) for v in (raw, golden)]
            close = np.isclose(
                a.astype(np.float64),
                b.astype(np.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True
            )
            bad = int(np.count_nonzero(~close))
            ok = bad <= tol * a.size
            print(
                ('PASS' if ok else 'Wrong Answer')
                + ': ' + name + f', mismatched={bad}/{a.size}'
            )
            passed &= ok
        except (OSError, ValueError) as error:
            print('VERIFY ERROR: ' + name + ': ' + str(error))
            passed = False
    print('PASSED' if passed else 'FAILED')
    return 0 if passed else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('generate', 'verify'))
    parser.add_argument('--directory', type=Path, default=ROOT / 'build-npu')
    args = parser.parse_args()
    try:
        if args.action == 'generate':
            generate(args.directory)
        else:
            raise SystemExit(verify(args.directory))
    except Exception as error:
        print('SAMPLE ERROR: ' + str(error))
        raise SystemExit(2)

"""Extract the two bundled models' cache laws and evaluate their stored bytes.

Run with HuggingArch installed at the revision recorded in result.json.
The published data uses only local snapshots and model-native cache formats.
"""
import json
import os
from pathlib import Path
import subprocess

os.environ['HF_HUB_OFFLINE'] = '1'

from backend import analyzer
from backend.arch.eval.cache_view import cache_view_from_analysis
from backend.arch.eval.program import Program
from backend.arch.eval.transforms import Deployment

CHECKOUT = Path(analyzer.__file__).resolve().parents[1]
CONTEXTS = [128, 4096, 4097, 4098, 8192, 32768]
MODELS = ['meta-llama/Llama-3.2-1B', 'deepseek-ai/DeepSeek-V4.1-Flash']


def measure(model_id):
    result = analyzer.analyze_model(model_id, batch_size=1, seq_len=4096,
                                    dtype='bfloat16', trust_remote_code=False)
    if not result['success']:
        raise RuntimeError(result.get('message', result.get('error')))
    program = Program.of(result[analyzer.RESOLVED_KEY])
    deployment = Deployment(program).at(batch=1, input_tokens=4096, logit_positions=1)
    cache = cache_view_from_analysis(deployment.analyze())
    totals = result['composition']['totals']
    growth, bounded = cache.growth_and_floor()
    assert (growth, bounded) == (totals['kv_per_token_bytes'], totals['kv_constant_bytes'])
    contexts = CONTEXTS + ([1_000_000] if model_id.startswith('deepseek-ai/') else [])
    snapshot = Path('backend/arch/snapshots') / (model_id.replace('/', '__') + '.json')
    header = json.loads((CHECKOUT / snapshot).read_text())['metadata']['header']
    return {
        'model_id': model_id,
        'spec_name': result['spec_name'],
        'snapshot_path': snapshot.as_posix(),
        'checkpoint_revision': header['revision'],
        'layers': result['summary']['num_layers'],
        'checkpoint_byte_difference': result['summary']['all_tensor_diff'],
        'cache_formats': totals['kv_dtypes'],
        'average_bytes_per_token': int(growth),
        'bounded_bytes_at_full_window': int(bounded),
        'stored_bytes': {n: int(cache.storage_bytes_at(n)) for n in contexts},
        'storage_rows_at_4096': cache.storage_rows(4096),
    }


if __name__ == '__main__':
    commit = subprocess.check_output(['git', '-C', str(CHECKOUT), 'rev-parse', 'HEAD'], text=True).strip()
    payload = {'huggingarch_commit': commit, 'batch_size': 1, 'analysis_seq_len': 4096,
               'cli_dtype_argument': 'bfloat16', 'cache_scope': 'model backbone, native formats, one rank',
               'models': [measure(name) for name in MODELS]}
    print(json.dumps(payload, indent=2))

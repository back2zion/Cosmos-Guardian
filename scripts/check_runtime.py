"""Probe installed inference modules, real CUDA execution and official model access.

No tokens, account details, or synthetic model results are written to the report.
"""
import argparse
import importlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check_runtime():
    report = {'checked_at': datetime.now(timezone.utc).isoformat(), 'python': platform.python_version(),
              'inference_performed': False, 'dependencies': {}, 'cuda': {}, 'model_config_access': {}}
    for name in ('torch', 'torchvision', 'transformers', 'accelerate', 'peft', 'yaml', 'PIL', 'cv2', 'qwen_vl_utils'):
        try:
            module = importlib.import_module(name)
            report['dependencies'][name] = {'imported': True, 'version': getattr(module, '__version__', None)}
        except Exception as exc:  # noqa: BLE001 - capture actual import incompatibilities for the readiness report
            report['dependencies'][name] = {'imported': False, 'error_type': type(exc).__name__, 'error': str(exc)}
    try:
        import torch
        if not torch.cuda.is_available():
            report['cuda'] = {'available': False}
        else:
            left = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
            result = left @ left
            torch.cuda.synchronize()
            report['cuda'] = {
                'available': True, 'device': torch.cuda.get_device_name(0),
                'total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
                'torch_cuda_version': torch.version.cuda,
                'bf16_matrix_multiply_passed': bool(torch.all(result == 32).item()),
            }
            del left, result
            torch.cuda.empty_cache()
    except Exception as exc:  # noqa: BLE001 - report driver, runtime and device errors without claiming inference
        report['cuda'] = {'available': False, 'error_type': type(exc).__name__, 'error': str(exc)}
    try:
        from huggingface_hub import get_token, hf_hub_download
        report['model_config_access']['authenticated'] = bool(get_token())
        path = hf_hub_download('nvidia/Cosmos-Reason2-2B', 'config.json', cache_dir=ROOT / 'data/huggingface')
        report['model_config_access'].update(accessible=True, config_path=path)
    except Exception as exc:  # noqa: BLE001 - never include exception text that could expose request credentials
        response = getattr(exc, 'response', None)
        report['model_config_access'].update(accessible=False, error_type=type(exc).__name__,
                                             http_status=getattr(response, 'status_code', None))
    report['ready_for_model_loading'] = (
        all(item['imported'] for item in report['dependencies'].values())
        and report['cuda'].get('bf16_matrix_multiply_passed', False)
        and report['model_config_access'].get('accessible', False)
    )
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/runtime-readiness.json')
    args = parser.parse_args()
    report = check_runtime()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report['ready_for_model_loading'] else 2


if __name__ == '__main__':
    raise SystemExit(main())

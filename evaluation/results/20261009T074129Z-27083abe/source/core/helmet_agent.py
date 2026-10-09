"""Measured image-only product profile, separate from the legacy physics prompt."""
import time
from pathlib import Path

import torch
import yaml

from .agent import CosmosGuardianAgent
from .helmet import parse_classification


class HelmetAgent(CosmosGuardianAgent):
    profile = 'helmet-v1'
    model_revision = '9ce19a195e423419c349abfc86fd07178b230561'

    def load_model(self, adapter_path=None):
        if adapter_path:
            raise ValueError('The measured product profile uses the pinned base model only')
        super().load_model(revision=self.model_revision)

    def _load_prompts(self):
        return yaml.safe_load((Path(__file__).resolve().parents[1] / 'prompts/helmet_inspector.yaml').read_text())

    def analyze_media(self, media_path, safety_context='Helmet inspection'):
        if Path(media_path).suffix.lower() not in {'.png', '.jpg', '.jpeg', '.webp'}:
            raise ValueError('안전모 판별 v1은 이미지만 지원합니다. 영상은 이미지로 추출해 검토하세요.')
        if self.use_label_grounding:
            raise ValueError('Product profile does not allow label grounding')
        start = time.perf_counter()
        if self.model is None:
            yield {'stage': 'model_loading', 'detail': '안전모 판별 모델 준비 중'}
            self.load_model()
        conversation = [
            {'role': 'system', 'content': [{'type': 'text', 'text': self.prompts['system_instructions']}]},
            {'role': 'user', 'content': [
                {'type': 'image', 'image': str(media_path)},
                {'type': 'text', 'text': self.prompts['user_prompt_template']},
            ]},
        ]
        inputs = self.processor.apply_chat_template(
            conversation, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt',
        ).to(self.model.device)
        if 'pixel_values' not in inputs or not inputs['pixel_values'].numel():
            raise ValueError('Image pixels were not supplied to the model')
        yield {'stage': 'inference', 'detail': '안전모 착용 여부 판별 중'}
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=8, do_sample=False, use_cache=True,
                                         repetition_penalty=1.0)
        text = self.processor.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        try:
            result = parse_classification(text)
        except (ValueError, TypeError):
            result = {'error': 'Invalid helmet classification answer', 'raw_output': text,
                      'assessment_status': 'unknown', 'profile': self.profile}
        result['inference'] = {'input_tokens': int(inputs['input_ids'].shape[1]),
                               'output_tokens': int(output.shape[1] - inputs['input_ids'].shape[1]),
                               'elapsed_seconds': round(time.perf_counter() - start, 3)}
        yield {'stage': 'complete', 'result': result}

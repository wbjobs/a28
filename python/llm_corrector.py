import os
import re
import json
import logging
from typing import List, Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class LLMCorrector:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                 use_gpu: bool = False, use_vllm: bool = False,
                 vllm_url: str = "http://localhost:8000"):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url
        self.model = None
        self.tokenizer = None
        self._load_model()

    def _load_model(self):
        if self.use_vllm:
            self._init_vllm_client()
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            device = "cuda" if self.use_gpu else "cpu"
            logger.info(f"[INFO] Loading LLM corrector model: {self.model_name} on {device}")

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else device,
                trust_remote_code=True
            )

            self.model.eval()
            self._model_loaded = True
            logger.info(f"[INFO] LLM corrector loaded successfully: {self.model_name}")

        except Exception as e:
            logger.warning(f"[WARN] Failed to load LLM corrector ({self.model_name}): {e}")
            logger.warning("[WARN] LLM correction disabled, using rule-based fallback")
            self.model = None
            self.tokenizer = None
            self._model_loaded = False

    def _init_vllm_client(self):
        try:
            import requests
            resp = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                self._vllm_available = True
                self._model_loaded = True
                logger.info(f"[INFO] vLLM server available at {self.vllm_url}")
            else:
                raise ConnectionError(f"vLLM returned status {resp.status_code}")
        except Exception as e:
            logger.warning(f"[WARN] vLLM server not available: {e}, using rule-based fallback")
            self._vllm_available = False
            self._model_loaded = False

    def correct_segments(self, segments: List[Dict], context: str = "") -> List[Dict]:
        if not segments:
            return segments

        if self._model_loaded:
            return self._llm_correct(segments, context)

        return self._rule_based_correct(segments)

    def _llm_correct(self, segments: List[Dict], context: str = "") -> List[Dict]:
        if self.use_vllm and getattr(self, '_vllm_available', False):
            return self._vllm_correct(segments, context)

        corrected = []
        for seg in segments:
            text = seg.get("text", "")
            if not text.strip():
                corrected.append(seg)
                continue

            prompt = self._build_prompt(text, context)

            try:
                inputs = self.tokenizer(prompt, return_tensors="pt")
                if self.use_gpu:
                    inputs = {k: v.cuda() for k, v in inputs.items()}

                with __import__('torch').no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        top_p=0.9,
                        do_sample=False,
                        pad_token_id=self.tokenizer.eos_token_id
                    )

                response = self.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                ).strip()

                corrected_text = self._parse_response(response, text)
            except Exception as e:
                logger.warning(f"[WARN] LLM correction failed for segment: {e}")
                corrected_text = text

            corrected_seg = dict(seg)
            corrected_seg["original_text"] = text
            corrected_seg["text"] = corrected_text
            corrected_seg["corrected"] = corrected_text != text
            corrected.append(corrected_seg)

        return corrected

    def _vllm_correct(self, segments: List[Dict], context: str = "") -> List[Dict]:
        import requests

        corrected = []
        for seg in segments:
            text = seg.get("text", "")
            if not text.strip():
                corrected.append(seg)
                continue

            prompt = self._build_prompt(text, context)

            try:
                payload = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 256,
                    "temperature": 0.1
                }
                resp = requests.post(
                    f"{self.vllm_url}/v1/chat/completions",
                    json=payload,
                    timeout=30
                )
                if resp.status_code == 200:
                    response = resp.json()["choices"][0]["message"]["content"].strip()
                    corrected_text = self._parse_response(response, text)
                else:
                    corrected_text = text
            except Exception as e:
                logger.warning(f"[WARN] vLLM correction failed: {e}")
                corrected_text = text

            corrected_seg = dict(seg)
            corrected_seg["original_text"] = text
            corrected_seg["text"] = corrected_text
            corrected_seg["corrected"] = corrected_text != text
            corrected.append(corrected_seg)

        return corrected

    def _build_prompt(self, text: str, context: str = "") -> str:
        context_part = ""
        if context:
            context_part = f"\n上下文信息：{context}\n"

        prompt = (
            f"你是一个语音识别纠错助手。以下是一段语音识别（ASR）的输出文本，"
            f"可能包含错别字、同音字错误、漏字或多字。请根据语境进行纠错，"
            f"只输出纠错后的文本，不要添加解释或标点符号修改。"
            f"如果文本没有错误，原样返回即可。{context_part}\n"
            f"识别文本：{text}\n"
            f"纠错结果："
        )
        return prompt

    def _parse_response(self, response: str, original: str) -> str:
        response = response.strip()

        prefixes = ["纠错结果：", "纠错结果:", "修正：", "修正:", "纠正后：", "纠正后:"]
        for prefix in prefixes:
            if response.startswith(prefix):
                response = response[len(prefix):].strip()

        if not response or len(response) > len(original) * 3:
            return original

        return response

    def _rule_based_correct(self, segments: List[Dict]) -> List[Dict]:
        common_errors = {
            "的 地": "地",
            "的 得": "得",
            "在 再": "再",
            "做 作": "做",
            "和 河": "河",
            "事 是": "是",
            "里 理": "里",
            "他 她": "他",
            "已 以": "已",
            "这 着": "着",
            "进 近": "进",
            "到 道": "到",
            "力 立": "力",
            "公 工": "公",
            "车 扯": "车",
            "数 树": "树",
            "花 话": "话",
            "加速 家速": "加速",
            "汽车 气车": "汽车",
            "交通 交同": "交通",
            "行人 形人": "行人",
            "路口 路扣": "路口",
            "风景 风井": "风景",
            "会议室 会已室": "会议室",
            "讨论 图论": "讨论",
            "产品 产平": "产品",
            "体验 体燕": "体验",
            "参与 参于": "参与",
        }

        corrected = []
        for seg in segments:
            text = seg.get("text", "")
            original_text = text
            was_corrected = False

            for wrong_pair, correct in common_errors.items():
                wrongs = wrong_pair.split()
                if len(wrongs) == 2:
                    if wrongs[1] in text:
                        text = text.replace(wrongs[1], correct)
                        was_corrected = True

            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            text = re.sub(r'\s{2,}', ' ', text)

            corrected_seg = dict(seg)
            corrected_seg["original_text"] = original_text
            corrected_seg["text"] = text.strip()
            corrected_seg["corrected"] = was_corrected
            corrected.append(corrected_seg)

        return corrected


class BatchLLMCorrector(LLMCorrector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._batch_size = 4

    def correct_segments(self, segments: List[Dict], context: str = "") -> List[Dict]:
        if not segments:
            return segments

        if not self._model_loaded:
            return self._rule_based_correct(segments)

        if self.use_vllm and getattr(self, '_vllm_available', False):
            return self._vllm_batch_correct(segments, context)

        return self._llm_correct(segments, context)

    def _vllm_batch_correct(self, segments: List[Dict], context: str = "") -> List[Dict]:
        import requests

        corrected = []
        batch_payloads = []

        for seg in segments:
            text = seg.get("text", "")
            if not text.strip():
                corrected.append(seg)
                continue

            prompt = self._build_prompt(text, context)
            batch_payloads.append({
                "model": self.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 256,
                "temperature": 0.1
            })

        for i in range(0, len(batch_payloads), self._batch_size):
            batch = batch_payloads[i:i + self._batch_size]
            batch_segments = segments[i:i + self._batch_size]

            for j, (payload, seg) in enumerate(zip(batch, batch_segments)):
                text = seg.get("text", "")
                try:
                    resp = requests.post(
                        f"{self.vllm_url}/v1/chat/completions",
                        json=payload,
                        timeout=30
                    )
                    if resp.status_code == 200:
                        response = resp.json()["choices"][0]["message"]["content"].strip()
                        corrected_text = self._parse_response(response, text)
                    else:
                        corrected_text = text
                except Exception as e:
                    logger.warning(f"[WARN] vLLM batch correction failed: {e}")
                    corrected_text = text

                corrected_seg = dict(seg)
                corrected_seg["original_text"] = text
                corrected_seg["text"] = corrected_text
                corrected_seg["corrected"] = corrected_text != text
                corrected.append(corrected_seg)

        return corrected

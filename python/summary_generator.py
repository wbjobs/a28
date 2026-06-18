import json
import logging
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SmartSummary:
    summary: str
    keywords: List[str]
    scene_summary: str
    speech_summary: str
    objects_summary: str


class SummaryGenerator:
    def __init__(self, model_name: str = "Qwen/Qwen2-7B-Instruct",
                 use_gpu: bool = False, use_vllm: bool = False,
                 vllm_url: str = "http://localhost:8000"):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self.use_vllm = use_vllm
        self.vllm_url = vllm_url
        self.model = None
        self.tokenizer = None
        self._model_loaded = False
        self._load_model()

    def _load_model(self):
        if self.use_vllm:
            self._init_vllm()
            return

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch

            logger.info(f"[SUMMARY] Loading model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.use_gpu else torch.float32,
                device_map="auto" if self.use_gpu else "cpu",
                trust_remote_code=True
            )
            self.model.eval()
            self._model_loaded = True
            logger.info(f"[SUMMARY] Model loaded: {self.model_name}")
        except Exception as e:
            logger.warning(f"[SUMMARY] Failed to load model: {e}, using rule-based fallback")
            self._model_loaded = False

    def _init_vllm(self):
        try:
            import requests
            resp = requests.get(f"{self.vllm_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                self._vllm_available = True
                self._model_loaded = True
                logger.info(f"[SUMMARY] vLLM server available at {self.vllm_url}")
            else:
                raise ConnectionError()
        except Exception as e:
            logger.warning(f"[SUMMARY] vLLM not available: {e}")
            self._vllm_available = False
            self._model_loaded = False

    def generate(self, timeline: List[Dict],
                 speech_segments: Optional[List[Dict]] = None) -> SmartSummary:
        scene_info = self._extract_scene_info(timeline)
        speech_info = self._extract_speech_info(timeline, speech_segments)
        objects_info = self._extract_objects_info(timeline)

        if self._model_loaded:
            return self._llm_generate(scene_info, speech_info, objects_info)

        return self._rule_based_generate(scene_info, speech_info, objects_info)

    def _extract_scene_info(self, timeline: List[Dict]) -> Dict:
        scene_counts = {}
        scene_transitions = []
        prev_scene = None

        for entry in timeline:
            scene = entry.get("scene", "unknown")
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
            if scene != prev_scene and prev_scene is not None:
                scene_transitions.append({
                    "from": prev_scene,
                    "to": scene,
                    "at": entry.get("timestamp", 0)
                })
            prev_scene = scene

        total = sum(scene_counts.values()) if scene_counts else 1
        scene_distribution = {
            k: {"count": v, "percentage": round(v / total * 100, 1)}
            for k, v in sorted(scene_counts.items(), key=lambda x: -x[1])
        }

        return {
            "distribution": scene_distribution,
            "transitions": scene_transitions[:10],
            "dominant_scene": max(scene_counts, key=scene_counts.get) if scene_counts else "unknown",
            "scene_count": len(scene_counts)
        }

    def _extract_speech_info(self, timeline: List[Dict],
                             speech_segments: Optional[List[Dict]] = None) -> Dict:
        all_speech = []
        for entry in timeline:
            speech = entry.get("speech", "")
            if speech:
                all_speech.append(speech)

        if speech_segments:
            for seg in speech_segments:
                text = seg.get("text", "")
                if text and text not in all_speech:
                    all_speech.append(text)

        full_text = " ".join(all_speech)
        words = full_text.split()
        sentences = [s.strip() for s in re.split(r'[。！？.!?\n]', full_text) if s.strip()]

        return {
            "full_text": full_text,
            "word_count": len(words),
            "sentence_count": len(sentences),
            "sentences": sentences[:30],
            "has_speech": len(all_speech) > 0
        }

    def _extract_objects_info(self, timeline: List[Dict]) -> Dict:
        object_counts = {}
        object_timestamps = {}

        for entry in timeline:
            for obj in entry.get("objects", []):
                cls = obj.get("class", "")
                if cls:
                    object_counts[cls] = object_counts.get(cls, 0) + 1
                    if cls not in object_timestamps:
                        object_timestamps[cls] = []
                    object_timestamps[cls].append(entry.get("timestamp", 0))

        return {
            "counts": dict(sorted(object_counts.items(), key=lambda x: -x[1])),
            "unique_objects": len(object_counts),
            "top_objects": list(object_counts.keys())[:15],
            "timestamps": {k: v[:5] for k, v in object_timestamps.items()}
        }

    def _llm_generate(self, scene_info: Dict, speech_info: Dict,
                      objects_info: Dict) -> SmartSummary:
        prompt = self._build_prompt(scene_info, speech_info, objects_info)

        try:
            if self.use_vllm and getattr(self, '_vllm_available', False):
                response = self._vllm_call(prompt)
            else:
                response = self._local_call(prompt)

            return self._parse_response(response, scene_info, speech_info, objects_info)
        except Exception as e:
            logger.warning(f"[SUMMARY] LLM generation failed: {e}, falling back to rules")
            return self._rule_based_generate(scene_info, speech_info, objects_info)

    def _build_prompt(self, scene_info: Dict, speech_info: Dict,
                      objects_info: Dict) -> str:
        scene_parts = []
        for scene, info in list(scene_info["distribution"].items())[:5]:
            scene_parts.append(f"- {scene}: {info['percentage']}%")

        objects_str = "、".join(objects_info["top_objects"][:10]) if objects_info["top_objects"] else "无"
        speech_str = speech_info["full_text"][:500] if speech_info["has_speech"] else "无对话内容"

        prompt = (
            f"请根据以下视频分析数据，生成一段简洁的文字摘要（100字以内）和5个关键词标签。\n\n"
            f"场景分布：\n" + "\n".join(scene_parts) + "\n\n"
            f"检测到的物体：{objects_str}\n\n"
            f"对话内容：{speech_str}\n\n"
            f"请严格按以下JSON格式输出：\n"
            f'{{"summary": "摘要内容", "keywords": ["关键词1", "关键词2", "关键词3", "关键词4", "关键词5"]}}'
        )
        return prompt

    def _local_call(self, prompt: str) -> str:
        import torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.use_gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=300,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        return self.tokenizer.decode(
            outputs[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()

    def _vllm_call(self, prompt: str) -> str:
        import requests
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.3
        }
        resp = requests.post(
            f"{self.vllm_url}/v1/chat/completions",
            json=payload, timeout=30
        )
        if resp.status_code == 200:
            return resp.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"vLLM returned {resp.status_code}")

    def _parse_response(self, response: str, scene_info: Dict,
                        speech_info: Dict, objects_info: Dict) -> SmartSummary:
        try:
            json_match = re.search(r'\{[^{}]*"summary"[^{}]*"keywords"[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                summary = data.get("summary", "")
                keywords = data.get("keywords", [])
                if len(keywords) > 5:
                    keywords = keywords[:5]
                while len(keywords) < 5:
                    keywords.append(self._generate_keyword(scene_info, objects_info, len(keywords)))
            else:
                summary = response[:100]
                keywords = self._extract_keywords_from_text(response, scene_info, objects_info)
        except json.JSONDecodeError:
            summary = response[:100]
            keywords = self._extract_keywords_from_text(response, scene_info, objects_info)

        scene_summary = self._make_scene_summary(scene_info)
        speech_summary = self._make_speech_summary(speech_info)
        objects_summary = self._make_objects_summary(objects_info)

        return SmartSummary(
            summary=summary,
            keywords=keywords,
            scene_summary=scene_summary,
            speech_summary=speech_summary,
            objects_summary=objects_summary
        )

    def _rule_based_generate(self, scene_info: Dict, speech_info: Dict,
                            objects_info: Dict) -> SmartSummary:
        dominant = scene_info["dominant_scene"]
        scene_labels = {
            "outdoor_nature": "户外自然",
            "city_urban": "城市",
            "indoor_room": "室内",
            "road_traffic": "道路交通",
            "water_activity": "水上活动",
            "air_transport": "航空",
            "sports": "运动",
            "food_kitchen": "餐饮厨房",
            "general": "综合"
        }

        scene_cn = scene_labels.get(dominant, dominant)
        top_objs = objects_info["top_objects"][:5]
        has_speech = speech_info["has_speech"]

        parts = [f"视频主要场景为{scene_cn}"]
        if top_objs:
            obj_cn = "、".join(top_objs[:3])
            parts.append(f"出现了{obj_cn}等物体")
        if has_speech:
            parts.append("包含对话内容")
        if scene_info["scene_count"] > 1:
            parts.append(f"共有{scene_info['scene_count']}种场景切换")

        summary = "，".join(parts) + "。"

        keywords = []
        if dominant != "unknown":
            keywords.append(scene_cn)
        keywords.extend(top_objs[:3])
        if has_speech:
            keywords.append("对话")
        while len(keywords) < 5:
            keywords.append(f"标签{len(keywords) + 1}")

        return SmartSummary(
            summary=summary,
            keywords=keywords[:5],
            scene_summary=self._make_scene_summary(scene_info),
            speech_summary=self._make_speech_summary(speech_info),
            objects_summary=self._make_objects_summary(objects_info)
        )

    def _make_scene_summary(self, scene_info: Dict) -> str:
        dist = scene_info["distribution"]
        parts = []
        for scene, info in list(dist.items())[:3]:
            parts.append(f"{scene}({info['percentage']}%)")
        return " → ".join(parts) if parts else "无场景信息"

    def _make_speech_summary(self, speech_info: Dict) -> str:
        if not speech_info["has_speech"]:
            return "无对话内容"
        return speech_info["full_text"][:200] + ("..." if len(speech_info["full_text"]) > 200 else "")

    def _make_objects_summary(self, objects_info: Dict) -> str:
        top = objects_info["top_objects"][:10]
        return "、".join(top) if top else "无检测物体"

    def _extract_keywords_from_text(self, text: str, scene_info: Dict,
                                    objects_info: Dict) -> List[str]:
        keywords = []
        for obj in objects_info["top_objects"][:3]:
            if obj not in keywords:
                keywords.append(obj)
        if scene_info["dominant_scene"] != "unknown":
            keywords.append(scene_info["dominant_scene"])

        chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
        for w in chinese_words:
            if w not in keywords and len(keywords) < 5:
                keywords.append(w)

        while len(keywords) < 5:
            keywords.append(f"标签{len(keywords) + 1}")
        return keywords[:5]

    def _generate_keyword(self, scene_info, objects_info, idx) -> str:
        pool = objects_info["top_objects"] + [scene_info["dominant_scene"]]
        if idx < len(pool):
            return pool[idx]
        return f"标签{idx + 1}"

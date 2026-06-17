import numpy as np
import re
from typing import List, Dict, Tuple
from dataclasses import dataclass, field


@dataclass
class TimelineSegment:
    timestamp: float
    duration: float
    scene: str
    objects: List[Dict] = field(default_factory=list)
    classifications: List[Dict] = field(default_factory=list)
    speech: str = ""

    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "duration": self.duration,
            "scene": self.scene,
            "objects": self.objects,
            "classifications": self.classifications,
            "speech": self.speech
        }


class QueryParser:
    OBJECT_KEYWORDS = {
        "车": ["car", "truck", "bus", "bicycle", "motorcycle", "vehicle", "汽车", "卡车", "公交车", "自行车", "摩托车"],
        "人": ["person", "人物", "人", "行人"],
        "狗": ["dog", "小狗", "狗狗"],
        "猫": ["cat", "小猫", "猫咪"],
        "食物": ["food", "pizza", "cake", "sandwich", "hot dog", "donut", "食物", "披萨", "蛋糕"],
        "树": ["tree", "植物", "树"],
        "水": ["water", "boat", "ship", "湖", "海", "河", "水"],
        "建筑": ["building", "house", "church", "castle", "建筑", "房子", "教堂"],
        "手机": ["cell phone", "手机", "电话"],
        "电脑": ["laptop", "keyboard", "mouse", "电脑", "笔记本"],
    }

    SCENE_KEYWORDS = {
        "户外": ["outdoor", "nature", "户外", "室外", "野外", "自然"],
        "城市": ["city", "urban", "road", "street", "城市", "街道", "道路"],
        "室内": ["indoor", "room", "室内", "房间", "办公室"],
        "运动": ["sports", "运动", "体育"],
        "水": ["water", "湖", "海", "河"],
        "飞行": ["air", "plane", "flight", "天空", "飞行"],
    }

    @classmethod
    def parse(cls, query: str) -> Dict:
        result = {
            "objects": [],
            "scenes": [],
            "keywords": [],
            "exact_phrases": []
        }

        phrases = re.findall(r'[""]([^""]+)[""]', query)
        result["exact_phrases"] = [p.strip() for p in phrases]

        query_clean = re.sub(r'[""][^""]*[""]', '', query)

        for keyword, aliases in cls.OBJECT_KEYWORDS.items():
            for alias in aliases:
                if alias.lower() in query_clean.lower() or alias in query_clean:
                    if keyword not in result["objects"]:
                        result["objects"].append(keyword)
                    break

        for keyword, aliases in cls.SCENE_KEYWORDS.items():
            for alias in aliases:
                if alias.lower() in query_clean.lower() or alias in query_clean:
                    if keyword not in result["scenes"]:
                        result["scenes"].append(keyword)
                    break

        words = re.findall(r'[\w\u4e00-\u9fff]+', query_clean)
        stop_words = {"的", "和", "与", "在", "有", "包含", "找到", "查找", "搜索",
                      "找", "片段", "视频", "这个", "那个", "我想", "请", "帮我",
                      "the", "and", "or", "with", "in", "on", "at", "find", "search",
                      "for", "show", "me", "where", "is", "that", "this", "video",
                      "clip", "segment", "containing", "has", "have", "a", "an"}
        result["keywords"] = [w for w in words if w.lower() not in stop_words and len(w) > 1]

        return result

    @classmethod
    def build_text_representation(cls, segment: Dict) -> str:
        parts = []

        parts.append(f"场景: {segment.get('scene', 'unknown')}")

        objects = segment.get('objects', [])
        if objects:
            obj_names = [obj.get('class', '') for obj in objects]
            parts.append(f"物体: {', '.join(obj_names)}")

        classifications = segment.get('classifications', [])
        if classifications:
            top_labels = [c.get('label', '') for c in classifications[:3]]
            parts.append(f"图像内容: {', '.join(top_labels)}")

        speech = segment.get('speech', '')
        if speech:
            parts.append(f"对话: {speech}")

        return " ".join(parts)

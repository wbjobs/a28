import os
import re
from typing import List, Dict, Optional
from pathlib import Path
from dataclasses import dataclass


@dataclass
class SRTEntry:
    index: int
    start_time: float
    end_time: float
    text: str
    original_text: Optional[str] = None

    def _format_timestamp(self, seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    def to_srt_string(self) -> str:
        start_str = self._format_timestamp(self.start_time)
        end_str = self._format_timestamp(self.end_time)

        lines = [
            str(self.index),
            f"{start_str} --> {end_str}",
            self.text
        ]

        if self.original_text and self.original_text != self.text:
            lines.append(f"(原文: {self.original_text})")

        return "\n".join(lines)

    def to_vtt_string(self) -> str:
        start_str = self._format_vtt_timestamp(self.start_time)
        end_str = self._format_vtt_timestamp(self.end_time)

        lines = [
            f"{start_str} --> {end_str}",
            self.text
        ]

        if self.original_text and self.original_text != self.text:
            lines.append(f"(原文: {self.original_text})")

        return "\n".join(lines)

    def _format_vtt_timestamp(self, seconds: float) -> str:
        if seconds < 0:
            seconds = 0
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


class SRTGenerator:
    def __init__(self, output_dir: str = "./results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_from_segments(self, segments: List[Dict],
                               video_id: str,
                               include_original: bool = True) -> str:
        entries = self._segments_to_entries(segments)
        srt_content = self._render_srt(entries, include_original)

        srt_path = self.output_dir / f"{video_id}.srt"
        with open(srt_path, 'w', encoding='utf-8') as f:
            f.write(srt_content)

        vtt_content = self._render_vtt(entries, include_original)
        vtt_path = self.output_dir / f"{video_id}.vtt"
        with open(vtt_path, 'w', encoding='utf-8') as f:
            f.write(vtt_content)

        json_path = self.output_dir / f"{video_id}_subtitles.json"
        subtitles_json = []
        for entry in entries:
            subtitles_json.append({
                "index": entry.index,
                "start": entry.start_time,
                "end": entry.end_time,
                "text": entry.text,
                "original_text": entry.original_text,
                "start_formatted": entry._format_timestamp(entry.start_time),
                "end_formatted": entry._format_timestamp(entry.end_time)
            })
        with open(json_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(subtitles_json, f, ensure_ascii=False, indent=2)

        return str(srt_path)

    def _segments_to_entries(self, segments: List[Dict]) -> List[SRTEntry]:
        entries = []
        for i, seg in enumerate(segments):
            start = seg.get("start", 0.0)
            end = seg.get("end", start + 1.0)
            text = seg.get("text", "").strip()
            original = seg.get("original_text", None)

            if original and original == text:
                original = None

            if not text:
                continue

            entries.append(SRTEntry(
                index=i + 1,
                start_time=start,
                end_time=end,
                text=text,
                original_text=original
            ))

        for i in range(len(entries)):
            entries[i].index = i + 1

        return entries

    def _render_srt(self, entries: List[SRTEntry],
                    include_original: bool = True) -> str:
        blocks = []
        for entry in entries:
            block = entry.to_srt_string() if include_original else \
                f"{entry.index}\n{entry._format_timestamp(entry.start_time)} --> {entry._format_timestamp(entry.end_time)}\n{entry.text}"
            blocks.append(block)

        return "\n\n".join(blocks) + "\n"

    def _render_vtt(self, entries: List[SRTEntry],
                    include_original: bool = True) -> str:
        header = "WEBVTT\n\n"
        blocks = []
        for entry in entries:
            block = entry.to_vtt_string() if include_original else \
                f"{entry._format_vtt_timestamp(entry.start_time)} --> {entry._format_vtt_timestamp(entry.end_time)}\n{entry.text}"
            blocks.append(block)

        return header + "\n\n".join(blocks) + "\n"

    @staticmethod
    def merge_subtitles_into_timeline(timeline: List[Dict],
                                      subtitle_segments: List[Dict]) -> List[Dict]:
        for entry in timeline:
            timestamp = entry.get("timestamp", 0.0)
            matching_texts = []
            for seg in subtitle_segments:
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                if seg_start <= timestamp <= seg_end:
                    matching_texts.append(seg.get("text", ""))

            if matching_texts:
                entry["speech"] = " ".join(t for t in matching_texts if t)

            original = []
            for seg in subtitle_segments:
                seg_start = seg.get("start", 0.0)
                seg_end = seg.get("end", 0.0)
                if seg_start <= timestamp <= seg_end and seg.get("original_text"):
                    original.append(seg["original_text"])

            if original:
                entry["speech_original"] = " ".join(original)

        return timeline

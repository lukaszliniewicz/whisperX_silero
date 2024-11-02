import torch
from silero_vad import get_speech_timestamps, load_silero_vad
import numpy as np
import pandas as pd
from typing import Optional, List
from .utils import interpolate_nans

def load_vad_model(device, vad_onset=0.500, vad_offset=0.363, **kwargs):
    """Load Silero VAD model"""
    model = load_silero_vad(onnx=False)
    model = model.to(device)
    return model

def merge_chunks(segments, chunk_size, onset: float = 0.5, offset: Optional[float] = None):
    """Merge speech segments into chunks"""
    if not segments:
        print("No active speech found in audio")
        return []
    
    merged_segments = []
    curr_start = segments[0]["start"] 
    curr_end = segments[0]["end"]
    seg_idxs = [(segments[0]["start"], segments[0]["end"])]

    for seg in segments[1:]:
        if seg["end"] - curr_start > chunk_size:
            merged_segments.append({
                "start": curr_start,
                "end": curr_end,
                "segments": seg_idxs,
            })
            curr_start = seg["start"]
            curr_end = seg["end"]
            seg_idxs = [(seg["start"], seg["end"])]
        else:
            curr_end = seg["end"]
            seg_idxs.append((seg["start"], seg["end"]))

    # Add final segment
    merged_segments.append({
        "start": curr_start, 
        "end": curr_end,
        "segments": seg_idxs,
    })

    return merged_segments

def get_speech_timestamps_from_audio(audio, model, sample_rate=16000, **vad_kwargs):
    """Get speech timestamps using Silero VAD"""
    if not torch.is_tensor(audio):
        audio = torch.tensor(audio)
    
    # Make sure audio is on same device as model
    device = next(model.parameters()).device
    audio = audio.to(device)
    
    timestamps = get_speech_timestamps(
        audio,
        model,
        sampling_rate=sample_rate,
        threshold=vad_kwargs.get("speech_threshold", 0.5),
        min_silence_duration_ms=vad_kwargs.get("min_silence_duration_ms", 100),
        speech_pad_ms=vad_kwargs.get("speech_pad_ms", 30),
        return_seconds=True
    )
    
    return [{"start": ts["start"], "end": ts["end"]} for ts in timestamps]

def merge_vad(vad_arr, pad_onset=0.0, pad_offset=0.0, min_duration_off=0.0, min_duration_on=0.0):
    """Merge VAD segments with padding"""
    if not vad_arr:
        return pd.DataFrame(columns=['start', 'end'])
        
    segments = []
    for seg in vad_arr:
        segments.append({
            'start': seg['start'] - pad_onset,
            'end': seg['end'] + pad_offset
        })
    
    # Convert to DataFrame and merge overlapping segments
    df = pd.DataFrame(segments)
    df = df.sort_values('start')
    
    merged = []
    current = df.iloc[0]
    
    for _, segment in df.iloc[1:].iterrows():
        if segment['start'] <= current['end'] + min_duration_off:
            current['end'] = max(current['end'], segment['end'])
        else:
            if current['end'] - current['start'] >= min_duration_on:
                merged.append(current)
            current = segment
    
    merged.append(current)
    return pd.DataFrame(merged)
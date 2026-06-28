#!/usr/bin/env python3
"""
EVEZ Autonomous Music Generator
Generates new tracks on a schedule — randomized presets, bass, FX, and parameters.
Each run produces one unique track. Appends to catalog.
"""
import sys, os, random, json, time
from pathlib import Path

# Ensure venv
VENV = Path.home() / ".openclaw/workspace/.venv"
if str(VENV / "bin") not in os.environ.get("PATH", ""):
    os.environ["PATH"] = str(VENV / "bin") + ":" + os.environ.get("PATH", "")

sys.path.insert(0, str(Path.home() / ".openclaw/workspace/evez-daw"))
os.chdir(Path.home() / ".openclaw/workspace/evez-daw")

import evez_daw_v2 as d
import numpy as np
import soundfile as sf

SR = d.SR

def make_adsr(dur, a=0.01, dec=0.1, s=0.7, r=0.1):
    total = int(dur*SR)
    ia, id_, ir = int(a*SR), int(dec*SR), int(r*SR)
    is_ = max(1, int(s*(dur-a-dec-r)*SR))
    env = np.zeros(total, dtype=np.float32)
    if 0 < ia < total: env[:ia] = np.linspace(0, 1, ia)
    end_d = min(ia+id_, total)
    if end_d > ia: env[ia:end_d] = np.linspace(1, 0.7, end_d-ia)
    end_s = min(end_d+is_, total)
    if end_s > end_d: env[end_d:end_s] = 0.7
    if total > end_s: env[end_s:] = np.linspace(0.7, 0, total-end_s)
    return env

def lead_pluck(dur, freq, dec=15):
    t = np.linspace(0, dur, int(dur*SR), dtype=np.float32)
    s = np.sin(2*np.pi*freq*t) + 0.5*np.sin(2*np.pi*freq*2*t) + 0.3*np.sin(2*np.pi*freq*3*t)
    return d.fad(d.norm(s * np.exp(-dec*t)))

def lead_pad(dur, freq, detune=3):
    t = np.linspace(0, dur, int(dur*SR), dtype=np.float32)
    s = sum(np.sin(2*np.pi*(freq + i*detune)*t)/(i+1) for i in range(5))
    env = make_adsr(dur, 0.5, 0.5, 0.8, 1.0)
    return d.fad(d.norm(s * env), 0.1, 0.5)

def lead_arpeggio(notes, bpm=170, freq_mult=4):
    sl = d.step_len(bpm, '16th')
    r = np.zeros(0, dtype=np.float32)
    for semis in notes:
        f = 55 * (2 ** (semis/12.0)) * freq_mult
        seg = lead_pluck(sl/SR * 2, f, dec=20)
        r = np.concatenate([r, seg])
    return r

# Random arpeggio patterns
def gen_arp(scale_notes, length=16):
    return [random.choice(scale_notes) for _ in range(length)]

SCALES = {
    'minor': [0, 2, 3, 5, 7, 8, 10, 12],
    'phrygian': [0, 1, 3, 5, 7, 8, 10, 12],
    'harmonic_minor': [0, 2, 3, 5, 7, 8, 11, 12],
    'diminished': [0, 2, 3, 5, 6, 8, 9, 11, 12],
    'whole_tone': [0, 2, 4, 6, 8, 10, 12],
    'blues': [0, 3, 5, 6, 7, 10, 12],
    'chromatic': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
}

FX_OPTIONS = ['distortion', 'delay', 'reverb', 'bitcrush', 'lowpass', 'highpass', 'comb', 'chorus', 'waveshaper']

def random_fx_chain(n=None):
    if n is None: n = random.randint(2, 4)
    chain = []
    for _ in range(n):
        fx = random.choice(FX_OPTIONS)
        params = {}
        if fx == 'distortion': params = {'drive': random.uniform(2.0, 6.0), 'mix': random.uniform(0.2, 0.5)}
        elif fx == 'delay': params = {'dms': random.choice([120, 150, 180, 200, 250, 300]), 'fb': random.uniform(0.3, 0.6), 'wet': random.uniform(0.2, 0.4)}
        elif fx == 'reverb': params = {'wet': random.uniform(0.15, 0.4), 'dec': random.uniform(0.3, 0.7)}
        elif fx == 'bitcrush': params = {'bits': random.randint(4, 8), 'ds': random.randint(1, 4)}
        elif fx == 'lowpass': params = {'cut': random.choice([1500, 2000, 3000, 4000, 5000])}
        elif fx == 'highpass': params = {'cut': random.choice([100, 150, 200, 300])}
        elif fx == 'comb': params = {'delay_ms': random.choice([10, 15, 20, 25, 30]), 'wet': random.uniform(0.1, 0.3)}
        elif fx == 'chorus': params = {'wet': random.uniform(0.1, 0.3)}
        elif fx == 'waveshaper': params = {'amount': random.uniform(0.3, 0.8)}
        chain.append({'name': fx, 'params': params})
    return chain

def generate_track():
    preset_name = random.choice(list(d.PRESETS.keys()))
    p = d.PRESETS[preset_name]
    bpm = p['bpm']
    bars = random.choice([4, 8, 8, 8, 16])
    bass_type = random.choice(list(d.BASS_FNS.keys()))
    bass_freq = random.choice([40, 42, 45, 50, 55, 60, 65, 70, 80])
    scale = random.choice(list(SCALES.keys()))
    scale_notes = SCALES[scale]
    swing = random.uniform(0.0, 0.2)
    sidechain_depth = random.uniform(0.4, 0.8)
    eq_low = random.uniform(1.1, 1.5)
    eq_mid = random.uniform(0.8, 1.0)
    eq_high = random.uniform(0.9, 1.3)
    widen = random.uniform(0.25, 0.5)
    include_lead = random.random() > 0.2
    include_pad = random.random() > 0.4
    fx_chain = random_fx_chain()
    
    # Render drums
    drums = d.render_pattern(p, bpm, bars, swing=swing)
    bd = 60.0/bpm*bars*4
    
    # Render bass
    bass_fn = d.BASS_FNS[bass_type]
    bass = bass_fn(bd, bass_freq)
    
    layers = [(drums, 0.7), (bass, 0.5)]
    
    # Lead arpeggio
    if include_lead:
        arp = gen_arp(scale_notes, 16) * (bars // 2)
        lead = lead_arpeggio(arp, bpm=bpm, freq_mult=random.choice([3, 4, 5]))
        # Apply random FX to lead
        for fx in random_fx_chain(random.randint(1, 3)):
            fn = d.FX_TABLE.get(fx['name'])
            if fn: lead = fn(lead, **fx['params'])
        layers.append((lead, random.uniform(0.2, 0.35)))
    
    # Pad
    if include_pad:
        pad_freq = bass_freq * random.choice([3, 4, 5])
        pad = lead_pad(bd, pad_freq, detune=random.randint(1, 4))
        pad = d.fx_reverb(pad, wet=random.uniform(0.3, 0.6))
        layers.append((pad, random.uniform(0.08, 0.15)))
    
    # Mix
    total = max(len(a) for a, _ in layers)
    mix = np.zeros(total, dtype=np.float32)
    for a, g in layers:
        m = np.zeros(total, dtype=np.float32)
        m[:len(a)] = a * g
        mix += m
    
    # Sidechain
    mix = d.sidechain(mix, bpm, depth=sidechain_depth)
    
    # FX chain on master
    for fx in fx_chain:
        fn = d.FX_TABLE.get(fx['name'])
        if fn: mix = fn(mix, **fx['params'])
    
    # Finalize
    mix = d.fad(mix, 0.02, 0.1)
    mix = d.master_bus(mix, eq_low=eq_low, eq_mid=eq_mid, eq_high=eq_high, widen=widen)
    
    # Metadata
    ts = int(time.time())
    meta = {
        'timestamp': ts,
        'preset': preset_name,
        'bpm': bpm,
        'bars': bars,
        'bass': bass_type,
        'bass_freq': bass_freq,
        'scale': scale,
        'swing': round(swing, 3),
        'sidechain': round(sidechain_depth, 2),
        'eq': {'low': round(eq_low, 2), 'mid': round(eq_mid, 2), 'high': round(eq_high, 2)},
        'widen': round(widen, 2),
        'fx': [f['name'] for f in fx_chain],
        'layers': ['drums', 'bass'] + (['lead'] if include_lead else []) + (['pad'] if include_pad else []),
        'duration': round(len(mix)/SR, 1),
    }
    
    name = f"auto_{preset_name}_{bass_type}_{ts}"
    wav_path = d.OUTPUT_DIR / f"{name}.wav"
    sf.write(str(wav_path), mix, SR)
    
    # MP3
    import subprocess
    mp3_path = d.OUTPUT_DIR / f"{name}.mp3"
    subprocess.run(["ffmpeg", "-y", "-i", str(wav_path), "-b:a", "192k", str(mp3_path)],
                   capture_output=True, check=True)
    
    # Save metadata
    meta_path = d.OUTPUT_DIR / f"{name}.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    
    # Append to catalog
    catalog_path = d.OUTPUT_DIR / "catalog.jsonl"
    with open(catalog_path, 'a') as f:
        f.write(json.dumps(meta) + '\n')
    
    return name, meta

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    for i in range(n):
        name, meta = generate_track()
        print(f"[{i+1}/{n}] {name} | {meta['preset']} {meta['bpm']}BPM | {meta['bass']} | {meta['duration']}s | {meta['layers']} | {meta['fx']}")
    print(f"Done. {n} track(s) generated.")

#!/usr/bin/env python3
"""EVEZ DAW v2 - Pure-Code Music Production Engine
Zero samples. Zero paid APIs. Pure NumPy/SciPy synthesis.
Author: EVEZ // Steven Crawford-Maggard | License: MIT"""
import json,math,os,sys,argparse,subprocess,tempfile
from pathlib import Path
import numpy as np
from scipy import signal
import soundfile as sf

SR=44100; PI2=2*math.pi
BASE_DIR=Path(__file__).parent; OUTPUT_DIR=BASE_DIR/"output"; OUTPUT_DIR.mkdir(parents=True,exist_ok=True)
NOTE_NAMES=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
NOTE_FREQS={i:440.0*(2**((i-69)/12.0)) for i in range(128)}

def note_to_freq(ns):
    if ns.isdigit(): return NOTE_FREQS.get(int(ns),440.0)
    name=ns[:-1].upper(); octv=int(ns[-1]); return NOTE_FREQS.get(NOTE_NAMES.index(name)+(octv+1)*12,440.0)
def mt(s): return int(s*SR)
def norm(a,t=0.9):
    p=np.max(np.abs(a)); return a*(t/p) if p>0 else a
def fad(a,fi=0.005,fo=0.01):
    r=a.copy(); i=mt(fi); o=mt(fo)
    if 0<i<len(r): r[:i]*=np.linspace(0,1,i)
    if 0<o<len(r): r[-o:]*=np.linspace(1,0,o)
    return r
def bflt(a,lo,hi,order=4):
    ny=SR/2; lo=max(20,min(lo,ny-100)); hi=max(lo+100,min(hi,ny-1))
    return signal.sosfilt(signal.butter(order,[lo/ny,hi/ny],btype='band',output='sos'),a).astype(np.float32)
def lflt(a,cut,order=4,res=None):
    ny=max(0.01,min(0.99,cut/(SR/2)))
    sos=signal.iirfilter(order,ny,rp=res,btype='low',output='sos') if res else signal.butter(order,ny,btype='low',output='sos')
    return signal.sosfilt(sos,a).astype(np.float32)
def hflt(a,cut,order=4):
    ny=max(0.01,min(0.99,cut/(SR/2)))
    return signal.sosfilt(signal.butter(order,ny,btype='high',output='sos'),a).astype(np.float32)
def stereo(a,width=1.0):
    if len(a.shape)>1: return a
    delay=int(0.015*SR*width); r=np.zeros((len(a),2),dtype=np.float32); r[:,0]=a
    r[delay:,1]=a[:len(a)-delay] if delay<len(a) else a
    return r
def soft_clip(a,drive=1.0): return np.tanh(a*drive)/np.tanh(drive)

# DRUMS
def synth_kick(dur=0.5,freq=55,dec=8.0,click=0.02,dist=0.0):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    pe=np.exp(-dec*t)*(freq*4-freq)+freq; ph=np.cumsum(pe/SR)*PI2
    body=np.sin(ph)*np.exp(-dec*t); cl=mt(click)
    ct=np.linspace(0,click,cl,dtype=np.float32); cw=np.sin(PI2*2000*ct)*np.exp(-60*ct)
    r=np.zeros(len(t),dtype=np.float32); r[:cl]+=cw*0.3; r+=body
    if dist>0: r=np.tanh(r*dist)/np.tanh(dist)
    return fad(norm(r))
def synth_snare(dur=0.3,freq=200,nm=0.7,dec=12):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    body=np.sin(PI2*freq*t)*np.exp(-20*t)
    noise=np.random.uniform(-1,1,len(t)).astype(np.float32)*np.exp(-dec*t)
    res=np.sin(PI2*freq*3*t)*np.exp(-30*t)*0.3
    return fad(norm((1-nm)*body+nm*noise+res))
def synth_hat(dur=0.1,freq=8000,dec=40,open=False):
    if open: dur=0.3; dec=12
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    met=(np.sin(PI2*freq*t)+np.sin(PI2*freq*1.5*t)*0.5+np.sin(PI2*freq*2.3*t)*0.3+np.sin(PI2*freq*3.1*t)*0.2)*0.25
    noise=np.random.uniform(-1,1,len(t)).astype(np.float32)*0.5
    return fad(norm(bflt((met+noise)*np.exp(-dec*t),6000,14000)))
def synth_clap(dur=0.15,layers=4):
    r=np.zeros(mt(dur),dtype=np.float32)
    for i in range(layers):
        off=int(i*0.008*SR); bl=mt(0.02)
        b=np.random.uniform(-1,1,bl).astype(np.float32)*np.exp(-80*np.linspace(0,0.02,bl))
        end=min(off+bl,len(r)); r[off:end]+=b[:end-off]
    return fad(norm(r*np.exp(-18*np.linspace(0,dur,len(r)))))
def synth_rim(dur=0.05,freq=1000):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    return fad(norm(np.sin(PI2*freq*t)*np.exp(-80*t)*0.3+np.random.uniform(-1,1,len(t)).astype(np.float32)*np.exp(-100*t)*0.7))
def synth_tom(dur=0.3,freq=100,dec=6):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    pe=np.exp(-dec*t)*(freq*2-freq)+freq; ph=np.cumsum(pe/SR)*PI2
    return fad(norm(np.sin(ph)*np.exp(-dec*t)))
def synth_crash(dur=2.0,freq=3000):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    noise=np.random.uniform(-1,1,len(t)).astype(np.float32)
    metals=sum(np.sin(PI2*freq*(1+i*0.37)*t)*0.3/(i+1) for i in range(6))
    return fad(norm(bflt((noise*0.5+metals)*np.exp(-3*t),4000,12000)),0.01,0.5)
def synth_perc(dur=0.1,freq=500,dec=30):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    return fad(norm(np.sin(PI2*freq*t)*np.exp(-dec*t)+np.random.uniform(-1,1,len(t)).astype(np.float32)*np.exp(-50*t)*0.3))

DRUM_FNS={'kick':synth_kick,'snare':synth_snare,'hat':synth_hat,'open_hat':lambda:synth_hat(open=True),
    'clap':synth_clap,'rim':synth_rim,'tom':lambda:synth_tom(freq=120),'crash':lambda:synth_crash(dur=1.5),'perc':lambda:synth_perc(freq=600)}

# BASS
def bass_sub(dur,freq=40):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32); return fad(norm(np.sin(PI2*freq*t)))
def bass_reese(dur,freq=55,det=8):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    return fad(norm((np.sin(PI2*freq*t)+np.sin(PI2*(freq+det)*t)*0.8+np.sin(PI2*(freq-det*0.5)*t)*0.6)/2.4))
def bass_wobble(dur,freq=55,lfo_rate=4,lfo_depth=800):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    src=norm(np.sin(PI2*freq*t)+0.5*np.sin(PI2*freq*2*t)+0.3*np.sin(PI2*freq*3*t))
    lfo=(np.sin(PI2*lfo_rate*t)+1)/2; cutoff=200+lfo*lfo_depth
    r=np.zeros_like(src)
    for i in range(0,len(src),256):
        e=min(i+256,len(src)); c=int(cutoff[i])
        r[i:e]=bflt(src[i:e],max(100,c-500),min(20000,c+500))
    return fad(norm(r))
def bass_phonk(dur,freq=65,dist=3.0):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    pe=np.exp(-3*t)*(freq*2-freq)+freq; ph=np.cumsum(pe/SR)*PI2
    body=np.tanh(np.sin(ph)*np.exp(-2*t)*dist)/np.tanh(dist)
    sub=np.sin(PI2*freq*0.5*t)*np.exp(-1.5*t)*0.5
    return fad(norm(body+sub),0.002,0.05)
def bass_scream(dur,freq=80):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    harm=norm(sum(np.sin(PI2*freq*n*t)/n for n in range(1,12)))
    return fad(norm(np.tanh(np.sin(PI2*freq*t+harm*3)*5)/np.tanh(5)))
def bass_fm_growl(dur,freq=70,mod_rate=3,mod_depth=50):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    return fad(norm(np.tanh(np.sin(PI2*freq*t+np.sin(PI2*mod_rate*freq*t)*mod_depth)*3)/np.tanh(3)))
def bass_square_sub(dur,freq=45):
    t=np.linspace(0,dur,mt(dur),dtype=np.float32)
    return fad(norm(lflt(np.sign(np.sin(PI2*freq*t)).astype(np.float32),freq*8)))

BASS_FNS={'sub':bass_sub,'reese':bass_reese,'wobble':bass_wobble,'phonk':bass_phonk,
    'scream':bass_scream,'fm_growl':bass_fm_growl,'square_sub':bass_square_sub}

# FX
def fx_dist(a,drive=3.0,mix=1.0):
    d=np.tanh(a*drive)/np.tanh(drive); return (1-mix)*a+mix*d
def fx_crush(a,bits=8,ds=4):
    lv=2**bits; c=np.round(a*lv)/lv
    if ds>1: idx=np.arange(len(c)); c=np.interp(idx,idx[::ds],c[::ds]).astype(np.float32)
    return c
def fx_reverb(a,dec=0.5,dms=50,wet=0.3):
    r=a.copy(); ds=int(dms*SR/1000)
    for i in range(1,7):
        off=int(ds*(1+i*0.37)); g=(dec**i)*wet
        if off<len(r): r[off:]+=a[:len(r)-off]*g
    return norm(r)
def fx_delay(a,dms=250,fb=0.4,wet=0.3):
    ds=int(dms*SR/1000); r=a.copy(); c=a
    for i in range(1,6):
        g=fb**i*wet; s=np.zeros(len(r),dtype=np.float32); off=ds*i
        if off<len(c): s[off:off+len(c)]=c[:len(c)-off]*g; r+=s; c=s
    return norm(r)
def fx_lowpass(a,cut=2000,res=5): return lflt(a,cut,res=res)
def fx_highpass(a,cut=200): return hflt(a,cut)
def fx_formant(a,vowel='a'):
    fm={'a':[(800,10),(1150,8),(2900,6)],'e':[(400,10),(1600,8),(2700,6)],
        'i':[(350,10),(2300,8),(3200,6)],'o':[(450,10),(800,8),(2830,6)],'u':[(325,10),(700,8),(2530,6)]}
    r=np.zeros_like(a)
    for f,q in fm.get(vowel,fm['a']):
        b,aa=signal.iirpeak(f,q,fs=SR); r+=signal.lfilter(b,aa,a).astype(np.float32)
    return norm(r)
def fx_comb(a,delay_ms=20,fb=0.7,wet=0.3):
    ds=int(delay_ms*SR/1000); r=a.copy(); c=a
    for i in range(1,10):
        g=fb**i*wet; off=ds*i
        if off<len(r): s=np.zeros(len(r),dtype=np.float32); s[off:off+len(c)]=c[:len(c)-off]*g; r+=s; c=s
    return norm(r)
def fx_chorus(a,rate=0.5,depth=0.003,wet=0.3):
    t=np.arange(len(a))/SR; lfo=np.sin(PI2*rate*t)*depth*SR; r=a.copy()
    for offset in [0.015,0.023,0.031]:
        ds=np.clip((offset*SR+lfo).astype(int),0,len(a)-1); delayed=np.zeros_like(a)
        for i in range(len(a)):
            if ds[i]<i: delayed[i]=a[i-ds[i]]
        r+=delayed*wet/3
    return norm(r)
def fx_waveshaper(a,amount=0.5): return np.tanh(a*(1+amount*10))/np.tanh(1+amount*10)

FX_TABLE={'distortion':fx_dist,'bitcrush':fx_crush,'reverb':fx_reverb,'delay':fx_delay,
    'lowpass':fx_lowpass,'highpass':fx_highpass,'formant':fx_formant,
    'comb':fx_comb,'chorus':fx_chorus,'waveshaper':fx_waveshaper}

# SIDECHAIN
def sidechain(audio,bpm,depth=0.6,release_ms=100):
    period=1.0/(bpm/60.0); env=np.ones(len(audio),dtype=np.float32); rel=mt(release_ms/1000); pos=0
    while pos<len(audio):
        end=min(pos+rel,len(audio)); env[pos:end]=np.linspace(1-depth,1,end-pos); pos+=int(period*SR)
    return audio*env

# MASTER BUS
def master_bus(audio,eq_low=1.0,eq_mid=1.0,eq_high=1.0,limiter=0.95,widen=0.3):
    if eq_low!=1.0: low=hflt(audio,200); audio=audio-low+low*eq_low
    if eq_mid!=1.0: mid=bflt(audio,200,4000); audio=audio-mid+mid*eq_mid
    if eq_high!=1.0: high=lflt(audio,4000); audio=audio-high+high*eq_high
    audio=stereo(audio,widen) if widen>0 else stereo(audio,0)
    audio=soft_clip(audio,limiter); return norm(audio,0.98).astype(np.float32)

# SEQUENCER
def step_len(bpm,subdiv='16th'):
    subs={'whole':1,'half':2,'quarter':4,'8th':8,'16th':16,'32nd':32,'64th':64}
    return int(SR/(bpm/60.0*subs.get(subdiv,16)/4))
def render_pattern(pat,bpm=170,bars=2,swing=0.0):
    sl=step_len(bpm); ts=16*bars; total=sl*ts; r=np.zeros(total,dtype=np.float32)
    sounds={k:fn() for k,fn in DRUM_FNS.items()}
    for voice,steps in pat.items():
        if voice not in sounds: continue
        s=sounds[voice]
        for i,vel in enumerate(steps):
            if vel>0 and i<ts:
                off=int(sl*swing) if swing>0 and i%2==1 else 0
                st=i*sl+off; e=min(st+len(s),total); r[st:e]+=s[:e-st]*vel
    return norm(r)
def render_notes(notes,bpm=170,bass_type='wobble',base_freq=55):
    sl=step_len(bpm); r=np.zeros(0,dtype=np.float32); fn=BASS_FNS.get(bass_type,bass_wobble)
    for semis,dur_steps in notes:
        freq=base_freq*(2**(semis/12.0)); dur=dur_steps*sl/SR; r=np.concatenate([r,fn(dur,freq)])
    return r

# PRESETS
PRESETS={
    "breakcore_170":{"bpm":170,
        "kick":[1,0,0,1,0,0,1,0,0,1,0,0,1,0,1,0,1,0,0,0,1,0,0,1,0,1,1,0,0,0,1,0],
        "snare":[0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,0,0,1,0,1,0,0,0,1,1,0,1],
        "hat":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        "clap":[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0]},
    "dubstep_140":{"bpm":140,
        "kick":[1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0],
        "snare":[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
        "hat":[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0],
        "open_hat":[0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,1]},
    "phonk_130":{"bpm":130,
        "kick":[1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0],
        "snare":[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,1],
        "hat":[1,1,0,1,1,0,1,1,1,1,0,1,1,0,1,0],
        "clap":[0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0]},
    "amen_break":{"bpm":170,
        "kick":[1,0,0,0,0,0,1,0,0,1,0,0,0,0,1,0],
        "snare":[0,0,1,0,1,0,0,1,0,0,1,0,0,1,0,0],
        "hat":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        "open_hat":[0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1]},
    "404_architecture":{"bpm":200,
        "kick":[1,0,0,1,0,1,0,0,1,0,1,0,0,0,1,1,1,1,0,0,1,0,0,1,0,1,0,1,1,0,0,0],
        "snare":[0,0,1,0,0,0,0,1,0,1,0,0,1,0,0,0,0,0,1,1,0,0,0,0,1,0,1,0,0,1,0,1],
        "hat":[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
        "rim":[0,1,0,0,1,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,1,0,0,1,0,0,1,0,0,0,1,0],
        "clap":[0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,1]},
    "industrial_160":{"bpm":160,
        "kick":[1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,1,0,0,1,0,0,1,0,0,0,1,0,1,0,0,1],
        "snare":[0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0],
        "hat":[1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1],
        "crash":[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]},
}

# ARRANGEMENT
def arrange_sections(sections,bpm,bass_type='wobble',bass_freq=55,fx_chain=None,sc_depth=0.0):
    parts=[]
    for preset_name,bars,intensity in sections:
        p=PRESETS.get(preset_name,"breakcore_170"); drums=render_pattern(p,bpm,bars)
        bd=60.0/bpm*bars*4; bass_fn=BASS_FNS.get(bass_type,bass_wobble); bass_audio=bass_fn(bd,bass_freq)
        total=max(len(drums),len(bass_audio))
        dp=np.zeros(total,dtype=np.float32); bp=np.zeros(total,dtype=np.float32)
        dp[:len(drums)]=drums*intensity; bp[:len(bass_audio)]=bass_audio*0.5*intensity
        mix=dp+bp
        if sc_depth>0: mix=sidechain(mix,bpm,depth=sc_depth)
        if fx_chain:
            for fx in fx_chain:
                fn=FX_TABLE.get(fx["name"])
                if fn: mix=fn(mix,**fx.get("params",{}))
        parts.append(fad(mix,0.01,0.05))
    return np.concatenate(parts) if parts else np.zeros(0,dtype=np.float32)

def render_track(preset="breakcore_170",bars=4,bass="wobble",bf=55,fx_chain=None,bpm_override=None,
                 sc_depth=0.0,eq_low=1.0,eq_mid=1.0,eq_high=1.0,widen=0.3,swing=0.0):
    p=PRESETS.get(preset,PRESETS["breakcore_170"]); bpm=bpm_override or p["bpm"]
    drums=render_pattern(p,bpm,bars,swing=swing); bd=60.0/bpm*bars*4
    bass_fn=BASS_FNS.get(bass,bass_sub); bass_audio=bass_fn(bd,bf)
    total=max(len(drums),len(bass_audio))
    dp=np.zeros(total,dtype=np.float32); bp=np.zeros(total,dtype=np.float32)
    dp[:len(drums)]=drums; bp[:len(bass_audio)]=bass_audio
    mix=dp*0.7+bp*0.5
    if sc_depth>0: mix=sidechain(mix,bpm,depth=sc_depth)
    if fx_chain:
        for fx in fx_chain:
            fn=FX_TABLE.get(fx["name"])
            if fn: mix=fn(mix,**fx.get("params",{}))
    mix=fad(mix,0.01,0.05); return master_bus(mix,eq_low,eq_mid,eq_high,widen=widen)

def render_full_arrangement(bpm=170,bass_type='wobble',bass_freq=55,fx_chain=None,sc_depth=0.6,
    eq_low=1.2,eq_mid=0.9,eq_high=1.1,widen=0.3,swing=0.0):
    """Full track: intro(2 bars) -> verse(4) -> drop(8) -> break(2) -> drop(8) -> outro(2)"""
    sections=[
        ("breakcore_170",2,0.5),
        ("breakcore_170",4,0.7),
        ("breakcore_170",8,1.0),
        ("breakcore_170",2,0.4),
        ("breakcore_170",8,1.0),
        ("breakcore_170",2,0.3),
    ]
    audio=arrange_sections(sections,bpm,bass_type,bass_freq,fx_chain,sc_depth)
    return master_bus(audio,eq_low,eq_mid,eq_high,widen=widen)

# DRUMKIT
def gen_drumkit(name="evez_breakcore",style="breakcore"):
    kd=BASE_DIR/"drumkits"/name; kd.mkdir(parents=True,exist_ok=True); files={}
    for i,(f,d) in enumerate([(55,8),(45,6),(60,12),(35,5)]):
        p=kd/f"kick_{i+1}.wav"; sf.write(str(p),synth_kick(freq=f,dec=d),SR); files[f"kick_{i+1}"]=str(p)
    for i,(f,nm) in enumerate([(200,0.7),(180,0.8),(250,0.6),(160,0.9)]):
        p=kd/f"snare_{i+1}.wav"; sf.write(str(p),synth_snare(freq=f,nm=nm),SR); files[f"snare_{i+1}"]=str(p)
    for fn,oh in [("hat_closed_1",False),("hat_closed_2",False),("hat_open_1",True),("hat_open_2",True)]:
        p=kd/f"{fn}.wav"; sf.write(str(p),synth_hat(open=oh),SR); files[fn]=str(p)
    for i in range(3):
        p=kd/f"clap_{i+1}.wav"; sf.write(str(p),synth_clap(layers=3+i*2),SR); files[f"clap_{i+1}"]=str(p)
    p=kd/"rimshot.wav"; sf.write(str(p),synth_rim(),SR); files["rimshot"]=str(p)
    if style in("breakcore","404"):
        for i in range(3):
            k=fx_dist(synth_kick(freq=50+i*10,dec=10),drive=4+i*2)
            p=kd/f"kick_distorted_{i+1}.wav"; sf.write(str(p),k,SR); files[f"kick_distorted_{i+1}"]=str(p)
    m={"name":name,"style":style,"samples":len(files),"files":files}
    (kd/"manifest.json").write_text(json.dumps(m,indent=2)); return m

# VOICE CHOP
def chop(a,n=8,gate=0.5):
    sl=len(a)//n; slices=[]
    for i in range(n):
        s=a[i*sl:(i+1)*sl].copy(); env=np.abs(s); th=np.max(env)*(1-gate)
        ab=np.where(env>th)[0]
        if len(ab)>0: s=s[ab[0]:ab[-1]+1]
        slices.append(fad(s))
    return slices
def rearrange(sl,pat=None):
    if pat is None: pat=[0,2,1,3,0,3,2,1]
    r=np.zeros(0,dtype=np.float32)
    for i in pat:
        if i<len(sl): r=np.concatenate([r,sl[i]])
    return r

# MP3 EXPORT
def export_mp3(wav_path,mp3_path=None,bitrate=192):
    if mp3_path is None: mp3_path=wav_path.rsplit('.',1)[0]+'.mp3'
    subprocess.run(["ffmpeg","-y","-i",wav_path,"-b:a",f"{bitrate}k",mp3_path],
                   capture_output=True,check=True)
    return mp3_path

# CLI
def cmd_render(args):
    fx_chain=None
    if args.fx:
        fx_chain=[]
        for f in args.fx.split(','):
            parts=f.split(':'); name=parts[0]; params={}
            if len(parts)>1:
                for p in parts[1].split(';'):
                    k,v=p.split('='); params[k]=float(v) if '.' in v else int(v)

            fx_chain.append({"name":name,"params":params})
    if args.arrange:
        audio=render_full_arrangement(bpm=args.bpm,bass_type=args.bass,bass_freq=args.bass_freq,
            fx_chain=fx_chain,sc_depth=args.sidechain,eq_low=args.eq_low,eq_mid=args.eq_mid,
            eq_high=args.eq_high,widen=args.widen,swing=args.swing)
    else:
        audio=render_track(preset=args.preset,bars=args.bars,bass=args.bass,bf=args.bass_freq,
            fx_chain=fx_chain,bpm_override=args.bpm if args.bpm else None,sc_depth=args.sidechain,
            eq_low=args.eq_low,eq_mid=args.eq_mid,eq_high=args.eq_high,widen=args.widen,swing=args.swing)
    fn=args.output or f"evez_{args.preset}_{int(__import__('time').time())}.wav"
    path=OUTPUT_DIR/fn; sf.write(str(path),audio,SR)
    dur=len(audio)/SR
    print(f"Rendered: {path} ({dur:.1f}s, {len(audio)} samples)")
    if args.mp3:
        mp3=export_mp3(str(path)); print(f"MP3: {mp3}")
    return str(path)

def cmd_drumkit(args):
    m=gen_drumkit(args.name,args.style)
    print(f"Drumkit: {args.name} ({m['samples']} samples)")
    for k,v in m["files"].items(): print(f"  {k}: {v}")

def cmd_presets(args):
    print("Available presets:")
    for name,p in PRESETS.items():
        voices=list(p.keys()); voices.remove("bpm")
        print(f"  {name}: {p['bpm']} BPM, voices: {voices}")
    print("\nBass types:",list(BASS_FNS.keys()))
    print("FX:",list(FX_TABLE.keys()))

def cmd_chop(args):
    if not os.path.exists(args.input): print(f"Error: {args.input} not found"); return
    a,sr=sf.read(args.input); a=a.astype(np.float32)
    if len(a.shape)>1: a=a.mean(axis=1)
    sl=chop(a,args.slices,args.gate)
    if args.pattern:
        pat=[int(x) for x in args.pattern.split(',')]
        r=rearrange(sl,pat)
    else:
        r=np.concatenate(sl)
    fn=args.output or f"chopped_{int(__import__('time').time())}.wav"
    path=OUTPUT_DIR/fn; sf.write(str(path),r,SR)
    print(f"Chopped: {path} ({len(sl)} slices, {len(r)/SR:.1f}s)")
    return str(path)

def main():
    ap=argparse.ArgumentParser(description="EVEZ DAW v2 - Pure-Code Music Production")
    sub=ap.add_subparsers(dest="command")
    
    rp=sub.add_parser("render",help="Render a track")
    rp.add_argument("--preset",default="breakcore_170",choices=list(PRESETS.keys()))
    rp.add_argument("--bars",type=int,default=4)
    rp.add_argument("--bass",default="wobble",choices=list(BASS_FNS.keys()))
    rp.add_argument("--bass-freq",type=float,default=55,dest="bass_freq")
    rp.add_argument("--bpm",type=float,default=170)
    rp.add_argument("--fx",default=None,help="FX chain: distortion,bitcrush:bits=4,reverb")
    rp.add_argument("--sidechain",type=float,default=0.0)
    rp.add_argument("--arrange",action="store_true",help="Full multi-section arrangement")
    rp.add_argument("--eq-low",type=float,default=1.0,dest="eq_low")
    rp.add_argument("--eq-mid",type=float,default=1.0,dest="eq_mid")
    rp.add_argument("--eq-high",type=float,default=1.0,dest="eq_high")
    rp.add_argument("--widen",type=float,default=0.3)
    rp.add_argument("--swing",type=float,default=0.0)
    rp.add_argument("--output",default=None)
    rp.add_argument("--mp3",action="store_true")
    rp.set_defaults(func=cmd_render)
    
    dp=sub.add_parser("drumkit",help="Generate drumkit WAV files")
    dp.add_argument("--name",default="evez_breakcore")
    dp.add_argument("--style",default="breakcore",choices=["breakcore","dubstep","phonk","404","industrial"])
    dp.set_defaults(func=cmd_drumkit)
    
    pp=sub.add_parser("presets",help="List available presets")
    pp.set_defaults(func=cmd_presets)
    
    cp=sub.add_parser("chop",help="Chop and rearrange audio")
    cp.add_argument("--input",required=True)
    cp.add_argument("--slices",type=int,default=8)
    cp.add_argument("--gate",type=float,default=0.5)
    cp.add_argument("--pattern",default=None,help="Rearrange pattern: 0,2,1,3")
    cp.add_argument("--output",default=None)
    cp.set_defaults(func=cmd_chop)
    
    args=ap.parse_args()
    if hasattr(args,"func"): args.func(args)
    else: ap.print_help()

if __name__=="__main__":
    main()

# ══════════════════════════════════════════════════════════
# BASELINE — Pretrained XTTS on Urdu (NO Finetuning)
# CS-419 Deep Learning Project
# ══════════════════════════════════════════════════════════

# ── CELL 1: Install ────────────────────────────────────────
# !pip install TTS torch torchaudio scipy matplotlib

# ── CELL 2: Imports ───────────────────────────────────────
import torch
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import numpy as np
import time
import os

print("=" * 50)
print("BASELINE: Pretrained XTTS — No Finetuning")
print("=" * 50)
print(f"GPU available : {torch.cuda.is_available()}")
print(f"GPU name      : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")


from TTS.api import TTS

print("\nDownloading pretrained XTTS v2...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts = tts.to("cuda" if torch.cuda.is_available() else "cpu")
print("Pretrained XTTS v2 loaded ")


REFERENCE_WAV = "xtts_dataset/test/wavs/000000.wav"   
OUTPUT_DIR    = "baseline_outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Test sentences
TEST_SENTENCES = [
    "خوشبو ہوا کی لہروں پر سفر کرتی ہے",
    "پرانے کالج کا بیڑہ غرق نہ کیجئے",
    "رونا ہے کہ کتنی شامیں ویران رہتی ہیں",
]

#  Generate baseline audio 
print("\nGenerating Urdu audio with NO finetuning...")
print("(Expect bad/broken Urdu — this is the baseline)\n")

results = []
for i, sentence in enumerate(TEST_SENTENCES):
    print(f"Sentence {i+1}: {sentence}")
    start = time.time()

    output_path = os.path.join(OUTPUT_DIR, f"baseline_sample_{i}.wav")

    tts.tts_to_file(
        text=sentence,
        speaker_wav=REFERENCE_WAV,
        language="ur",
        file_path=output_path,
    )

    elapsed = time.time() - start
    results.append({"sentence": sentence, "time": elapsed, "file": output_path})
    print(f"  → Saved: {output_path} | Time: {elapsed:.2f}s\n")

# ── CELL 6: Report ────────────────────────────────────────
print("=" * 50)
print("BASELINE RESULTS SUMMARY")
print("=" * 50)
for r in results:
    print(f"  Sentence : {r['sentence']}")
    print(f"  Time     : {r['time']:.2f}s")
    print(f"  Output   : {r['file']}")
    print()

avg_time = np.mean([r["time"] for r in results])
print(f"Avg inference time : {avg_time:.2f}s")
print(f"Total samples      : {len(results)}")

# ── CELL 7: Plot spectrogram ──────────────────────────────
print("\nPlotting spectrogram of baseline output...")

sample_rate, audio = wav.read(results[0]["file"])
audio = audio.astype(np.float32)

plt.figure(figsize=(12, 4))
plt.specgram(audio, Fs=sample_rate, cmap="viridis")
plt.title("BASELINE — Spectrogram (No Finetuning)\nExpect poor Urdu quality")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Intensity (dB)")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "baseline_spectrogram.png"))
plt.show()
print("Spectrogram saved ✅")

print("\n✅ Baseline complete. Outputs in:", OUTPUT_DIR)
print("Note: Quality should be BAD — this proves finetuning is needed.")
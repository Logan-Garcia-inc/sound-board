import wave
import pyaudio
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from scipy.signal import lfilter
import math

# --- Shared State ---
volume = 0.5
bass_gain = 0.0
treble_gain = 0.0
running = True
selected_device_index = None

# --- WAV File ---
WAV_FILE = "C:\\Users\\Logan\\Desktop\\[MP3DL.CC] Travis Scott - SICKO MODE (Audio)-320k.wav"  # Replace this with a valid WAV file

# --- Audio Setup ---
p = pyaudio.PyAudio()

def list_output_devices():
    devices = []
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info["maxOutputChannels"] > 0:
            devices.append((i, info["name"]))
    return devices

# --- EQ Filters ---
def biquad_shelf(data, rate, gain_db, freq, shelf_type):
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * freq / rate
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1/A) * (1/0.707 - 1) + 2)
    cos_w0 = np.cos(w0)

    if shelf_type == "low":
        b0 = A * ((A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = 2 * A * ((A - 1) - (A + 1) * cos_w0)
        b2 = A * ((A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = -2 * ((A - 1) + (A + 1) * cos_w0)
        a2 = (A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    elif shelf_type == "high":
        b0 = A * ((A + 1) + (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha)
        b1 = -2 * A * ((A - 1) + (A + 1) * cos_w0)
        b2 = A * ((A + 1) + (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha)
        a0 = (A + 1) - (A - 1) * cos_w0 + 2 * np.sqrt(A) * alpha
        a1 = 2 * ((A - 1) - (A + 1) * cos_w0)
        a2 = (A + 1) - (A - 1) * cos_w0 - 2 * np.sqrt(A) * alpha
    else:
        return data

    b = np.array([b0, b1, b2]) / a0
    a = np.array([1.0, a1 / a0, a2 / a0])
    return lfilter(b, a, data)

# --- Audio Thread ---
def audio_thread():
    global running, volume, bass_gain, treble_gain, selected_device_index

    wf = wave.open(WAV_FILE, 'rb')
    rate = wf.getframerate()

    stream = p.open(format=pyaudio.paInt16,
                    channels=wf.getnchannels(),
                    rate=rate,
                    output=True,
                    output_device_index=selected_device_index)

    chunk = 1024
    data = wf.readframes(chunk)

    while data and running:
        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)

        # Apply EQ
        samples = biquad_shelf(samples, rate, bass_gain, 200, "low")
        samples = biquad_shelf(samples, rate, treble_gain, 4000, "high")

        samples *= volume
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        stream.write(samples.tobytes())
        data = wf.readframes(chunk)

    stream.stop_stream()
    stream.close()
    wf.close()

# --- GUI Handlers ---
def on_volume(val):
    global volume
    volume = float(val)

def on_bass(val):
    global bass_gain
    bass_gain = float(val)

def on_treble(val):
    global treble_gain
    treble_gain = float(val)

def on_device_select(event):
    global selected_device_index
    name = device_var.get()
    for idx, dev in output_devices:
        if dev == name:
            selected_device_index = idx
            break

def on_play():
    global running
    if not selected_device_index:
        print("No output device selected.")
        return
    running = True
    threading.Thread(target=audio_thread, daemon=True).start()

def on_close():
    global running
    running = False
    root.destroy()

# --- GUI Setup ---
root = tk.Tk()
root.title("Audio Player with EQ + Output Selection")
root.geometry("350x300")

# Device dropdown
tk.Label(root, text="Output Device:").pack()
output_devices = list_output_devices()
device_names = [name for _, name in output_devices]
device_var = tk.StringVar()
device_dropdown = ttk.Combobox(root, textvariable=device_var, values=device_names, state="readonly")
device_dropdown.bind("<<ComboboxSelected>>", on_device_select)
device_dropdown.pack(pady=5)

# Play button
play_button = tk.Button(root, text="Play", command=on_play)
play_button.pack(pady=5)

# Volume
tk.Label(root, text="Volume").pack()
tk.Scale(root, from_=0.0, to=2.0, resolution=0.1,
         orient="horizontal", command=on_volume).set(0.5)

# Bass
tk.Label(root, text="Bass Boost (dB)").pack()
tk.Scale(root, from_=-12, to=12, resolution=1,
         orient="horizontal", command=on_bass).set(0)

# Treble
tk.Label(root, text="Treble Boost (dB)").pack()
tk.Scale(root, from_=-12, to=12, resolution=1,
         orient="horizontal", command=on_treble).set(0)

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
p.terminate()

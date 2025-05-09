import os
import wave
#os.system("pip install imageio[ffmpeg]")
try:
    import pyaudio
except ImportError:
    os.system("pip install pyaudio")
    import pyaudio
try:
    import numpy as np
except ImportError:
    os.system("pip install numpy")
    import numpy as np
import threading
try:
    import tkinter as tk
except ImportError:
    os.system("pip install tkinter")
    import tkinter as tk
from tkinter import ttk
try:
    from scipy.signal import lfilter
except ImportError:
    os.system("pip install scipy")
    from scipy.signal import lfilter
try:
    import audioread
except ImportError:
    os.system("pip install audioread")
    import audioread
# --- Shared State ---
volume = 0.5
bass_gain = 0.0
treble_gain = 0.0
running = True
selected_device_index = None


# --- WAV File ---
AUDIO_FOLDER = "C:\\Users\\Logan\\Desktop"  # Change to the folder containing your .wav files
WAV_FILE = None  # Default will be selected from list
# --- Audio Setup ---
p = pyaudio.PyAudio()

def convert_mp3_to_wav(mp3_file):
    with audioread.audio_open(mp3_file) as f:
        # Get parameters
        rate = f.samplerate
        channels = f.channels
        sampwidth = 2  # PCM 16-bit
        frames = f.frames

        # Create the output WAV file
        with wave.open(AUDIO_FOLDER+mp3_file+".wav", 'wb') as out_f:
            out_f.setnchannels(channels)
            out_f.setsampwidth(sampwidth)
            out_f.setframerate(rate)

            # Read the MP3 data and write to the WAV file
            while True:
                try:
                    samples = f.read_data()
                    # Convert to numpy array and write as PCM
                    pcm_samples = np.frombuffer(samples, dtype=np.int16)
                    out_f.writeframes(pcm_samples.tobytes())
                except audioread.exceptions.DecodeError:
                    break
    return AUDIO_FOLDER+mp3_file+".wav"

def load_audio_files():
    files = [f for f in os.listdir(AUDIO_FOLDER) if (f.lower().endswith(".wav"))] #or f.lower().endswith(".mp3"))]
    return files

def on_file_select(event):
    global WAV_FILE
    selection = file_listbox.curselection()
    if selection:
        filename = file_listbox.get(selection[0])
        WAV_FILE = os.path.join(AUDIO_FOLDER, filename)

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
    global running, volume, bass_gain, treble_gain, selected_device_index,WAV_FILE
    print(WAV_FILE[-4:])
    if (WAV_FILE[-4:]==".mp3"):
        WAV_FILE=convert_mp3_to_wav(WAV_FILE)
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
    if not WAV_FILE:
        print("No audio file selected.")
        return

    # Stop current playback if running
    running = False
    threading.Event().wait(0.1)  # Small delay to allow thread to stop

    # Start new playback
    running = True
    threading.Thread(target=audio_thread, daemon=True).start()

def on_close():
    global running
    running = False
    root.destroy()

# --- GUI Setup ---
root = tk.Tk()
root.title("Audio Player with EQ + Output Selection")
root.geometry("350x400")

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
volume_slider = tk.Scale(root, from_=0.0, to=5.0, resolution=0.05,
                         orient="horizontal", command=on_volume)
volume_slider.set(0.5)
volume_slider.pack()

# Bass
tk.Label(root, text="Bass Boost (dB)").pack()
bass_slider = tk.Scale(root, from_=-15, to=15, resolution=1,
                       orient="horizontal", command=on_bass)
bass_slider.set(0)
bass_slider.pack()

# Treble
tk.Label(root, text="Treble Boost (dB)").pack()
treble_slider = tk.Scale(root, from_=-12, to=12, resolution=1,
                         orient="horizontal", command=on_treble)
treble_slider.set(0)
treble_slider.pack()

tk.Label(root, text="Select Audio File:").pack()
file_listbox = tk.Listbox(root, height=6)
file_listbox.pack(pady=5)
file_listbox.bind("<<ListboxSelect>>", on_file_select)

# Load files into the listbox
for file in load_audio_files():
    file_listbox.insert(tk.END, file)

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
p.terminate()

import os
AUDIO_FOLDER= os.path.expanduser("~")+"\\music"
                #<------important white space
try:
    if not AUDIO_FOLDER:
        AUDIO_FOLDER=input("Path to audio folder: ")
        with open(os.path.abspath(__file__),"r") as file:
            lines = file.readlines()
        with open(os.path.abspath(__file__),"w") as file:
            lines[2] =  f'AUDIO_FOLDER = "{AUDIO_FOLDER}"' +'\n'
            file.writelines(lines)
except Exception as e:
    print(e)

print(AUDIO_FOLDER)
import io
import random
import wave
import ffmpeg
#os.system("pip install ffmpeg-python")
from pydub import AudioSegment
from pydub.utils import which
import json
import pyaudio
import numpy as np
import threading
import tkinter as tk
from tkinter import ttk
from scipy.signal import lfilter
import librosa
import soundfile as sf

# --- Shared State ---

script_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["PATH"] += os.pathsep + script_dir
AudioSegment.converter =  os.path.join(script_dir, "scripts\\ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(script_dir, "scripts\\ffprobe.exe")
volume = 0.5
bass_gain = 0.0
treble_gain = 0.0
current_frame = 0
speed = 1.0
scan_slider = None
user_seeking = False
running = True
selected_device_index = None
slider_update_id=None
shuffle=False
looping=False
audioMapPath="audioMap.json"
delete_temp_wav_files=False
history=[]
historyPosition=0

# --- WAV File ---
WAV_FILE = None  
# --- Audio Setup ---
p = pyaudio.PyAudio()    

def convert_mp3_to_wav(mp3_file_path):
    global audioMapPath, audioMap
    if mp3_file_path in audioMap.keys():
        wav_file_path=audioMap[mp3_file_path]
    else:
        wav_file_path = script_dir+"\\"+os.path.basename(mp3_file_path) + ".convertedTo.wav"
        sound = AudioSegment.from_mp3(mp3_file_path)
        sound.export(wav_file_path, format="wav")
        audioMap[mp3_file_path]=wav_file_path
        with open(audioMapPath,"w") as file:
            file.write(json.dumps(audioMap))
    print(audioMap)
    return wav_file_path

def load_audio_files():
    try:
        files = [f for f in os.listdir(AUDIO_FOLDER) if (f.lower().endswith(".wav") or f.lower().endswith(".mp3"))]
        return files
    except FileNotFoundError:
        print("\nAudio directory path not accessible: "+AUDIO_FOLDER+"\nEdit path on line 1")
        threading.Event().wait(5)

def on_file_select(event):
    global WAV_FILE
    selection = file_listbox.curselection()
    if selection:
        filename = file_listbox.get(selection[0])
        WAV_FILE = os.path.join(AUDIO_FOLDER, filename)

def list_output_devices():
    devices = []
    for f in range(p.get_device_count()):
        info = p.get_device_info_by_index(f)
        if info["maxOutputChannels"] > 0:
            devices.append((f, info["name"]))
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

def fast_speed_change(file_path, speed):
    buf = io.BytesIO()
    out, _ = (
        ffmpeg.input(file_path)
        .filter("atempo", speed)
        .output("pipe:", format="wav")
        .run(capture_stdout=True, capture_stderr=True)
    )
    buf.write(out)
    buf.seek(0)
    return buf

def returnRandomSong():
    return os.path.join(AUDIO_FOLDER, random.choice(load_audio_files()))

def next_song():
    print("Next")
    global shuffle,WAV_FILE
    print(f"{historyPosition}, {len(history)}")
    if historyPosition < len(history)-1:     #[song1, song2, _song3_] length=3 pos=2
        song=history[historyPosition+1]
        print("Playing next song...")
    elif shuffle:
        song = returnRandomSong()
        print("Playing next random song...")
    WAV_FILE=song
    on_play()


# --- Audio Thread ---
def audio_thread():
    global running, volume, bass_gain, treble_gain, selected_device_index,WAV_FILE, current_frame,looping
    if (WAV_FILE[-4:]==".mp3"):
        WAV_FILE=convert_mp3_to_wav(WAV_FILE)
    wf = wave.open(WAV_FILE, 'rb')
    
    if speed!=1:
        temp_wav = fast_speed_change(WAV_FILE, speed)
        wf = wave.open(temp_wav, 'rb')
    rate = wf.getframerate()
    stream = p.open(format=pyaudio.paInt16,
                channels=wf.getnchannels(),
                rate=rate,
                output=True,
                output_device_index=selected_device_index,
                frames_per_buffer=1024,
                stream_callback=None,
                start=True)

    chunk = 1024
    data = wf.readframes(chunk)

    total_frames = wf.getnframes()
    def update_slider():
        global slider_update_id, current_frame
        if not user_seeking:
            scan_slider.set(current_frame * 100 / total_frames)
            #print(current_frame * 100 / total_frames)
        slider_update_id = root.after(20, update_slider)  # ~50fps is plenty

    update_slider()
    if len(history)>0:
        if history[-1] != WAV_FILE:
            history.append(WAV_FILE)
            
    while current_frame < total_frames and running:
        if user_seeking:
            threading.Event().wait(0.2)
            continue

        wf.setpos(current_frame)
        data = wf.readframes(chunk)
        if not data:
            break

        samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
        # if speed != 1.0:
        #      indices = np.arange(0, len(samples), speed)
        #      indices = indices[indices < len(samples)]
        #      samples = samples[indices.astype(int)]
        #if speed != 1.0:
         #   samples = librosa.effects.time_stretch(y=samples, rate=speed)

        samples = biquad_shelf(samples, rate, bass_gain, 200, "low")
        samples = biquad_shelf(samples, rate, treble_gain, 4000, "high")

        samples *= volume
        samples = np.clip(samples, -32768, 32767).astype(np.int16)
        stream.write(samples.tobytes())

        current_frame = wf.tell()
    if looping:
        next_song()
    stream.stop_stream()
    stream.close()
    wf.close()

def on_slider_change(val):
    global current_frame, user_seeking,WAV_FILE
    if WAV_FILE:
        temp=WAV_FILE
        print(speed)
        if speed!=1.0:
            temp = fast_speed_change(temp, speed)
        wf = wave.open(temp, 'rb')
        total_frames = wf.getnframes()
        new_pos = int(float(val) / 100 * total_frames)
        current_frame = new_pos
        wf.close()

def on_slider_start(event):
    global user_seeking
    user_seeking = True

def on_slider_end(event):
    global user_seeking
    user_seeking = False
    on_slider_change(scan_slider.get())


# --- GUI Handlers ---
def on_volume(val):
    val=float(val)
    if abs(val - 1.0) <= 0.02:
        val = 1.0
        volume_slider.set(1.0)
    global volume
    volume = val

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

def on_speed(val):
    val=float(val)
    if abs(val - 1.0) <= 0.02:
        val = 1.0
        speed_slider.set(1.0)
    global speed
    speed = val
    if running:
        on_play(restart=False)
    

def on_play(restart=True):
    global running, current_frame, user_seeking,shuffle,WAV_FILE,looping
    print(f"Shuffling: {shuffle}, looping: {looping}")
    if selected_device_index is None:
        print("No output device selected.")
        return
    if not WAV_FILE:
        print("No audio file selected.")
        if shuffle:
            print("Picking random...")
            WAV_FILE=returnRandomSong()
        else:
            return
    print(WAV_FILE)
    # Stop current playback if running
    running = False
    threading.Event().wait(0.25)  # Slight delay for previous thread to exit

    # Reset playback position and slider state
    if restart:
        current_frame = 0
        user_seeking = True   # Temporarily stop slider updates
        scan_slider.set(0)
    user_seeking = False  # Resume updates
    global slider_update_id
    if slider_update_id is not None:
        root.after_cancel(slider_update_id)
        slider_update_id = None

    # Start new playback
    running = True
    threading.Thread(target=audio_thread, daemon=True).start()
    #threading.Thread(target=audio_thread, daemon=True).start()

def on_close():
    global running, WAV_FILE, delete_temp_wav_files
    running = False
    threading.Event().wait(0.2)
    print()
    files = [f for f in os.listdir(script_dir) if f.endswith("mp3.convertedTo.wav")]
    # Delete the converted .wav file if it exists
    if delete_temp_wav_files:
        os.remove(audioMapPath)
        for f in files:
            if os.path.exists(os.path.join(script_dir,f)):
                try:
                    os.remove(os.path.join(script_dir,f))
                    print(f"Deleted temporary file: {f}")
                except Exception as e:
                    print(f"Failed to delete {f}: {e}")

    root.destroy()

def toggle(btn, var):
    global shuffle, looping
    if var=="shuffle":
        shuffle= not shuffle
        btn.config(bg="green" if shuffle else "red")
    elif var =="looping":
        looping= not looping
        btn.config(bg="green" if looping else "red")

# --- GUI Setup ---
root = tk.Tk()
root.title("Audio Player with EQ + Output Selection")
root.geometry("350x700")

# Device dropdown
tk.Label(root, text="Output Device:").pack()
output_devices = list_output_devices()
device_names = [name for _, name in output_devices]
device_var = tk.StringVar()
device_dropdown = ttk.Combobox(root, textvariable=device_var, width=30, values=device_names, state="readonly")
device_dropdown.bind("<<ComboboxSelected>>", on_device_select)
device_dropdown.pack(pady=5)

# Play button
play_button = tk.Button(root, text="Play", command=on_play)
play_button.pack(pady=4)

# Volume
volume_slider = tk.Scale(root, from_=0.0, to=3.0, resolution=0.01,
        orient="horizontal",length=300, command=on_volume)
volume_slider.set(1.0)
volume_slider.pack()
tk.Label(root, text="Volume").pack(pady=(4, 0))
# Bass

bass_slider = tk.Scale(root, from_=-15, to=15, resolution=1,
                       orient="horizontal",length=300, command=on_bass)
bass_slider.set(0)
bass_slider.pack()
tk.Label(root, text="Bass Boost (dB)").pack(pady=(4, 0))
# Treble

treble_slider = tk.Scale(root, from_=-12, to=12, resolution=1,
                         orient="horizontal",length=300, command=on_treble)
treble_slider.set(0)
treble_slider.pack()
tk.Label(root, text="Treble Boost (dB)").pack(pady=(4, 0))
#speed

speed_slider = tk.Scale(
    root, from_=0.5, to=2.0, resolution=0.01,
    orient="horizontal",length=300,
    #command=lambda v: on_speed(float(v))
)
speed_slider.bind("<ButtonRelease-1>", lambda e: on_speed(float(speed_slider.get())))
speed_slider.set(1.0)
speed_slider.pack(pady=6)
tk.Label(root, text="Playback Speed (x)").pack(pady=(4, 0))


tk.Label(root, text="Select Audio File:").pack(pady=(10, 0))
file_listbox = tk.Listbox(root, height=6, width=35)
file_listbox.pack(pady=6)
file_listbox.bind("<<ListboxSelect>>", on_file_select)
# Load files into the listbox
for file in load_audio_files():
    file_listbox.insert(tk.END, file)

tk.Label(root, text="Scan Audio").pack(pady=(4, 0))
scan_slider = tk.Scale(root, from_=0, to=100,length=300, orient="horizontal")#, command=on_slider_change)
scan_slider.pack(fill="x", padx=10)

# Bind mouse events to pause/resume seek
scan_slider.bind("<Button-1>", on_slider_start)
scan_slider.bind("<ButtonRelease-1>", on_slider_end)

button_row = tk.Frame(root)
button_row.pack(pady=10,anchor="center")

shuffleBtn = tk.Button(button_row, text="Shuffle", bg="red", command=lambda: toggle(shuffleBtn, "shuffle"))
shuffleBtn.pack(pady=20,side="left", padx=5)

loopBtn = tk.Button(button_row, text="Loop", bg="red", command=lambda: toggle(loopBtn, "looping"))
loopBtn.pack(pady=20,side="left", padx=5)

nextBtn = tk.Button(button_row, text="Next", command=lambda: next_song())
nextBtn.pack(pady=20,side="left", padx=5)

if os.path.exists(audioMapPath):
    with open(audioMapPath, "r") as file:
        audioMap=json.loads(file.read())   
else:
    open(audioMapPath, "a").close()
    audioMap={}
        


root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
p.terminate()

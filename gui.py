# simple gui for launching the programs
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
import subprocess
import sys

def launch_calibrator():
    subprocess.Popen([sys.executable, "camera_calibration.py"])

def launch_l2cs():
    subprocess.Popen([sys.executable, "L2CS_run.py"])

def launch_gaze():
    subprocess.Popen([sys.executable, "openVINO_run.py"])

def launch_face():
    subprocess.Popen([sys.executable, "head_rotation.py"])

# create the window
window = tk.Tk()
window.title("Gaze Tracking")
window.geometry("300x200")
window.resizable(False, False)

# create the notebook
notebook = ttk.Notebook(window)
notebook.pack(pady=10, expand=True)

# create frames
frame1 = ttk.Frame(notebook, width=300, height=200)
frame2 = ttk.Frame(notebook, width=300, height=200)
frame3 = ttk.Frame(notebook, width=300, height=200)
frame4 = ttk.Frame(notebook, width=300, height=200)

# add frames to notebook
notebook.add(frame1, text="Camera Calibration")
notebook.add(frame2, text="L2CS")
notebook.add(frame3, text="OpenVINO")
notebook.add(frame4, text="Head Rotation")

# create buttons
button1 = ttk.Button(frame1, text="Launch", command=launch_calibrator)
button1.pack(pady=10, padx=10, expand=True)

button2 = ttk.Button(frame2, text="Launch", command=launch_l2cs)
button2.pack(pady=10, padx=10, expand=True)

button3 = ttk.Button(frame3, text="Launch", command=launch_gaze)
button3.pack(pady=10, padx=10, expand=True)

button4 = ttk.Button(frame4, text="Launch", command=launch_face)
button4.pack(pady=10, padx=10, expand=True)


window.mainloop()

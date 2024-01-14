# simple gui for launching the programs
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import os
import subprocess
import sys

def launch_calibrator():
    subprocess.Popen([sys.executable, "camera_calibration.py"] + [chessboard_entry.get()] + [square_entry.get()])

def launch_l2cs():
    subprocess.Popen([sys.executable, "L2CS_run.py"] + [ip_entry.get()] + [port_entry.get()])

def launch_gaze():
    subprocess.Popen([sys.executable, "openVINO_run.py"] + [ip_entry.get()] + [port_entry.get()])

def launch_face():
    subprocess.Popen([sys.executable, "head_rotation.py"] + [ip_entry.get()] + [port_entry.get()])

def lauch_server():
    subprocess.Popen([sys.executable, "receiver_server.py"] + [ip_entry.get()] + [port_entry.get()])

def launch_take_photos():
    subprocess.Popen([sys.executable, "take_photos.py"])

# create the window
window = tk.Tk()
window.title("System Estymacji Kąta Patrzenia - SEKP")
window.geometry("500x500")
window.resizable(False, False)

# create the notebook
notebook = ttk.Notebook(window)
notebook.pack(pady=10, expand=True)

# create frames
frame1 = ttk.Frame(notebook, width=300, height=200)
frame2 = ttk.Frame(notebook, width=300, height=200)
frame3 = ttk.Frame(notebook, width=300, height=200)
frame4 = ttk.Frame(notebook, width=300, height=200)
frame5 = ttk.Frame(notebook, width=300, height=200)
frame6 = ttk.Frame(notebook, width=300, height=200)

# add frames to notebook
notebook.add(frame6, text="Take Photos")
notebook.add(frame1, text="Camera Calibration")
notebook.add(frame2, text="L2CS")
notebook.add(frame3, text="OpenVINO")
notebook.add(frame4, text="Head Rotation")
notebook.add(frame5, text="Server")

# create buttons
button1 = ttk.Button(frame1, text="Launch", command=launch_calibrator)
button1.pack(pady=10, padx=10, expand=True)

button2 = ttk.Button(frame2, text="Launch", command=launch_l2cs)
button2.pack(pady=10, padx=10, expand=True)

button3 = ttk.Button(frame3, text="Launch", command=launch_gaze)
button3.pack(pady=10, padx=10, expand=True)

button4 = ttk.Button(frame4, text="Launch", command=launch_face)
button4.pack(pady=10, padx=10, expand=True)

button5 = ttk.Button(frame5, text="Launch", command=lauch_server)
button5.pack(pady=10, padx=10, expand=True)

button6 = ttk.Button(frame6, text="Launch", command=launch_take_photos)
button6.pack(pady=10, padx=10, expand=True)

# add ip and port entry boxes
ip_label = ttk.Label(window, text="IP Address")
ip_label.pack(pady=0, padx=10, expand=True)
ip_entry = ttk.Entry(window)
ip_entry.pack(pady=0, padx=10, expand=True)
ip_entry.insert(0, "localhost")

port_label = ttk.Label(window, text="Port")
port_label.pack(pady=0, padx=10, expand=True)
port_entry = ttk.Entry(window)
port_entry.pack(pady=0, padx=10, expand=True)
port_entry.insert(0, "15200")

# add chessboard size and square size entry boxes in calibration tab
chessboard_label = ttk.Label(frame1, text="Chessboard Size x,y")
chessboard_label.pack(pady=0, padx=10, expand=True)
chessboard_entry = ttk.Entry(frame1)
chessboard_entry.pack(pady=0, padx=10, expand=True)
chessboard_entry.insert(0, "9,6")

square_label = ttk.Label(frame1, text="Square Size (mm)")
square_label.pack(pady=0, padx=10, expand=True)
square_entry = ttk.Entry(frame1)
square_entry.pack(pady=0, padx=10, expand=True)
square_entry.insert(0, "7")

window.mainloop()

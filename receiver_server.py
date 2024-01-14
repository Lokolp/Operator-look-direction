import socket
import tkinter as tk
from threading import Thread
import pickle
import matplotlib.pyplot as plt
import sys


class SocketServer:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(1)
        self.client_socket, self.client_address = self.server_socket.accept()

    def receive_data(self):
        while True:
            data = self.client_socket.recv(1024)
            if not data:
                data = "No data received"
                self.server_socket.close()
                self.__init__(self.host, self.port)
            else:
                try:
                    data = pickle.loads(data)
                    data = str(data[0]) + ", " + str(data[1])
                except:
                    data = "Invalid data received"
            # received_data.set(data.decode('utf-8'))
            received_data.set(data)

    def start_server(self):
        server_thread = Thread(target=self.receive_data)
        server_thread.start()


def main():
    # Set up the Tkinter window
    root = tk.Tk()
    root.title("Socket Server Receiver")

    global received_data
    received_data = tk.StringVar()
    received_data.set("Waiting for data...")

    label = tk.Label(root, textvariable=received_data, wraplength=300)
    label.pack(padx=10, pady=10)

    # Start the socket server
    host = IP
    port = PORT

    server = SocketServer(host, port)
    server.start_server()
    # Run the Tkinter main loop
    root.mainloop()


if __name__ == "__main__":
    IP = (sys.argv[1])
    PORT = int(sys.argv[2])
    main()


import socket
import json


HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65433        # Port to listen on (non-privileged ports are > 1023)


class StateUpdateHandler():
    def __init__(self):
        server_socket = socket.socket()
        server_socket.bind((HOST, PORT))
        while True:
            server_socket.listen(1)
            conn, add = server_socket.accept()
            print("Connected to: " + str(add))
            while True:
                data = conn.recv(1000024)
                if not data:
                    break
                print(data.decode('utf-8'))
            conn.close()

class StateUpdate():
    def __init__(self):
        self.messages = []

    def AddMessageToMessages(self, message):
        self.messages.append(message)
        print(message)



st = StateUpdateHandler()

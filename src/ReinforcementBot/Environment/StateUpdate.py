from Environment.MessageHandler import MessageHandler
import socket

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432       # Port to listen on (non-privileged ports are > 1023)

class StateUpdateHandler():
    server_socket = None
    msg = None

    def __init__(self, dungeon):
        self.server_socket = socket.socket()
        self.server_socket.bind((HOST, PORT))
        self.msg = MessageHandler(dungeon)
        
    def start(self):
        while True:
            self.server_socket.listen(1)
            conn, add = self.server_socket.accept()
            while True:
                data = conn.recv(10000000)
                if not data:
                    break
                self.msg.ReceiveMsg(data)
            conn.close()
from Environment.MessageHandler import MessageHandler
import socket
from time import sleep

HOST = '127.0.0.1'  # Standard loopback interface address (localhost)
PORT = 65432       # Port to listen on (non-privileged ports are > 1023)

class StateUpdateHandler():
    def __init__(self, dungeon):
        self.server_socket = socket.socket()
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((HOST, PORT))
        self.msg = MessageHandler(dungeon)
        self.run = True

    def start(self):
        while self.run:
            self.server_socket.listen(1)
            conn, add = self.server_socket.accept()
            while self.run:
                data = conn.recv(10000000)
                if not data:
                    break
                if(data.decode('utf-8') == "exit"):
                    self.run = False
                else:
                    self.msg.ReceiveMsg(data)
            conn.close()

        self.server_socket.shutdown(socket.SHUT_RDWR)
        self.server_socket.close()
        
    def stop(self):
        try:    
            while True:
                client_socket = socket.socket()  # instantiate
                client_socket.connect((HOST, PORT))
                client_socket.send("exit".encode('utf-8'))
        except:
            print("Exit message sent")
        client_socket.close()

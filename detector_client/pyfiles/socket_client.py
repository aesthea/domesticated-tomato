import json
from websocket import create_connection


class client_connection:
    def __init__(self, ip, port):
        data = {"ip" : ip, "port" : port}
        self.ws = create_connection("ws://%(ip)s:%(port)s" % data, timeout = 5)
    def send(self, json_text):
        self.ws.send(json_text)
        return self.ws.recv()
    def close(self):
        self.ws.close()
        

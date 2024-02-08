import os
import sys
import datetime
import time
from tkinter import *
import sqlite3
import subprocess
import math
import _thread
import multiprocessing
import importlib
import shutil
import ctypes
import random
from PIL import Image
from PIL import ImageTk

import asyncio
import websockets
import json
import urllib
import binascii

EnumWindows = ctypes.windll.user32.EnumWindows
EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
GetWindowText = ctypes.windll.user32.GetWindowTextW
GetWindowTextLength = ctypes.windll.user32.GetWindowTextLengthW
IsWindowVisible = ctypes.windll.user32.IsWindowVisible
titles = []

MYNAME = os.path.split(__file__)[-1]

os.chdir(os.path.split(__file__)[0])

#os.chdir(r"C:\test\summon demon")  

def foreach_window(HWND,LPARAM):
    if IsWindowVisible(HWND):
        length = GetWindowTextLength(HWND)
        buff = ctypes.create_unicode_buffer(length + 1)
        GetWindowText(HWND,buff,length + 1)
        titles.append(buff.value)
        return True
    
EnumWindows(EnumWindowsProc(foreach_window),0)
#print(titles)

def getWindows():
    EnumWindows(EnumWindowsProc(foreach_window),0)
    #print(titles)


class load:
    def __init__(self,func):
        self.func = func
        self.processes = None

def broadcast(active_clients, data):
    loop = asyncio.get_event_loop()
    tasks = []
    for client_ip in active_clients:
        if client_ip not in CLIENTS:
            pass
        else:
            websocket = CLIENTS[client_ip]
            asyncio.ensure_future(send(websocket, data))
            

async def handler(websocket, path):
    try:
        flag = 0
        async for rawdata in websocket:
            try:
                data = json.loads(rawdata)
                flag = 10
            except Exception as e:
                data = {}
                flag = 0

            client_ip, client_port = websocket.remote_address
            CLIENTS[client_ip] = websocket

            active_clients = [client_ip]

            if flag:
                if k in list(DATA):
                    broadcast(active_clients, json.dumps({"data" : data}))
        
            print(client_ip, client_port, "data", len(data), datetime.datetime.now())
            
    except websockets.exceptions.ConnectionClosedError as e:
        print("E222", e)
        
    except websockets.exceptions.ConnectionClosed as e:
        print("E224", e)

    except Exception as e:
        print("E229", e)
        
    finally:
        pass

class widget:
    def __init__(self):
        self.root = Tk()
        self.width = 150
        self.height = 200
        self.canvas = Canvas(self.root, bg="#e9d3f1", height=self.height, width=self.width)
        self.canvas.bind("<Enter>", self.enter)
        self.canvas.bind("<Leave>", self.leave)
        self.canvas.bind("<Button>", self.clicked)
        self.canvas.pack()
        self.root.wm_title(MYNAME)
        self.img = {}
        self.hover = False
        self.img_frames = {}
        self.animation = None
        self.root.wm_attributes("-transparentcolor", "#e9d3f1")
        self.n = 0
        self.click_register = 0
        self.click_register_time = 50
        self.single_animation_ev_counter = 0
        self.event_flag = 0
        self.text = "POK POIK"
        self.root_border_x_before = 200
        self.root_border_y_before = 200
        self.root_border_x_after = 200
        self.root_border_y_after = 200
        self.root_border_x = 0
        self.root_border_y = 0
        self.root_overrideredirect = None


    def betarun_init(self):
        self.q = multiprocessing.Queue()
        self.RUNNING_PROCESSES = {}
        self.PROCESS_DATA = {}
        self.total_restart = 0
        self.BETARUN_PATH = "betarun"
        self.SAVE_PATH = "running"
        print(self.BETARUN_PATH, self.SAVE_PATH)


    def betarun_cycle(self):
        self.alive = 0
        self.started = 0
        self.killed = 0
        #os.chdir(self.BETARUN_PATH) 
        for f in os.listdir(self.BETARUN_PATH):
            fn, ext = os.path.splitext(f)
            if ext.lower() == ".py" and f != MYNAME:
                fp = os.path.join(self.BETARUN_PATH,f)
                if f not in self.RUNNING_PROCESSES:
                    shutil.copy(fp, self.SAVE_PATH)
                    print("LOAD", fn)
                    func = importlib.import_module("%s.%s" % (self.BETARUN_PATH, fn))
                    self.RUNNING_PROCESSES[f] = load(func)
                    print(dir(self.RUNNING_PROCESSES[f].func))
                if f not in self.PROCESS_DATA:
                    self.PROCESS_DATA[f] = {}
                    self.PROCESS_DATA[f]["alive"] = 0
                    self.PROCESS_DATA[f]["dt"] = datetime.datetime.now()
                    self.PROCESS_DATA[f]["restart"] = 0
                run = False
                
                if hasattr(self.RUNNING_PROCESSES[f].func, "run"):
                    if self.RUNNING_PROCESSES[f].processes:
                        if not self.RUNNING_PROCESSES[f].processes.is_alive():
                            print("DEAD", fn)

                            func = importlib.import_module("%s.%s" % (self.BETARUN_PATH, fn))
                            print("RESURRECT",fn)
                            self.RUNNING_PROCESSES[f] = load(func)
                            
                            self.total_restart += 1
                            self.PROCESS_DATA[f]["restart"] += 1
                            
                            run = True
                        else:
                            self.alive += 1
                            self.PROCESS_DATA[f]["alive"] = 1
                            self.PROCESS_DATA[f]["dt"] = datetime.datetime.now()
                    else:
                        print(fn, "NONE?")
                        run = True
                if run:
                    self.started += 1
                    importlib.invalidate_caches()
                    self.RUNNING_PROCESSES[f].processes = multiprocessing.Process(target = self.RUNNING_PROCESSES[f].func.run)
                    self.RUNNING_PROCESSES[f].processes.start()
                    importlib.reload(self.RUNNING_PROCESSES[f].func)
                    #print(f, self.RUNNING_PROCESSES[f].processes.is_alive())
                    print("STARTED", fn, datetime.datetime.now())
                    self.action()
        for f in self.RUNNING_PROCESSES:
            if f not in os.listdir(self.BETARUN_PATH):
                if self.RUNNING_PROCESSES[f].processes:
                    if self.RUNNING_PROCESSES[f].processes.is_alive():
                        print("KILL", fn)
                        self.killed += 1
                        self.RUNNING_PROCESSES[f].processes.terminate()
                        
        self.text = "Alive: %d\nRestarted: %d" %(self.alive, self.total_restart)
        #print("ALIVE : ", alive, "STARTED : ", started, "KILLED : ", killed, "total restart : ", total_restart)
        #print(self.q.get())
        
    def load_img(self, path, setname = None):
        img_path = []
        img_arr = []
        img_fol = os.listdir(path)
        for f in img_fol:
            fn, ext  = os.path.splitext(f)
            if ext.lower() in (".png",".jpg",".gif"):
                img_path.append(os.path.join(path,f))
        img_path.sort()
        for fp in img_path:
            img = Image.open(fp).resize((self.width, self.height), Image.ANTIALIAS)
            img_arr.append(ImageTk.PhotoImage(img))
        if not setname:
            setname = path
        self.img[setname] = img_arr
        self.img_frames[setname] = len(img_arr)
        self.animation = setname

    def clicked(self,event):
        print("CLICKED",event)
        self.event_flag = 1
        self.n = -1
        self.single_animation_ev_counter = self.img_frames["clicked"]
        self.animation = "clicked"

    def enter(self,event):
        self.set_overrideredirect(False)
        #self.root.wm_attributes("-topmost", False)
        #self.root.wm_attributes("-disabled", False)
        #self.root.overrideredirect(False)
        self.click_register = self.click_register_time
        self.hover = True
        self.event_flag = 1
        self.n = -1
        self.single_animation_ev_counter = self.img_frames["action"]
        self.animation = "action"

    def action(self):
        self.event_flag = 1
        self.n = -1
        self.single_animation_ev_counter = self.img_frames["action"]
        self.animation = "action"

    def leave(self,event):
        self.hover = False

    def change_event(self):
        self.animation = "idle"


    def set_overrideredirect(self, f):
        self.root.update()
        if f == self.root_overrideredirect:
            return False
        if f:
            self.root_border_x_before = w.root.winfo_rootx()
            self.root_border_y_before = w.root.winfo_rooty()

            self.root.wm_attributes("-topmost", True)
            #self.root.wm_attributes("-disabled", True)
            self.root.overrideredirect(True)

            self.root_border_x_after = w.root.winfo_rootx()
            self.root_border_y_after = w.root.winfo_rooty()
            
            self.root_border_x = self.root_border_x_before - self.root_border_x_after
            self.root_border_y = self.root_border_y_before - self.root_border_y_after

            self.root.update()
            
            self.root.geometry(self.root.wm_geometry().split("+")[0] + "+%d+%d" %(w.root.winfo_rootx() + self.root_border_x, w.root.winfo_rooty() + self.root_border_y))
        else:
            self.root.geometry(self.root.wm_geometry().split("+")[0] + "+%d+%d" %(w.root.winfo_rootx() - self.root_border_x, w.root.winfo_rooty() - self.root_border_y))
            self.root.update()
            self.root.wm_attributes("-topmost", False)
            #self.root.wm_attributes("-disabled", False)
            self.root.overrideredirect(False)

        self.root_overrideredirect = f
        
    def animate(self):
        self.betarun_cycle()
        self.n += 1
        try:
            self.root.after_cancel(self.l)
        except Exception as e:
            print(e)
        self.canvas.delete("all")
        img = self.img[self.animation][self.n % self.img_frames[self.animation]]
        self.canvas.create_image(0, 0, image=img, anchor=NW)
        self.canvas.create_text(5, 5, fill="black", font="Times 9 italic bold", text= self.text, anchor=NW)
        self.l = self.root.after(60, self.animate)
        

        if self.click_register > 0 and not self.hover:
            self.click_register -= 1
        if self.click_register <= 0:
            #self.root.wm_attributes("-topmost", True)
            #self.root.wm_attributes("-disabled", True)
            #self.root.overrideredirect(True)
            #self.set_overrideredirect(True)
            pass


        if self.single_animation_ev_counter > 0:
            self.single_animation_ev_counter -= 1
        elif self.single_animation_ev_counter <= 0 and self.event_flag != 0:
            self.event_flag = 0
            self.change_event()
        elif self.event_flag == 0 and self.single_animation_ev_counter <= 0 and self.n % self.img_frames[self.animation] == 0:
            if random.random() > 0.88:
                self.event_flag = 1
                self.n = -1
                self.single_animation_ev_counter = self.img_frames["blink"]
                self.animation = "blink"

    def run(self):
        self.load_img(r"image\click","clicked")
        self.load_img(r"image\blink","blink")
        self.load_img(r"image\action","action")
        self.load_img(r"image\idle","idle")
        self.betarun_init()
        self.animate()
        self.root.resizable(False, False)
        #print("RUN DATA 8789 SERVER")
        #start_server = websockets.serve(handler, port = 8789)
        #asyncio.get_event_loop().run_until_complete(start_server)
        #asyncio.get_event_loop().run_forever()
        self.root.mainloop()


        
if __name__ == "__main__":
    if MYNAME not in titles:
        w = widget()
        w.run()
        pass

    else:
        print("IS RUNNING")
        


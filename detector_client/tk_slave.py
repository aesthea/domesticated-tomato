import os
import sys
import time
import datetime
from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
from PIL import Image
from PIL import ImageTk
from win32api import GetMonitorInfo, MonitorFromPoint
import math
#import sqlite3
#import pymysql
import threading
#import difflib
import pickle
#import re
#import barcode
#from barcode.writer import ImageWriter
#import csv
from functools import partial

#import asyncio
#import websockets

#import multiprocessing
import cv2
import json
import socket
from io import BytesIO
import base64
from websocket import create_connection
import websocket
import pyfiles.camera
import pyfiles.socket_client


monitor_info = GetMonitorInfo(MonitorFromPoint((0,0)))
monitor_area = monitor_info.get("Monitor")
work_area = monitor_info.get("Work")

if not os.path.isdir("c:/test"):
    os.mkdir("c:/test")
    
class ThreadWithReturnValue(threading.Thread):
    #https://stackoverflow.com/questions/6893968/how-to-get-the-return-value-from-a-thread-in-python
    def __init__(self, group=None, target=None, name=None, args=(), kwargs={}, Verbose=None):
        threading.Thread.__init__(self, group, target, name, args, kwargs)
        self._return = None
    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)
    def join(self, *args):
        threading.Thread.join(self, *args)
        return self._return

class widget:
    def __init__(self):
        self.root = Tk()
        self.root.title("SLAVE MODULE 0.20240118")
        self.width = work_area[2]
        self.height = work_area[3] - 25

        self.frame_main = Frame(master = self.root, bg = "blue", width = 800, height = 600)
        self.frame_main.grid_propagate(0)
        self.frame_main.pack_propagate(0)
        self.frame_main.pack()

        self.bold_font = "Helvetica 13 bold"
        self.normal_font = "Helvetica 12"
        self.label_font = "Helvetica 12 bold"
        self.small_font = "Helvetica 10"
        self.very_small_font = "Helvetica 8"
        self.inputs = {}
        self.objects = {}
        self.loaded_gk_lot_data = {}
        self.machine_mode = ""
        
        self.data = {}
        self.initialdir = "/"

        self.cycletime = 100
        self.l = None
        self.ws = None
        self.ws_retry_wait = datetime.datetime.now()
        self.ai_detect_rate = 2
        self.ai_detect_last_sample = datetime.datetime.now()
        self.IMAGEOUT_SIZE = (240,240)
        self.STRINGVAR = StringVar(self.root)
        self.CONSOLE_DATA = StringVar(self.root)
        self.THREAD = None

    def worker(self, func, args):
        if not self.THREAD:
            return ThreadWithReturnValue(target = func, args = args)
        else:
            return None

    def serve_ai_button(self):
        tk_socket_server.run()

    def numerical_set(self, input_name, value, integer = False):
        if input_name in self.inputs:
            if integer:
                value = self.inputs[input_name].set("{:,}".format(value))
            else:
                value = self.inputs[input_name].set("{:,.4f}".format(value))
        else:
            return False
        
    def numerical_get(self, input_name, integer = False):
        if input_name in self.inputs:
            value = self.inputs[input_name].get()
        else:
            return 0
        try:
            if not integer:
                _format = "{:,.4f}"
                value = float(value.replace(",",""))
            else:
                _format = "{:,}"
                value = math.floor(float(value.replace(",","")))
            self.inputs[input_name].set(_format.format(value))
        except Exception as e:
            #print(e)
            value = 0
            self.inputs[input_name].set("")
        return value

    def forget(self, l):
        for input_name in l:
            if input_name in self.objects:
                self.objects[input_name]["main_frame"].pack_forget()
                self.objects[input_name]["main_frame"].grid_forget()

    def remember(self,  l):
        for input_name in l:
            if input_name in self.objects:
                column = self.objects[input_name]["column"]
                row = self.objects[input_name]["row"]
                self.objects[input_name]["main_frame"].grid(column = column, row = row)

    def destroy(self, l):
        unbind_list = ("<FocusIn>", "<Return>", "<FocusOut>", "<Tab>")
        for input_name in l:
            if input_name in self.objects:
                obj = self.objects[input_name]
                for item in obj["items"]:
                    for u in unbind_list:
                        try:
                            self.objects[input_name][item].unbind(u)
                            #print("destroy: unbound ", input_name, u)
                        except Exception as e:
                            pass
                    self.objects[input_name][item].destroy()
                self.inputs[input_name].set("")
                
    def color(self, input_name, color):
        for item in self.objects[input_name]["items"]:
            self.objects[input_name][item].config(bg = color)
        
    def label_w_input(self, master, label_width, input_width, height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind_focusout = None, bind_return = None, bind_tab = None):
        if not font_size:
            font_size = self.normal_font
            
        f0 = Frame(master = master, width = label_width + input_width, height = height)
        f0.grid_propagate(0)
        f0.pack_propagate(0)
        f0.grid(column = column, row = row, columnspan = columnspan, rowspan = rowspan)
        
        f1 = Frame(master = f0, width = label_width, height = height) #bg = self.console_color, 
        f1.grid_propagate(0)
        f1.pack_propagate(0)
        f1.grid(column = 0, row = 0)
        
        l1 = Label(master = f1, text = label_text, anchor = NE, font = font_size) #bg = self.console_color,
        l1.pack(fill = BOTH, expand = True)

        f2 = Frame(master = f0, width = input_width, height = height)
        f2.grid_propagate(0)
        f2.pack_propagate(0)
        f2.grid(column = 1, row = 0)
        
        self.inputs[input_name] = StringVar(self.root)
        i1 = Entry(master = f2, textvariable = self.inputs[input_name], bd = 2, font = font_size, relief=GROOVE, justify="center")
        i1.pack(fill = BOTH, expand = True)

        i1.bind("<FocusIn>", lambda v : i1.select_range(0, "end"))
        if bind_focusout:
            i1.bind("<FocusOut>", bind_focusout)
        if bind_return:
            i1.bind("<Return>", bind_return)
        if bind_tab:
            i1.bind("<Tab>", bind_tab)
        
        package = {}
        package["main_frame"] = f0
        package["label_frame"] = f1
        package["input_frame"] = f2
        package["label"] = l1
        package["input"] = i1
        package["row"] = row
        package["column"] = column
        package["items"] = ("input", "label", "input_frame", "label_frame", "main_frame")
        self.objects[input_name] = package

    def vlabel_w_input(self, master, column_width, height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind_focusout = None, bind_return = None, bind_tab = None):
        if not font_size:
            font_size = self.normal_font
            
        f0 = Frame(master = master, width = column_width, height = int(height * 2))
        f0.grid_propagate(0)
        f0.pack_propagate(0)
        f0.grid(column = column, row = row, columnspan = columnspan, rowspan = rowspan)
        
        f1 = Frame(master = f0, width = column_width, height = height) #bg = self.console_color, 
        f1.grid_propagate(0)
        f1.pack_propagate(0)
        f1.grid(column = 0, row = 0)
        
        l1 = Label(master = f1, text = label_text, anchor = NE, font = font_size) #bg = self.console_color,
        l1.pack(fill = BOTH, expand = True)

        f2 = Frame(master = f0, width = column_width, height = height)
        f2.grid_propagate(0)
        f2.pack_propagate(0)
        f2.grid(column = 0, row = 1)
        
        self.inputs[input_name] = StringVar(self.root)
        i1 = Entry(master = f2, textvariable = self.inputs[input_name], bd = 2, font = font_size, relief=GROOVE, justify="center")
        i1.pack(fill = BOTH, expand = True)

        i1.bind("<FocusIn>", lambda v : i1.select_range(0, "end"))
        if bind_focusout:
            i1.bind("<FocusOut>", bind_focusout)
        if bind_return:
            i1.bind("<Return>", bind_return)
        if bind_tab:
            i1.bind("<Tab>", bind_tab)
        
        package = {}
        package["main_frame"] = f0
        package["label_frame"] = f1
        package["input_frame"] = f2
        package["label"] = l1
        package["input"] = i1
        package["row"] = row
        package["column"] = column
        package["items"] = ("input", "label", "input_frame", "label_frame", "main_frame")
        self.objects[input_name] = package


    def vlabel_w_textarea(self, master, column_width, height, t_height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind_focusout = None, bind_return = None, bind_tab = None):
        if not font_size:
            font_size = self.normal_font
            
        f0 = Frame(master = master, width = column_width, height = int(height + t_height))
        f0.grid_propagate(0)
        f0.pack_propagate(0)
        f0.grid(column = column, row = row, columnspan = columnspan, rowspan = rowspan)
        
        f1 = Frame(master = f0, width = column_width, height = height) #bg = self.console_color, 
        f1.grid_propagate(0)
        f1.pack_propagate(0)
        f1.grid(column = 0, row = 0)
        
        l1 = Label(master = f1, text = label_text, anchor = NW, font = font_size) #bg = self.console_color,
        l1.pack(fill = BOTH, expand = True)

        f2 = Frame(master = f0, width = column_width, height = t_height)
        f2.grid_propagate(0)
        f2.pack_propagate(0)
        f2.grid(column = 0, row = 1)
       
        self.inputs[input_name] = StringVar(self.root)

        #self.textarea = Text(self.frame3, width = 600, height = 100)
        #self.textarea.grid_propagate(0)
        #self.textarea.pack_propagate(0)
        #self.textarea.grid(column = 0, row = 1, columnspan = 1)
        i1 = Text(master = f2, width = column_width, height = t_height, font = self.very_small_font)
        #i1 = Entry(master = f2, textvariable = self.inputs[input_name], bd = 2, font = font_size, relief=GROOVE, justify="center")
        i1.pack(fill = BOTH, expand = True)

        #i1.bind("<FocusIn>", lambda v : i1.select_range(0, "end"))
        if bind_focusout:
            i1.bind("<FocusOut>", bind_focusout)
        if bind_return:
            i1.bind("<Return>", bind_return)
        if bind_tab:
            i1.bind("<Tab>", bind_tab)
        
        package = {}
        package["main_frame"] = f0
        package["label_frame"] = f1
        package["input_frame"] = f2
        package["label"] = l1
        package["input"] = i1
        package["row"] = row
        package["column"] = column
        package["items"] = ("input", "label", "input_frame", "label_frame", "main_frame")
        self.objects[input_name] = package

    def label(self, master, label_width, height, row, column, label_text, font_size = None, columnspan = 1, rowspan = 1):
        if not font_size:
            font_size = self.label_font
            
        f0 = Frame(master = master, width = label_width, height = height)
        f0.grid_propagate(0)
        f0.pack_propagate(0)
        f0.grid(column = column, row = row, columnspan = columnspan, rowspan = rowspan)
        
        f1 = Frame(master = f0, width = label_width, height = height) #bg = self.console_color, 
        f1.grid_propagate(0)
        f1.pack_propagate(0)
        f1.grid(column = 0, row = 0)
        
        l1 = Label(master = f1, text = label_text, anchor = NW, font = font_size) #bg = self.console_color,
        l1.pack(fill = BOTH, expand = True)
        
        package = {}
        package["main_frame"] = f0
        package["label_frame"] = f1
        package["label"] = l1
        package["row"] = row
        package["column"] = column
        package["items"] = ("label", "label_frame", "main_frame")
        self.objects[label_text] = package

    def window(self):
        self.column_width = math.floor(self.width / 12)
        self.row_height = math.floor(self.height / 35)
        self.frame1 = Frame(master = self.frame_main, bg = self.console_color, width = self.column_width * 12, height = self.row_height * 4, highlightbackground="#e0e0e0", highlightthickness=1)
        self.frame1.grid_propagate(0)
        self.frame1.pack_propagate(0)
        self.frame1.grid(column = 0, row = 0)

    def build(self):        
        self.frame1 = Frame(master = self.frame_main, width = 640, height = 100)
        self.frame1.grid_propagate(0)
        self.frame1.pack_propagate(0)
        self.frame1.grid(column = 0, row = 0)
        
        self.frame2 = Frame(master = self.frame_main, width = 160, height = 580)
        self.frame2.grid_propagate(0)
        self.frame2.pack_propagate(0)
        self.frame2.grid(column = 1, row = 0, rowspan = 2)

        self.frame3 = Frame(master = self.frame_main, width = 640, height = 480)
        self.frame3.grid_propagate(0)
        self.frame3.pack_propagate(0)
        self.frame3.grid(column = 0, row = 1)

        self.frame4 = Frame(master = self.frame_main, width = 800, height = 20)
        self.frame4.grid_propagate(0)
        self.frame4.pack_propagate(0)
        self.frame4.grid(column = 0, row = 2, columnspan = 2)


        #FRAME1
        self.label(self.frame1, 240, 25, 0, 0, "select Camera")
        
        self.camera_select_frame = Frame(master = self.frame1, width = 400, height = 25)
        self.camera_select_frame.grid_propagate(0)
        self.camera_select_frame.pack_propagate(0)
        self.camera_select_frame.grid(column = 1, row = 0)
        self.camera_select_value = IntVar(self.root)
        self.camera_select_value.set(0)


        self.label(self.frame1, 240, 25, 1, 0, "enable - ")
        
        self.ai_checkbox_value = BooleanVar(self.root)
        self.ai_checkbox_value.set(False)
        self.ai_checkbox_checkbutton = Checkbutton(self.frame1, text='AI function',variable=self.ai_checkbox_value, onvalue=True, offvalue=False, font = self.normal_font, command = self.update)
        self.ai_checkbox_checkbutton.grid(column = 1, row = 1)


        self.label_w_input(self.frame1, 240, 400, 25, 2, 0, "AI source IP", "ai_source_ipaddress", bind_focusout=self.update, bind_return=self.update, bind_tab=self.update, columnspan = 2)
        self.label_w_input(self.frame1, 240, 400, 25, 3, 0, "AI source PORT", "ai_source_port", bind_focusout=self.update, bind_return=self.update, bind_tab=self.update, columnspan = 2)
        

        #FRAME2
        self.b1_frame = Frame(master = self.frame2, width = 160, height = 50)
        self.b1_frame.grid_propagate(0)
        self.b1_frame.pack_propagate(0)
        self.b1_frame.grid(column = 0, row = 0)
        self.train_button = Button(master = self.b1_frame, text = "DETECT", relief="groove", font = self.bold_font, bd = 2, command = self.detect)
        self.train_button.pack(fill = BOTH, expand = True)
        

        #FRAME3
        self.console_image_canvas = Canvas(master = self.frame3, bg = "black", width = 640, height = 480)
        self.console_image_canvas.grid_propagate(0)
        self.console_image_canvas.pack_propagate(0)
        self.console_image_canvas.grid(column = 0, row = 0)
        self.console_image_canvas.bind('<Button>', self.canvas_click)

        #FRAME4
        self.log_text = Label(master = self.frame4, textvariable = self.STRINGVAR, font="Times 10")
        self.log_text.pack(fill = BOTH, expand = True)
        

    def update(self, event = None):
        self.save_setting()
        #print(event)

    def canvas_click(self, event = None):
        print(event)

    def cycle(self):
        if self.THREAD:
            if self.THREAD.is_alive():
                #print("thread alive")
                pass
            else:
                #print("END THREAD")
                v = self.THREAD.join()
                #print("THREAD VAL", v)
                self.THREAD = None
                if v == "ai_predict_done":
                    self.ai_detect_last_sample = datetime.datetime.now()
                    #print("AI DOONE")
                elif v == "ai_predict_error":
                    self.ai_detect_last_sample = datetime.datetime.now() + datetime.timedelta(0, 5)
                elif v == "ai_connect_error":
                    self.ai_detect_last_sample = datetime.datetime.now() + datetime.timedelta(0, 2)
                

    def worker(self, func, args):
        if not self.THREAD:
            return ThreadWithReturnValue(target = func, args = args)
        else:
            return None

    def predict(self, args):
        self.STRINGVAR.set("RUN THREAD")
        if not args:
            return "ai_predict_error"
        try:
            v = self.ws.send(args)
        except websocket._exceptions.WebSocketConnectionClosedException as e:
            self.STRINGVAR.set("E432, " + e)
            return "ai_connect_error"
        try:
            v = json.loads(v)
            self.CONSOLE_DATA.set(json.dumps(v["rawdata"]))
            self.STRINGVAR.set(v["rawdata"])
            return "ai_predict_done"
        except Exception as e:
            self.STRINGVAR.set("E436, " + e)
            return "ai_predict_error"


    def hoohah(self, args):
        self.STRINGVAR.set(args)
        return "YEHAWWW"


    def cycle_detect(self):
        if self.ai_checkbox_value.get():
            if (datetime.datetime.now() - self.ai_detect_last_sample).total_seconds() * 1000 > self.ai_detect_rate * 1000:
                self.ai_detect_last_sample = datetime.datetime.now() + datetime.timedelta(0, 30)
                self.detect()
                 

    def detect(self, event = None):
        ret = None
        for i in range(10):
            if type(self.camera.device_index) != int:
                continue
            ret, im = self.camera.read()
            if not ret:
                continue
            break
        if not ret:
            return None

        im = im.astype("uint8")
        b,g,r = cv2.split(im)
        im = cv2.merge((r,g,b))
        im = Image.fromarray(im)

        img_byte_arr = BytesIO()
        im.resize(self.IMAGEOUT_SIZE).save(img_byte_arr, format='JPEG', quality = 75, optimize = True, progressive = True)

        img_byte_arr = img_byte_arr.getvalue()
                                
        im_b64 = base64.b64encode(img_byte_arr).decode('utf-8', 'ignore')
        im_b64 = ",".join(['data:image/jpeg;base64', im_b64])
                                
        data = {}
        data["filename"] = "doodad.jpg"
        data["bytes"] = im_b64
        data["tensorflow"] = True
        data["rawdata_only"] = True
        data["size"] = len(b)

        self.initialize_socket_connection()
        if self.ws:
            self.THREAD = self.worker(self.predict, (json.dumps(data), ))
            if self.THREAD:
                self.STRINGVAR.set("START THREAD")
                self.THREAD.start()
                

    def initialize_camera(self):
        self.camera = pyfiles.camera.camera()
        self.camera_radio = {}
        for i, camera_index in enumerate(self.camera.device_array):
            self.camera_radio[camera_index] = Radiobutton(self.camera_select_frame, text="device %01d" % camera_index, variable = self.camera_select_value, value = camera_index, font = self.normal_font, command = self.switch_camera)
            self.camera_radio[camera_index].grid(column = i, row = 0)

        
    def switch_camera(self, event = None):
        self.camera.close()
        self.camera.connect(self.camera_select_value.get())


    def update_canvas(self):
        if type(self.camera.device_index) != int:
            return None
        ret, im = self.camera.read()
        if not ret:
            return None
        im = im.astype("uint8")
        b,g,r = cv2.split(im)
        im = cv2.merge((r,g,b))

        if self.CONSOLE_DATA.get():
            try:
                rawdata = json.loads(self.CONSOLE_DATA.get())
                im = self.image_label(im, rawdata, lw = 2)
            except Exception as e:
                self.STRINGVAR.set("E524, " + e)
                pass
        
        im = Image.fromarray(im)
        
        self.img = ImageTk.PhotoImage(image = im)
        self.console_image_canvas.delete("all")
        self.console_image_canvas.create_image(0, 0, image=self.img, anchor=NW)


    def image_label(self, image, rawdata, lw = 2):
        height, width, chan = image.shape
        for data in rawdata:
            if data:
                if "color" in data:
                    color = data["color"]
                else:
                    color = [255, 0, 0]

                if "x1" in data and "x2" in data and "y1" in data and "y2" in data and "h" in data and "w" in data and "tag" in data and "score" in data:
                    x1 = math.floor((data["x1"] / data["w"]) * width)
                    x2 = math.ceil((data["x2"] / data["w"]) * width)
                    y1 = math.floor((data["y1"] / data["h"]) * height)
                    y2 = math.ceil((data["y2"] / data["h"]) * height)
                    image[y1 : y2, x1 : x1 + lw] = color
                    image[y1 : y2, x2 - lw : x2] = color
                    image[y1 : y1 + lw, x1 : x2] = color
                    image[y2 - lw: y2, x1 : x2] = color

                    cv2.putText(image, data["tag"], (x1 + 1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 1, color, 1)
                    cv2.putText(image, "%02d%%" % (data["score"] * 100), (x1 + 1, y1 + 18), cv2.FONT_HERSHEY_PLAIN, 0.6, color)
        return image
    

    def initialize_socket_connection(self):
        if self.ws_retry_wait:
            if (datetime.datetime.now() - self.ws_retry_wait).total_seconds() > 10:
                pass
            else:
                return None
        if self.ws:
            try:
                initdata = {"__INIT__" : "__INIT__"}
                v = self.ws.send(initdata)
                return None
            except Exception as e:
                pass
            
        socket_ip = self.inputs["ai_source_ipaddress"].get()
        socket_port = self.inputs["ai_source_port"].get()
        if len(socket_ip) >= 8 and len(socket_port) > 3:
            pass
        else:
            self.ws_retry_wait = datetime.datetime.now()
            return None
            
        try:
            self.ws = pyfiles.socket_client.client_connection(socket_ip, socket_port)
            print("Connection success")
        except socket.timeout as e:
            print("E451, failed to connect to a socket server, ", e)
            self.ws = None
        except Exception as e:
            print("E453", e)
            self.ws = None
        self.ws_retry_wait = datetime.datetime.now()
        

    def save_setting(self):
        data = {}
        if self.inputs["ai_source_ipaddress"].get():
            data["ai_source_ipaddress"] = self.inputs["ai_source_ipaddress"].get()
        if self.inputs["ai_source_port"].get():
            data["ai_source_port"] = self.inputs["ai_source_port"].get()
        with open("slave.pik", "wb") as fio:
            pickle.dump(data, fio)

    def load_setting(self):
        if os.path.isfile("slave.pik"):
            with open("slave.pik", "rb") as fio:
                data = pickle.load(fio)
            for k in data:
                if k in self.inputs:
                    self.inputs[k].set(data[k])
        
        
    def loop(self):
        self.cycle()
        self.cycle_detect()
        self.update_canvas()
        if not self.ws:
            self.initialize_socket_connection()
        if self.l:
            try:
                self.root.after_cancel(self.l)
            except Exception as e:
                print("ERROR widget.loop", e)
        self.l = self.root.after(self.cycletime , self.loop)

        
    def top(self):
        self.toplevel_top = Toplevel(self.root)
        self.toplevel_a = Frame(master = self.toplevel_top, bg = "blue", width = 480, height = 180)
        self.toplevel_a.grid_propagate(0)
        self.toplevel_a.pack_propagate(0)
        self.toplevel_a.grid(column = 0, row = 0)
        self.toplevel_b = Frame(master = self.toplevel_top, bg = "red", width = 480, height = 180)
        self.toplevel_b.grid_propagate(0)
        self.toplevel_b.pack_propagate(0)
        self.toplevel_b.grid(column = 0, row = 1)

        self.toplevel_a1 = Frame(master = self.toplevel_a, bg = "yellow", width = 160, height = 180)
        self.toplevel_a1.grid_propagate(0)
        self.toplevel_a1.pack_propagate(0)
        self.toplevel_a1.grid(column = 0, row = 0)
        self.dst_button = Button(master = self.toplevel_a1, text = "DST", relief="groove", font = self.bold_font, bd = 5, command = self.DST_mode)
        self.dst_button.pack(fill = BOTH, expand = True)
        self.toplevel_a2 = Frame(master = self.toplevel_a, bg = "green", width = 160, height = 180)
        self.toplevel_a2.grid_propagate(0)
        self.toplevel_a2.pack_propagate(0)
        self.toplevel_a2.grid(column = 1, row = 0)
        self.humo6_button = Button(master = self.toplevel_a2, text = "HUMO6", relief="groove", font = self.bold_font, bd = 5, command = self.HUMO6_mode)
        self.humo6_button.pack(fill = BOTH, expand = True)
        self.toplevel_a3 = Frame(master = self.toplevel_a, bg = "cyan", width = 160, height = 180)
        self.toplevel_a3.grid_propagate(0)
        self.toplevel_a3.pack_propagate(0)
        self.toplevel_a3.grid(column = 2, row = 0)
        self.humo8_button = Button(master = self.toplevel_a3, text = "HUMO8", relief="groove", font = self.bold_font, bd = 5, command = self.HUMO8_mode)
        self.humo8_button.pack(fill = BOTH, expand = True)


    def top_assign_apply_button_press(self):
        self.calculate()
        self.toplevel_top.destroy()

   
def run():
    w = widget()
    w.build()
    #w.inputs["ai_source_ipaddress"].set("192.168.160.76")
    #w.inputs["ai_source_port"].set("9890")
    w.load_setting()
    w.initialize_camera()
    w.loop()
    w.root.mainloop()

    




if __name__ == "__main__":
    run()
    #pass

import os
import sys
from os import system
#import time
import datetime
from tkinter import *
from tkinter import messagebox
from tkinter import simpledialog
from tkinter import filedialog
#from PIL import Image
#from PIL import ImageTk
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
#from functools import partial
import HARR_VOTT
import tk_socket_server
import numpy as np
import camera
import cv2
from PIL import Image
from PIL import ImageTk
import json

#import asyncio
#import websockets
#import multiprocessing

monitor_info = GetMonitorInfo(MonitorFromPoint((0,0)))
monitor_area = monitor_info.get("Monitor")
work_area = monitor_info.get("Work")
DEBOUNCE_TIME = 500
CYCLETIME = 100
    
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
        self.root.title("HARR VOTT 1.2024f")
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
        self.DET_MODEL = None
        self.predict = None
        self.MP_THREAD = False
        self.initialdir = "/"
        self.RELOAD_ML_FLAG = False

        self.augment_01_var = StringVar(self.root)
        self.augment_01_var.set("1")
        self.augment_02_var = StringVar(self.root)
        self.augment_02_var.set("1")
        self.augment_03_var = StringVar(self.root)
        self.augment_03_var.set("1")
        self.augment_04_var = StringVar(self.root)
        self.augment_04_var.set("1")
        self.augment_05_var = StringVar(self.root)
        self.augment_05_var.set("1")
        self.augment_06_var = StringVar(self.root)
        self.augment_06_var.set("1")        
        self.augment_07_var = StringVar(self.root)
        self.augment_07_var.set("1")
        self.augment_08_var = StringVar(self.root)
        self.augment_08_var.set("1")

    def worker(self, func, args):
        if not self.THREAD:
            return ThreadWithReturnValue(target = func, args = args)
        else:
            return None

    def serve_model_button(self):
        self.root.withdraw()
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

    
    def label_w_slider(self, master, label_width, input_width, height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind = None, min_value=0, max_value=100, interval=1):
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

        f2 = Frame(master = f0, width = input_width - 50, height = height)
        f2.grid_propagate(0)
        f2.pack_propagate(0)
        f2.grid(column = 1, row = 0)

        f3 = Frame(master = f0, width = 50, height = height)
        f3.grid_propagate(0)
        f3.pack_propagate(0)
        f3.grid(column = 2, row = 0)
        
        self.inputs[input_name] = DoubleVar(self.root)
        i1 = Scale(master = f2, from_=min_value, to=max_value, resolution=interval, orient=HORIZONTAL, variable = self.inputs[input_name], showvalue=False)
        i1.set(min_value)
        i1.pack(fill = BOTH, expand = True)
        self.inputs[input_name].set(min_value)

        i2 = Entry(master = f3, textvariable = self.inputs[input_name], bd = 2, font = font_size, relief=GROOVE, justify="center")
        i2.pack(fill = BOTH, expand = True)

        i2["state"] = "disabled"

        #i1.bind("<FocusIn>", lambda v : i1.select_range(0, "end"))
        if bind:
            i1['command'] = lambda x : self.debounce(bind)
        
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

    def label_w_input(self, master, label_width, input_width, height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind = None):
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
        if bind:
            i1.bind("<FocusOut>", bind)
            i1.bind("<Return>", bind)
            i1.bind("<Tab>", bind)
            
        
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

    def vlabel_w_input(self, master, column_width, height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind = None):
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
        if bind:
            i1.bind("<FocusOut>", bind)
            i1.bind("<Return>", bind)
            i1.bind("<Tab>", bind)
        
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


    def vlabel_w_textarea(self, master, column_width, height, t_height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind = None):
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
        if bind:
            i1.bind("<FocusOut>", bind)
            i1.bind("<Return>", bind)
            i1.bind("<Tab>", bind)
        
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


    def logslider(self, master, label_width, input_width, height, row, column, label_text, input_name, font_size = None, columnspan = 1, rowspan = 1, bind = None, min_value=0.1, max_value=0.00001):
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

        f2 = Frame(master = f0, width = input_width - 80, height = height)
        f2.grid_propagate(0)
        f2.pack_propagate(0)
        f2.grid(column = 1, row = 0)

        f3 = Frame(master = f0, width = 80, height = height)
        f3.grid_propagate(0)
        f3.pack_propagate(0)
        f3.grid(column = 2, row = 0)


        log_max = math.floor(math.log(max_value, 10))
        log_min = math.floor(math.log(min_value, 10))

        temp = input_name + '__TEMP__'
        self.inputs[temp] = DoubleVar(self.root)
        self.inputs[input_name] = DoubleVar(self.root)
        i1 = Scale(master = f2, from_=log_min, to=log_max, resolution=1, orient=HORIZONTAL, variable = self.inputs[temp], showvalue=False)
        i1.set(min_value)
        i1.pack(fill = BOTH, expand = True)

        self.inputs[temp].set(log_min)
        self.update_log_value(self.inputs[temp], self.inputs[input_name])
        i1['command'] = lambda x : self.update_log_value(self.inputs[temp], self.inputs[input_name], bind)

        i2 = Entry(master = f3, textvariable = self.inputs[input_name], bd = 2, font = font_size, relief=GROOVE, justify="center")
        i2.pack(fill = BOTH, expand = True)

        i2["state"] = "disabled"
            
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

    def update_log_value(self, source, target, bind = None):
        target.set(pow(10, source.get()))
        if bind:
            self.debounce(bind)


    def debounce(self, func, t = DEBOUNCE_TIME):
        try:
            self.root.after_cancel(self._debounce_af)
        except Exception as e:
            #print(e)
            pass
        self._debounce_af = self.root.after(t, func)


#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------------------------------------------------------------------
        

    def build(self):

        self.inputs["augment"] = DoubleVar(self.root)
        self.inputs["augment"].set(255)
        
        self.frame1 = Frame(master = self.frame_main, bg = "red", width = 600, height = 100)
        self.frame1.grid_propagate(0)
        self.frame1.pack_propagate(0)
        self.frame1.grid(column = 0, row = 0)
        
        self.frame2 = Frame(master = self.frame_main, bg = "yellow", width = 200, height = 600)
        self.frame2.grid_propagate(0)
        self.frame2.pack_propagate(0)
        self.frame2.grid(column = 1, row = 0, rowspan = 3)

        self.frame3 = Frame(master = self.frame_main, bg = "orange", width = 600, height = 500)
        self.frame3.grid_propagate(0)
        self.frame3.pack_propagate(0)
        self.frame3.grid(column = 0, row = 1)

        #FRAME1
        self.label(self.frame1, 600, 25, 0, 0, "HARRVOTT 1.2024F")
        self.label_w_input(self.frame1, 200, 400, 25, 1, 0, "SAVEFILE", "savefile", bind=self.update)
        self.label_w_input(self.frame1, 200, 400, 25, 2, 0, "SERVE PORT", "port", bind=self.update)

        #FRAME2
        self.b0_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b0_frame.grid_propagate(0)
        self.b0_frame.pack_propagate(0)
        self.b0_frame.grid(column = 0, row = 0)
        
        self.b1_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b1_frame.grid_propagate(0)
        self.b1_frame.pack_propagate(0)
        self.b1_frame.grid(column = 0, row = 1)
        self.train_button = Button(master = self.b1_frame, text = "TRAIN", relief="groove", font = self.bold_font, bd = 2, command = self.train)
        self.train_button.pack(fill = BOTH, expand = True)

        self.b2_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b2_frame.grid_propagate(0)
        self.b2_frame.pack_propagate(0)
        self.b2_frame.grid(column = 0, row = 2)
        self.save_button = Button(master = self.b2_frame, text = "SAVE", relief="groove", font = self.bold_font, bd = 2, command = self.save)
        self.save_button.pack(fill = BOTH, expand = True)

        self.b3_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b3_frame.grid_propagate(0)
        self.b3_frame.pack_propagate(0)
        self.b3_frame.grid(column = 0, row = 3)
        self.load_button = Button(master = self.b3_frame, text = "LOAD", relief="groove", font = self.bold_font, bd = 2, command = self.load)
        self.load_button.pack(fill = BOTH, expand = True)

        self.b4_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b4_frame.grid_propagate(0)
        self.b4_frame.pack_propagate(0)
        self.b4_frame.grid(column = 0, row = 4)

        self.b5_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b5_frame.grid_propagate(0)
        self.b5_frame.pack_propagate(0)
        self.b5_frame.grid(column = 0, row = 5)
        self.serve_button = Button(master = self.b5_frame, text = "SERVE", relief="groove", font = self.bold_font, bd = 2, command = self.serve_model_button)
        self.serve_button.pack(fill = BOTH, expand = True)

        self.b6_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b6_frame.grid_propagate(0)
        self.b6_frame.pack_propagate(0)
        self.b6_frame.grid(column = 0, row = 6)
        
        self.b7_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b7_frame.grid_propagate(0)
        self.b7_frame.pack_propagate(0)
        self.b7_frame.grid(column = 0, row = 7)
        self.generator_check_button = Button(master = self.b7_frame, text = "Sanity check", relief="groove", font = self.bold_font, bd = 2, command = self.sanity_check)
        self.generator_check_button.pack(fill = BOTH, expand = True)

        self.b8_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b8_frame.grid_propagate(0)
        self.b8_frame.pack_propagate(0)
        self.b8_frame.grid(column = 0, row = 8)
        self.trial_button = Button(master = self.b8_frame, text = "Test folder", relief="groove", font = self.bold_font, bd = 2, command = self.trial)
        self.trial_button.pack(fill = BOTH, expand = True)

        self.b9_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b9_frame.grid_propagate(0)
        self.b9_frame.pack_propagate(0)
        self.b9_frame.grid(column = 0, row = 9)
        self.predict_button = Button(master = self.b9_frame, text = "Predict image", relief="groove", font = self.bold_font, bd = 2, command = self.predict_image)
        self.predict_button.pack(fill = BOTH, expand = True)


        self.b10_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b10_frame.grid_propagate(0)
        self.b10_frame.pack_propagate(0)
        self.b10_frame.grid(column = 0, row = 10)
        
        self.b11_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b11_frame.grid_propagate(0)
        self.b11_frame.pack_propagate(0)
        self.b11_frame.grid(column = 0, row = 11)
        self.train_button_early_stopping = Button(master = self.b11_frame, text = "TRAIN (early stopping)", relief="groove", font = self.bold_font, bd = 2, command = lambda : self.train(early_stopping = True))
        self.train_button_early_stopping.pack(fill = BOTH, expand = True)
        #self.train_button_early_stopping["state"] = "disabled"

        self.b12_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b12_frame.grid_propagate(0)
        self.b12_frame.pack_propagate(0)
        self.b12_frame.grid(column = 0, row = 12)
        self.segment_image_button = Button(master = self.b12_frame, text = "segment image", relief="groove", font = self.bold_font, bd = 2, command = lambda : self.segment_image())
        self.segment_image_button.pack(fill = BOTH, expand = True)

        self.b13_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b13_frame.grid_propagate(0)
        self.b13_frame.pack_propagate(0)
        self.b13_frame.grid(column = 0, row = 13)
        self.video_capture_button = Button(master = self.b13_frame, text = "video detect", relief="groove", font = self.bold_font, bd = 2, command = lambda : self.video_capture())
        self.video_capture_button.pack(fill = BOTH, expand = True)
        
##        self.b14_frame = Frame(master = self.frame2, width = 200, height = 25)
##        self.b14_frame.grid_propagate(0)
##        self.b14_frame.pack_propagate(0)
##        self.b14_frame.grid(column = 0, row = 14)
##        self.top_activate = Button(master = self.b14_frame, text = "TOP", relief="groove", font = self.bold_font, bd = 2, command = self.top_augment)
##        self.top_activate.pack(fill = BOTH, expand = True)

        self.b14_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b14_frame.grid_propagate(0)
        self.b14_frame.pack_propagate(0)
        self.b14_frame.grid(column = 0, row = 14)
        self.predict_debug_button = Button(master = self.b14_frame, text = "Debug Predict", relief="groove", font = self.bold_font, bd = 2, command = lambda : self.predict_image(debug=True))
        self.predict_debug_button.pack(fill = BOTH, expand = True)

        #FRAME3
        self.label(self.frame3, 600, 25, 0, 0, "MODEL SETTING", columnspan = 2)
        
        self.backbone_and_optimizer = Frame(master = self.frame3, width = 600, height = 50, bg = "orange")
        self.backbone_and_optimizer.grid_propagate(0)
        self.backbone_and_optimizer.pack_propagate(0)
        self.backbone_and_optimizer.grid(column = 0, row = 1, columnspan = 2)

        
        self.label(self.backbone_and_optimizer, 450, 25, 0, 0, "Backbone", columnspan = 1)
        self.backbone_frame = Frame(master = self.backbone_and_optimizer, width = 450, height = 25)
        self.backbone_frame.grid_propagate(0)
        self.backbone_frame.pack_propagate(0)
        self.backbone_frame.grid(column = 0, row = 1, columnspan = 1)
        self.backbone = StringVar(self.root)
        self.backbone.set("B0")

        self.backbone_radioMBN2 = Radiobutton(self.backbone_frame, text="MobileNetV2", variable = self.backbone, value = "MobileNetV2", font = self.small_font, command = self.update)
        self.backbone_radioMBN2.grid(column = 0, row = 0)

        self.backbone_radio1 = Radiobutton(self.backbone_frame, text="B0", variable = self.backbone, value = "B0", font = self.small_font, command = self.update)
        self.backbone_radio1.grid(column = 1, row = 0)
        
        self.backbone_radio2 = Radiobutton(self.backbone_frame, text="B1", variable = self.backbone, value = "B1", font = self.small_font, command = self.update)
        self.backbone_radio2.grid(column = 2, row = 0)

        self.backbone_radio3 = Radiobutton(self.backbone_frame, text="B3", variable = self.backbone, value = "B3", font = self.small_font, command = self.update)
        self.backbone_radio3.grid(column = 3, row = 0)

        #self.backbone_radio4 = Radiobutton(self.backbone_frame, text="B5", variable = self.backbone, value = "B5", font = self.small_font, command = self.update)
        #self.backbone_radio4.grid(column = 4, row = 0)

        #self.backbone_radio5 = Radiobutton(self.backbone_frame, text="B7", variable = self.backbone, value = "B7", font = self.small_font, command = self.update)
        #self.backbone_radio5.grid(column = 5, row = 0)

        self.backbone_radio6 = Radiobutton(self.backbone_frame, text="V2B0", variable = self.backbone, value = "V2B0", font = self.small_font, command = self.update)
        self.backbone_radio6.grid(column = 4, row = 0)
        
        self.backbone_radio7 = Radiobutton(self.backbone_frame, text="V2B1", variable = self.backbone, value = "V2B1", font = self.small_font, command = self.update)
        self.backbone_radio7.grid(column = 5, row = 0)
        
        #self.backbone_radio8 = Radiobutton(self.backbone_frame, text="V2B3", variable = self.backbone, value = "V2B3", font = self.small_font, command = self.update)
        #self.backbone_radio8.grid(column = 7, row = 0)
        
        self.backbone_radio9 = Radiobutton(self.backbone_frame, text="V2S", variable = self.backbone, value = "V2S", font = self.small_font, command = self.update)
        self.backbone_radio9.grid(column = 6, row = 0)


        self.label(self.backbone_and_optimizer, 150, 25, 0, 1, "Optimizer", columnspan = 1)
        self.optimizer_frame = Frame(master = self.backbone_and_optimizer, width = 150, height = 25)
        self.optimizer_frame.grid_propagate(0)
        self.optimizer_frame.pack_propagate(0)
        self.optimizer_frame.grid(column = 1, row = 1, columnspan = 1)
        self.optimizer = StringVar(self.root)
        self.optimizer.set("Adam")


        self.optimizer_adam = Radiobutton(self.optimizer_frame, text="Adam", variable = self.optimizer, value = "Adam", font = self.small_font, command = self.update)
        self.optimizer_adam.grid(column = 0, row = 0)

        self.optimizer_sgd = Radiobutton(self.optimizer_frame, text="SGD", variable = self.optimizer, value = "SGD", font = self.small_font, command = self.update)
        self.optimizer_sgd.grid(column = 1, row = 0)
        
        #-----------------------------------------------------------------------------

        self.input_and_channel = Frame(master = self.frame3, width = 600, height = 50, bg = "blue")
        self.input_and_channel.grid_propagate(0)
        self.input_and_channel.pack_propagate(0)
        self.input_and_channel.grid(column = 0, row = 2, columnspan = 2)
        
        
        self.label(self.input_and_channel, 400, 25, 0, 0, "Input size")
        self.input_size_frame = Frame(master = self.input_and_channel, width = 400, height = 25)
        self.input_size_frame.grid_propagate(0)
        self.input_size_frame.pack_propagate(0)
        self.input_size_frame.grid(column = 0, row = 1)
        self.input_size = StringVar(self.root)
        self.input_size.set("256")

        self.input_size_radioB = Radiobutton(self.input_size_frame, text="64", variable = self.input_size, value = 64, font = self.small_font, command = self.update)
        self.input_size_radioB.grid(column = 0, row = 0)
        self.input_size_radioA = Radiobutton(self.input_size_frame, text="96", variable = self.input_size, value = 96, font = self.small_font, command = self.update)
        self.input_size_radioA.grid(column = 1, row = 0)
        self.input_size_radio1 = Radiobutton(self.input_size_frame, text="128", variable = self.input_size, value = 128, font = self.small_font, command = self.update)
        self.input_size_radio1.grid(column = 2, row = 0)
        self.input_size_radio2 = Radiobutton(self.input_size_frame, text="224", variable = self.input_size, value = 224, font = self.small_font, command = self.update)
        self.input_size_radio2.grid(column = 3, row = 0)
        self.input_size_radio3 = Radiobutton(self.input_size_frame, text="256", variable = self.input_size, value = 256, font = self.small_font, command = self.update)
        self.input_size_radio3.grid(column = 4, row = 0)
        self.input_size_radio4 = Radiobutton(self.input_size_frame, text="320", variable = self.input_size, value = 320, font = self.small_font, command = self.update)
        self.input_size_radio4.grid(column = 5, row = 0)


        self.label(self.input_and_channel, 200, 25, 0, 1, "Color channel")
        self.color_channel_frame = Frame(master = self.input_and_channel, width = 200, height = 25)
        self.color_channel_frame.grid_propagate(0)
        self.color_channel_frame.pack_propagate(0)
        self.color_channel_frame.grid(column = 1, row = 1)
        self.color_channel = StringVar(self.root)
        self.color_channel.set("1")
        self.color_channel_radio1 = Radiobutton(self.color_channel_frame, text="Grayscale", variable = self.color_channel, value = 1, font = self.normal_font, command = self.update)
        self.color_channel_radio1.grid(column = 0, row = 0)
        self.color_channel_radio2 = Radiobutton(self.color_channel_frame, text="RGB", variable = self.color_channel, value = 3, font = self.normal_font, command = self.update)
        self.color_channel_radio2.grid(column = 1, row = 0, )


        #-------------------------------------------------------------------------------
        
        self.box_class_dropout = Frame(master = self.frame3, width = 600, height = 25, bg = "blue")
        self.box_class_dropout.grid_propagate(0)
        self.box_class_dropout.pack_propagate(0)
        self.box_class_dropout.grid(column = 0, row = 3, columnspan = 2)


        self.label_w_slider(self.box_class_dropout, 50, 120, 25, 0, 0, "Boxes", "region", font_size = self.normal_font, min_value =1, max_value = 5, interval = 1, bind=self.update)
        self.label_w_slider(self.box_class_dropout, 50, 170, 25, 0, 1, "Tags", "tags", font_size = self.normal_font, min_value =100, max_value = 1000, interval = 100, bind=self.update)
        self.label_w_slider(self.box_class_dropout, 80, 130, 25, 0, 3, "Dropout", "dropout", font_size = self.normal_font, min_value =0, max_value = 0.6, interval = 0.1, bind=self.update)
        

        self.label(self.frame3, 600, 25, 4, 0, "", columnspan = 2)
        self.label(self.frame3, 600, 25, 5, 0, "TRAINING PARAMETERS", columnspan = 2)

        self.vlabel_w_textarea(self.frame3, 600, 25, 125, 6, 0, "Input files : ", "input_files", columnspan = 2, font_size = self.normal_font, bind=self.update)

        self.label_w_slider(self.frame3, 100, 200, 25, 7, 0, "Anchor size", "anchor_size", font_size = self.normal_font, min_value =1, max_value = 10, interval = 1, bind=self.update)
        self.label_w_slider(self.frame3, 100, 200, 25, 7, 1, "Null ratio", "null_ratio", font_size = self.normal_font, min_value =0, max_value = 2, interval = 0.1, bind=self.update)

        self.label_w_slider(self.frame3, 100, 200, 25, 8, 0, "Batch size", "batch_size", font_size = self.normal_font, min_value =16, max_value = 128, interval = 16, bind=self.update)
        self.label_w_slider(self.frame3, 100, 200, 25, 8, 1, "Train Test", "train_test_ratio", font_size = self.normal_font, min_value =1, max_value = 0.5, interval = 0.1, bind=self.update)

        self.logslider(self.frame3, 100, 200, 25, 9, 0, "Learning rate", "learning_rate", font_size = self.normal_font, min_value=0.01, max_value=0.0001, bind=self.update)
        self.label_w_slider(self.frame3, 100, 200, 25, 9, 1, "training steps", "steps", font_size = self.normal_font, min_value =10, max_value = 100, interval = 10, bind=self.update)

        self.augment_button_frame = Frame(master = self.frame3, width = 300, height = 25)
        self.augment_button_frame.grid_propagate(0)
        self.augment_button_frame.pack_propagate(0)
        self.augment_button_frame.grid(column = 1, row = 10, columnspan = 1)
        self.top_augment_activate = Button(master = self.augment_button_frame, text = "Augment", relief="groove", font = self.normal_font, bd = 2, command = self.top_augment)
        self.top_augment_activate.pack(fill = BOTH, expand = True)
        augment = self.value_type(self.inputs["augment"], int)
        self.top_augment_activate.config(text = "Augment : %s" % augment)

        self.spacer0301 = Frame(master = self.frame3, width = 600, height = 25)
        self.spacer0301.grid_propagate(0)
        self.spacer0301.pack_propagate(0)
        self.spacer0301.grid(column = 0, row = 11, columnspan = 2)
        
        self.label_w_input(self.frame3, 100, 500, 25, 12, 0, "Test folder", "testfolder", columnspan = 2, bind=self.update)

# END OF BUILD ---------------------------------------------------------

    def load_pik(self):
        if os.path.isfile("main_config.pik"):
            with open("main_config.pik", "rb") as fio:
                self.data = pickle.load(fio)
                self.input_value_initialize()
                return True
        self.input_value_initialize()
        return False

    def input_value_initialize(self):
        print("INITIALIZING")
        if "savefile" in self.data:
            if self.data["savefile"]:
                self.inputs["savefile"].set(self.data["savefile"])

        if "port" in self.data:
            if self.data["port"]:
                self.inputs["port"].set(self.data["port"])
                    
        if "testfolder" in self.data:
            if self.data["testfolder"]:
                self.inputs["testfolder"].set(self.data["testfolder"])

        if "input_files" in self.data:
            if self.data["input_files"]:
                input_files = self.data["input_files"]
                self.objects["input_files"]["input"].delete("1.0", END)
                self.objects["input_files"]["input"].insert("1.0", "\n".join(input_files))

        if "learning_rate" in self.data:
            if self.data["learning_rate"]:
                self.inputs["learning_rate"].set(self.data["learning_rate"])
                self.inputs["learning_rate__TEMP__"].set(math.floor(math.log(self.data["learning_rate"], 10)))
            else:
                self.inputs["learning_rate"].set(0.01)
                self.data["learning_rate"] = 0.01
                self.inputs["learning_rate__TEMP__"].set(math.floor(math.log(0.01, 10)))
        else:
            self.inputs["learning_rate"].set(0.01)
            self.data["learning_rate"] = 0.01
            self.inputs["learning_rate__TEMP__"].set(math.floor(math.log(0.01, 10)))
            
                
        if "steps" in self.data:
            if self.data["steps"] or self.data["steps"] == 0:
                self.inputs["steps"].set(self.data["steps"])
            else:
                self.inputs["steps"].set(10)
                self.data["steps"] = 10
        else:
            self.inputs["steps"].set(10)
            self.data["steps"] = 10
                
        if "anchor_size" in self.data:
            if self.data["anchor_size"] or self.data["anchor_size"] == 0:
                self.inputs["anchor_size"].set(self.data["anchor_size"])
            else:
                self.inputs["anchor_size"].set(2)
                self.data["anchor_size"] = 2
        else:
            self.inputs["anchor_size"].set(2)
            self.data["anchor_size"] = 2

        if "null_ratio" in self.data:
            if self.data["null_ratio"] or self.data["null_ratio"] == 0:
                self.inputs["null_ratio"].set(self.data["null_ratio"])
            else:
                self.inputs["null_ratio"].set(1.0)
                self.data["null_ratio"] = 1.0
        else:
            self.inputs["null_ratio"].set(1.0)
            self.data["null_ratio"] = 1.0

        if "dropout" in self.data:
            if self.data["dropout"] or self.data["dropout"] == 0:
                self.inputs["dropout"].set(self.data["dropout"])
            else:
                self.inputs["dropout"].set(0.2)
                self.data["dropout"] = 0.2
        else:
            self.inputs["dropout"].set(0.2)
            self.data["dropout"] = 0.2

        if "batch_size" in self.data:
            if self.data["batch_size"]:
                self.inputs["batch_size"].set(self.data["batch_size"])
            else:
                self.inputs["batch_size"].set(16)
                self.data["batch_size"] = 16
        else:
            self.inputs["batch_size"].set(16)
            self.data["batch_size"] = 16
                    
        if "train_test_ratio" in self.data:
            if self.data["train_test_ratio"]:
                self.inputs["train_test_ratio"].set(self.data["train_test_ratio"])
            else:
                self.inputs["train_test_ratio"].set(0.7)
                self.data["train_test_ratio"] = 0.7
        else:
            self.inputs["train_test_ratio"].set(0.7)
            self.data["train_test_ratio"] = 0.7

        if "backbone" in self.data:
            if self.data["backbone"]:
                self.backbone.set(self.data["backbone"])
            else:
                self.backbone.set("B0")
                self.data["backbone"] = "B0"
        else:
            self.backbone.set("B0")
            self.data["backbone"] = "B0"
            
        if "input_shape" in self.data:
            if self.data["input_shape"]:
                self.input_size.set(self.data["input_shape"][0])
                self.color_channel.set(self.data["input_shape"][2])
            else:
                self.input_size.set(64)
                self.color_channel.set(3)
                self.data["input_shape"] = (64, 64, 3)         
        else:
            self.input_size.set(64)
            self.color_channel.set(3)
            self.data["input_shape"] = (64, 64, 3)

        if "optimizer" in self.data:
            if self.data["optimizer"]:        
                self.optimizer.set(self.data["optimizer"])
            else:
                self.optimizer.set("Adam")
        else:
            self.optimizer.set("Adam")

        if "tags" in self.data:
            if self.data["tags"]:
                self.inputs["tags"].set(self.data["tags"])
            else:
                self.inputs["tags"].set(100)
                self.data["tags"] = 100
        else:
            self.inputs["tags"].set(100)
            self.data["tags"] = 100
                    
        if "region" in self.data:
            if self.data["region"]:
                self.inputs["region"].set(self.data["region"])
            else:
                self.inputs["region"].set(2)
                self.data["region"] = 2
        else:
            self.inputs["region"].set(2)
            self.data["region"] = 2

        if "augment" in self.data:
            if self.data["augment"]:
                self.inputs["augment"].set(self.data["augment"])
            else:
                self.inputs["augment"].set(255)
                self.data["augment"] = 255
        else:
            self.inputs["augment"].set(255)
            self.data["augment"] = 255
        #self.update()
        
            
    def load_model(self):
        print("LOAD MODEL")
        self.update()
        MODEL_NAME = self.inputs["savefile"].get()
        IMAGE_SHAPE = self.data["input_shape"]
        REGIONS = self.data["region"]
        CLASSES = self.data["tags"]
        DROPOUT = self.data["dropout"]
        BACKBONE = self.data["backbone"]
        OPTIMIZER = self.data["optimizer"]
        self.loaded_settings = {}
        if MODEL_NAME and IMAGE_SHAPE and REGIONS and CLASSES and DROPOUT and BACKBONE and OPTIMIZER:
            print("LOADING HARRVOTT")
            self.DET_MODEL = HARR_VOTT.load_model(MODEL_NAME, IMAGE_SHAPE, REGIONS, CLASSES, DROPOUT, BACKBONE, OPTIMIZER)
            self.loaded_settings["savefile"] = MODEL_NAME
            self.loaded_settings["input_shape"] = IMAGE_SHAPE
            self.loaded_settings["region"] = REGIONS
            self.loaded_settings["tags"] = CLASSES
            self.loaded_settings["dropout"] = DROPOUT
            self.loaded_settings["backbone"] = BACKBONE
            self.loaded_settings["optimizer"] = OPTIMIZER

    def setting_check_for_model(self):
        self.update()
        self.current_setting = {}
        self.current_setting["savefile"] = self.inputs["savefile"].get()
        self.current_setting["input_shape"] = self.data["input_shape"]
        self.current_setting["region"] = self.data["region"]
        self.current_setting["tags"] = self.data["tags"]
        self.current_setting["dropout"] = self.data["dropout"]
        self.current_setting["backbone"] = self.data["backbone"]
        self.current_setting["optimizer"] = self.data["optimizer"]
        return self.loaded_settings == self.current_setting
        

    def value_type(self, el, t = str):
        s = el.get()
        if type(s) == t:
            return s
        else:
            try:
                el.set(t(s))
                return t(s)
            except ValueError as e:
                pass
        el.set("")
        return None


    def update(self, event = None):
        #print("update")
        augment = self.value_type(self.inputs["augment"], int)
        self.top_augment_activate.config(text = "Augment : %s" % augment)
        b = '0000000000' + bin(augment)[2:]
        self.augment_08_var.set(b[-1])
        self.augment_07_var.set(b[-2])
        self.augment_06_var.set(b[-3])
        self.augment_05_var.set(b[-4])
        self.augment_04_var.set(b[-5])
        self.augment_03_var.set(b[-6])
        self.augment_02_var.set(b[-7])
        self.augment_01_var.set(b[-8])

        input_files = [n for n in self.objects["input_files"]["input"].get("1.0",END).split("\n") if os.path.isfile(n)]
        self.data["input_files"] = input_files
        self.objects["input_files"]["input"].delete("1.0", END)
        self.objects["input_files"]["input"].insert("1.0", "\n".join(input_files))
        
        self.data["backbone"] = self.value_type(self.backbone, str)
        self.data["input_shape"] = (self.value_type(self.input_size, int), \
                                    self.value_type(self.input_size, int), \
                                    self.value_type(self.color_channel, int))
        
        self.data["region"] = self.value_type(self.inputs["region"], int)
        self.data["tags"] = self.value_type(self.inputs["tags"], int)

        self.data["anchor_size"] = self.value_type(self.inputs["anchor_size"], int)
        self.data["batch_size"] = self.value_type(self.inputs["batch_size"], int)
        self.data["savefile"] = self.value_type(self.inputs["savefile"], str)
        
        self.data["testfolder"] = self.value_type(self.inputs["testfolder"], str)
        self.data["optimizer"] = self.value_type(self.optimizer, str)


        self.data["train_test_ratio"] = self.value_type(self.inputs["train_test_ratio"], float)
        self.data["learning_rate"] = self.value_type(self.inputs["learning_rate"], float)
        self.data["steps"] = self.value_type(self.inputs["steps"], int)
        self.data["null_ratio"] = self.value_type(self.inputs["null_ratio"], float)

        self.data["port"] = self.value_type(self.inputs["port"], int)
        self.data["augment"] = self.value_type(self.inputs["augment"], int)

        with open("main_config.pik", "wb") as fio:
            pickle.dump(self.data, fio)
            
        if self.data["savefile"]:
            if len(self.data["savefile"]) > 3:
                with open(self.data["savefile"] + "_cfg.pik", "wb") as fio:
                    pickle.dump(self.data, fio)


    def segment_image(self):
        if not self.setting_check_for_model():
            self.load_model()
        if not self.DET_MODEL:
            print("MODEL NOT SET, LOADING MODEL")
            self.load_model()
        if not messagebox.askyesno(title="segment", message="create segmented imageset?"):
            return False
        self.update()
        self.DET_MODEL.load_input(self.data["input_files"])                  
        ANCHOR_SIZE = self.value_type(self.inputs["anchor_size"], int)
        self.DET_MODEL.segment_image(ANCHOR_SIZE)


    def train(self, early_stopping = False):
        self.update()
        if not self.setting_check_for_model():
            self.load_model()
            print("PLEASE LOAD WEIGHT")
            return False
        if not self.DET_MODEL:
            print("MODEL NOT SET, LOADING MODEL")
            self.load_model()
            return False
        LOAD_SEGMENT = False
        if os.path.isdir(self.data["savefile"]):
            if len(os.listdir(self.data["savefile"])) > 10:
                if messagebox.askyesno(title="segment", message="Run with segmented images?"):
                    LOAD_SEGMENT = True
        EPOCH = simpledialog.askstring(title="Train", prompt="How many EPOCH?:")
        if EPOCH:
            try:
                EPOCH = int(EPOCH)
            except Exception as e:
                print(e)
                messagebox.showwarning("Warning", "invalid EPOCH")
                return False
        else:
            return False
        print("LOADING IMAGE DATA")
        if LOAD_SEGMENT:
            self.DET_MODEL.load_input([self.data["savefile"]])
        self.DET_MODEL.load_input(self.data["input_files"])
        print("PREPARE TO TRAIN")
        ANCHOR_SIZE = self.value_type(self.inputs["anchor_size"], int)
        BATCH_SIZE = self.value_type(self.inputs["batch_size"], int)
        NULL_RATIO = self.value_type(self.inputs["null_ratio"], float)
        TRAIN_TEST_RATIO = self.value_type(self.inputs["train_test_ratio"], float)
        LEARNING_RATE = self.value_type(self.inputs["learning_rate"], float)
        STEPS = self.value_type(self.inputs["steps"], int)
        AUGMENT = self.value_type(self.inputs["augment"], int)
        print("TRAINING : ", EPOCH, "Learning rate : ", LEARNING_RATE)
        self.DET_MODEL.train(LEARNING_RATE, \
                             EPOCH, \
                             STEPS, \
                             TRAIN_TEST_RATIO, \
                             BATCH_SIZE, \
                             ANCHOR_SIZE, \
                             NULL_RATIO, \
                             AUGMENT, \
                             callback_earlystop = early_stopping)
        self.DET_MODEL.save()
        self.DET_MODEL.chart()
        self.DET_MODEL.save_config()

        
    def save(self):
        if not self.DET_MODEL:
            print("MODEL NOT SET")
            return False
        print("widget.save")
        self.DET_MODEL.save()
        

    def load(self):
        print("widget.load")
        if not self.setting_check_for_model():
            self.load_model()
            return False
        if not self.DET_MODEL:
            print("MODEL NOT SET, LOADING A MODEL")
            self.load_model()
            return False
        self.DET_MODEL.load()


    def trial(self):
        if not self.setting_check_for_model():
            self.load_model()
            print("PLEASE LOAD WEIGHT")
            return False
        if not self.DET_MODEL:
            print("MODEL NOT SET, LOADING A MODEL")
            self.load_model()
            return False
        ANCHOR_SIZE = self.value_type(self.inputs["anchor_size"], int)
        BATCH_SIZE = self.value_type(self.inputs["batch_size"], int)
        NULL_RATIO = self.value_type(self.inputs["null_ratio"], float)
        TRAIN_TEST_RATIO = self.value_type(self.inputs["train_test_ratio"], float)
        LEARNING_RATE = self.value_type(self.inputs["learning_rate"], float)
        STEPS = self.value_type(self.inputs["steps"], int)
        AUGMENT = self.value_type(self.inputs["augment"], int)
        TESTFOLDER = self.value_type(self.inputs["testfolder"], str)
        self.DET_MODEL.trial(TESTFOLDER, \
                             ANCHOR_SIZE, \
                             nms_iou = 0.01, \
                             segment_minimum_ratio = 0.75)
        print("TRIAL END")

        
    def sanity_check(self):
        self.update()
        if not self.setting_check_for_model():
            self.load_model()
        if not self.DET_MODEL:
            print("MODEL NOT SET, LOADING MODEL")
            self.load_model()
        if not np.any(self.DET_MODEL.model.c.df):
            LOAD_SEGMENT = False
            if os.path.isdir(self.data["savefile"]):
                if len(os.listdir(self.data["savefile"])) > 10:
                    if messagebox.askyesno(title="segment", message="Run with segmented images?"):
                        LOAD_SEGMENT = True
            if LOAD_SEGMENT:                
                self.DET_MODEL.load_input([self.data["savefile"]])
            self.DET_MODEL.load_input(self.data["input_files"])
        ANCHOR_SIZE = self.value_type(self.inputs["anchor_size"], int)
        BATCH_SIZE = self.value_type(self.inputs["batch_size"], int)
        NULL_RATIO = self.value_type(self.inputs["null_ratio"], float)
        TRAIN_TEST_RATIO = self.value_type(self.inputs["train_test_ratio"], float)
        LEARNING_RATE = self.value_type(self.inputs["learning_rate"], float)
        STEPS = self.value_type(self.inputs["steps"], int)
        AUGMENT = self.value_type(self.inputs["augment"], int)
        self.DET_MODEL.sanity_check(BATCH_SIZE, \
                                    NULL_RATIO, \
                                    ANCHOR_SIZE, \
                                    AUGMENT)


    def predict_image(self, debug = False):
        if not self.setting_check_for_model():
            self.load_model()
            print("PLEASE LOAD WEIGHT")
            return False
        if not self.DET_MODEL:
            print("MODEL NOT SET")
            return False
        filename = filedialog.askopenfilename(initialdir = self.initialdir, title = "Select a File", \
                                              filetypes = (("image files", ("*.bmp*", "*.jpg*", "*.png*")), \
                                                           ("all files", "*.*")))
        INPUT_SHAPE = (self.value_type(self.input_size, int), \
                                    self.value_type(self.input_size, int), \
                                    self.value_type(self.color_channel, int))
        ANCHOR_SIZE = self.value_type(self.inputs["anchor_size"], int)
        BATCH_SIZE = self.value_type(self.inputs["batch_size"], int)
        NULL_RATIO = self.value_type(self.inputs["null_ratio"], float)
        TRAIN_TEST_RATIO = self.value_type(self.inputs["train_test_ratio"], float)
        LEARNING_RATE = self.value_type(self.inputs["learning_rate"], float)
        STEPS = self.value_type(self.inputs["steps"], int)
        AUGMENT = self.value_type(self.inputs["augment"], int)
        if os.path.isfile(filename):
            self.DET_MODEL.predict(filename, \
                                   True, \
                                   INPUT_SHAPE, \
                                   ANCHOR_SIZE, \
                                   nms_iou = 0.01, \
                                   segment_minimum_ratio = 0.75, \
                                   output_size = None, \
                                   debug = debug)
            self.initialdir = os.path.split(filename)[0]
        
        
    def top_augment(self):
        self.update()        
        self.toplevel_top = Toplevel(self.root)
        self.toplevel_a = Frame(master = self.toplevel_top, width = 480, height = 240)
        self.toplevel_a.grid_propagate(0)
        self.toplevel_a.pack_propagate(0)
        self.toplevel_a.grid(column = 0, row = 0)
        self.toplevel_b = Frame(master = self.toplevel_top, bg = "red", width = 480, height = 25)
        self.toplevel_b.grid_propagate(0)
        self.toplevel_b.pack_propagate(0)
        self.toplevel_b.grid(column = 0, row = 1)
        
        self.augment_01 = Checkbutton(self.toplevel_a, text='AverageBlur', variable = self.augment_01_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_01.grid(column = 0, row = 0, sticky = W)
        self.augment_02 = Checkbutton(self.toplevel_a, text='AdditiveGaussianNoise', variable = self.augment_02_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_02.grid(column = 0, row = 1, sticky = W)
        self.augment_03 = Checkbutton(self.toplevel_a, text='Affine.translate_percent', variable = self.augment_03_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_03.grid(column = 0, row = 2, sticky = W)
        self.augment_04 = Checkbutton(self.toplevel_a, text='Affine.scale', variable = self.augment_04_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_04.grid(column = 0, row = 3, sticky = W)
        self.augment_05 = Checkbutton(self.toplevel_a, text='Affine.rotate', variable = self.augment_05_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_05.grid(column = 0, row = 4, sticky = W)
        self.augment_06 = Checkbutton(self.toplevel_a, text='Affine.shear', variable = self.augment_06_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_06.grid(column = 0, row = 5, sticky = W)
        self.augment_07 = Checkbutton(self.toplevel_a, text='Fliplr', variable = self.augment_07_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_07.grid(column = 0, row = 6, sticky = W)
        self.augment_08 = Checkbutton(self.toplevel_a, text='Flipud', variable = self.augment_08_var, onvalue="1", offvalue="0", font = self.small_font, command = self.augment_change)
        self.augment_08.grid(column = 0, row = 7, sticky = W)
        
    def augment_change(self):
        b = '0000000000'
        b += self.augment_01_var.get()
        b += self.augment_02_var.get()
        b += self.augment_03_var.get()
        b += self.augment_04_var.get()
        b += self.augment_05_var.get()
        b += self.augment_06_var.get()
        b += self.augment_07_var.get()
        b += self.augment_08_var.get()
        augment = int(b, 2)
        self.inputs["augment"].set(augment)
        self.top_augment_activate.config(text = "Augment : %s" % augment)
        self.update()
        
    def top_assign_apply_button_press(self):
        self.update()
        self.top_augment.destroy()


    def video_capture(self):
        if not np.any(self.DET_MODEL.model.c.df_label):
            print("PLEASE LOAD WEIGHTS")
            return False
        self.STRINGVAR = StringVar(self.root)
        self.THREAD = None
        self.update()        
        self.vid_top = Toplevel(self.root)
        self.vid_top_a = Frame(master = self.vid_top, bg = "blue", width = 480, height = 480)
        self.vid_top_a.grid(column = 0, row = 0)

        self.console_image_canvas = Canvas(master = self.vid_top_a, bg = "black", width = 480, height = 480)
        self.console_image_canvas.grid(column = 0, row = 0)
        self.console_image_canvas.bind('<Button>', self.canvas_click)

        
        self.vid_top_b = Frame(master = self.vid_top, bg = "red", width = 480, height = 25)
        self.vid_top_b.grid(column = 0, row = 1)

        self.camera = camera.camera()
        self.camera_radio = {}
        self.camera_select_value = IntVar(self.root)
        
        for i, camera_index in enumerate(self.camera.device_array):
            self.camera_radio[camera_index] = Radiobutton(self.vid_top_b, text="dv%01d" % camera_index, variable = self.camera_select_value, value = camera_index, font = self.normal_font, command = self.switch_camera)
            self.camera_radio[camera_index].grid(column = i, row = 0)

    def switch_camera(self, event = None):
        self.camera.close()
        self.camera.connect(self.camera_select_value.get())
        self.loop()

    def canvas_click(self, event):
        print(event)

    def update_canvas(self):
        if type(self.camera.device_index) != int:
            return None
        ret, im = self.camera.read()
        if not ret:
            return None
        if np.any(im):
            if im.ndim == 3:
                shape = im.shape
                if int(self.console_image_canvas.config()["width"][-1]) != shape[1]:
                    self.console_image_canvas.config(width = shape[1])
                if int(self.console_image_canvas.config()["height"][-1]) != shape[0]:
                    self.console_image_canvas.config(height = shape[0])
        b,g,r = cv2.split(im)
        img = cv2.merge((r,g,b))
        val = self.STRINGVAR.get()
        if val:
            rawdata = json.loads(val)
            im = self.image_label(im, rawdata, lw = 2)
        img = im.astype("uint8")
        img = Image.fromarray(img)
        self.img = ImageTk.PhotoImage(image = img)
        self.console_image_canvas.delete("all")
        self.console_image_canvas.create_image(0, 0, image=self.img, anchor=NW)


    def image_label(self, image, rawdata, lw = 2):
        height, width, chan = image.shape
        for data in rawdata:
            if data:
                if "color" in data:
                    color = HARR_VOTT.from_hex(data["color"])
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
    
    def loop(self):
        try:
            self.root.after_cancel(self.l)
        except Exception as e:
            #print("ERROR widget.loop", e)
            pass
        if self.vid_top.winfo_exists():
            self.cycle()
            self.camera_detect()
            self.update_canvas()
            self.l = self.root.after(CYCLETIME , self.loop)
        else:
            self.camera.close()
            
    def cycle(self):
        if self.THREAD:
            if self.THREAD.is_alive():
                pass
            else:
                v = self.THREAD.join()
                self.THREAD = None
                
                
    def worker(self, func, args):
        if not self.THREAD:
            return ThreadWithReturnValue(target = func, args = args)
        else:
            return None

    def camera_detect(self, event = None):
        if self.THREAD:
            if self.THREAD.is_alive():
                return False
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

        self.THREAD = self.worker(self.camera_predict, (im, ))
        if self.THREAD:
            self.THREAD.start()

    def camera_predict(self, args):
        if np.any(args):
            if args.ndim == 3:
                if args.shape[0] > 100 and args.shape[1] > 100 and args.shape[2] in (1, 3): 
                    x, y = self.DET_MODEL.predict(args, False)
                    self.STRINGVAR.set(json.dumps(eval(str(y))))
                    print("DONE DETECT")
        
        
    
def run():
    w = widget()
    w.build()
    w.load_pik()
    w.load_model()
    #w.load()
    #w.video_capture()
    w.root.mainloop()
    return w


if __name__ == "__main__":
    runwidget = True
    if len(sys.argv) == 2:
        if sys.argv[1] == "-serve":
            tk_socket_server.run()
            runwidget = False
    if runwidget:
        system("title HARRVOTT widget")
        w = run()


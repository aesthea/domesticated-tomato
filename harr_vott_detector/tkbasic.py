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
#from functools import partial
import HARR_VOTT
import tk_socket_server

#import asyncio
#import websockets
#import multiprocessing

monitor_info = GetMonitorInfo(MonitorFromPoint((0,0)))
monitor_area = monitor_info.get("Monitor")
work_area = monitor_info.get("Work")
    
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
        self.root.title("HARR VOTT 1.2024d")
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
        self.RELOAD_AI_FLAG = False

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
        self.label(self.frame1, 600, 25, 0, 0, "Details")
        self.label_w_input(self.frame1, 200, 400, 25, 1, 0, "AI SAVEFILE", "savefile", bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame1, 200, 400, 25, 2, 0, "SERVE PORT", "port", bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)

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
        self.serve_button = Button(master = self.b5_frame, text = "SERVE", relief="groove", font = self.bold_font, bd = 2, command = self.serve_ai_button)
        self.serve_button.pack(fill = BOTH, expand = True)

        self.b6_frame = Frame(master = self.frame2, width = 200, height = 50)
        self.b6_frame.grid_propagate(0)
        self.b6_frame.pack_propagate(0)
        self.b6_frame.grid(column = 0, row = 6)
        
        self.b7_frame = Frame(master = self.frame2, width = 200, height = 25)
        self.b7_frame.grid_propagate(0)
        self.b7_frame.pack_propagate(0)
        self.b7_frame.grid(column = 0, row = 7)
        self.generator_check_button = Button(master = self.b7_frame, text = "Sanity check", relief="groove", font = self.bold_font, bd = 2, command = self.generator_check)
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
        self.train_button_cpu = Button(master = self.b11_frame, text = "TRAIN (early stopping)", relief="groove", font = self.bold_font, bd = 2, command = lambda : self.train(early_stopping = True, cpu_training = False))
        self.train_button_cpu.pack(fill = BOTH, expand = True)

        #FRAME3
        self.label(self.frame3, 600, 25, 0, 0, "ML parameter (Important!!)", columnspan = 2)
        self.label(self.frame3, 300, 25, 2, 0, "Backbone")
        self.backbone_frame = Frame(master = self.frame3, width = 300, height = 25)
        self.backbone_frame.grid_propagate(0)
        self.backbone_frame.pack_propagate(0)
        self.backbone_frame.grid(column = 0, row = 3)
        self.backbone = StringVar(self.root)
        self.backbone.set("B0")
#        self.backbone_radioA = Radiobutton(self.backbone_frame, text="S1", variable = self.backbone, value = "S1", font = self.normal_font, command = self.update)
#        self.backbone_radioA.grid(column = 0, row = 0)
#        self.backbone_radioB = Radiobutton(self.backbone_frame, text="S0", variable = self.backbone, value = "S0", font = self.normal_font, command = self.update)
#        self.backbone_radioB.grid(column = 1, row = 0)
        self.backbone_radio1 = Radiobutton(self.backbone_frame, text="B0", variable = self.backbone, value = "B0", font = self.small_font, command = self.update)
        self.backbone_radio1.grid(column = 0, row = 0)
        self.backbone_radio2 = Radiobutton(self.backbone_frame, text="B1", variable = self.backbone, value = "B1", font = self.small_font, command = self.update)
        self.backbone_radio2.grid(column = 1, row = 0)
##        self.backbone_radio3 = Radiobutton(self.backbone_frame, text="B2", variable = self.backbone, value = "B2", font = self.normal_font, command = self.update)
##        self.backbone_radio3.grid(column = 4, row = 0)
        self.backbone_radio4 = Radiobutton(self.backbone_frame, text="B3", variable = self.backbone, value = "B3", font = self.small_font, command = self.update)
        self.backbone_radio4.grid(column = 2, row = 0)
##        self.backbone_radio5 = Radiobutton(self.backbone_frame, text="B4", variable = self.backbone, value = "B4", font = self.normal_font, command = self.update)
##        self.backbone_radio5.grid(column = 6, row = 0)
#        self.backbone_radio6 = Radiobutton(self.backbone_frame, text="B5", variable = self.backbone, value = "B5", font = self.normal_font, command = self.update)
#        self.backbone_radio6.grid(column = 7, row = 0)
        self.backbone_radio7 = Radiobutton(self.backbone_frame, text="V2S", variable = self.backbone, value = "V2S", font = self.small_font, command = self.update)
        self.backbone_radio7.grid(column = 3, row = 0)
        self.backbone_radio8 = Radiobutton(self.backbone_frame, text="V2B0", variable = self.backbone, value = "V2B0", font = self.small_font, command = self.update)
        self.backbone_radio8.grid(column = 4, row = 0)
#        self.backbone_radio9 = Radiobutton(self.backbone_frame, text="V2B1", variable = self.backbone, value = "V2B1", font = self.small_font, command = self.update)
#        self.backbone_radio9.grid(column = 5, row = 0)
        
        self.label(self.frame3, 300, 25, 2, 1, "FPN mode")
        self.fpn_frame = Frame(master = self.frame3, width = 300, height = 25)
        self.fpn_frame.grid_propagate(0)
        self.fpn_frame.pack_propagate(0)
        self.fpn_frame.grid(column = 1, row = 3)
        self.fpn_mode = StringVar(self.root)
        self.fpn_mode.set("0")
        self.fpn_radio1 = Radiobutton(self.fpn_frame, text="FPN", variable = self.fpn_mode, value = 0, font = self.small_font, command = self.update)
        self.fpn_radio1.grid(column = 0, row = 0)
        self.fpn_radio2 = Radiobutton(self.fpn_frame, text="Bi-FPN", variable = self.fpn_mode, value = 2, font = self.small_font, command = self.update)
        self.fpn_radio2.grid(column = 1, row = 0)
        
        self.lstm = BooleanVar(self.root)
        self.lstm.set(False)
        #self.lstm_checkbutton = Checkbutton(self.fpn_frame, text='LSTM',variable=self.lstm, onvalue=True, offvalue=False, font = self.normal_font, command = self.update)
        #self.lstm_checkbutton.grid(column = 2, row = 0)

        self.normalization = BooleanVar(self.root)
        self.normalization.set(False)
        self.normalization_checkbutton = Checkbutton(self.fpn_frame, text='NORMALIZE',variable=self.normalization, onvalue=True, offvalue=False, font = self.small_font, command = self.update)
        self.normalization_checkbutton.grid(column = 2, row = 0)
        
        self.label(self.frame3, 300, 25, 4, 0, "Input size")
        self.input_size_frame = Frame(master = self.frame3, width = 300, height = 25)
        self.input_size_frame.grid_propagate(0)
        self.input_size_frame.pack_propagate(0)
        self.input_size_frame.grid(column = 0, row = 5)
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

        self.label(self.frame3, 300, 25, 4, 1, "Color channel")
        self.color_channel_frame = Frame(master = self.frame3, width = 300, height = 25)
        self.color_channel_frame.grid_propagate(0)
        self.color_channel_frame.pack_propagate(0)
        self.color_channel_frame.grid(column = 1, row = 5)
        self.color_channel = StringVar(self.root)
        self.color_channel.set("1")
        self.color_channel_radio1 = Radiobutton(self.color_channel_frame, text="Grayscale", variable = self.color_channel, value = 1, font = self.normal_font, command = self.update)
        self.color_channel_radio1.grid(column = 0, row = 0)
        self.color_channel_radio2 = Radiobutton(self.color_channel_frame, text="RGB", variable = self.color_channel, value = 3, font = self.normal_font, command = self.update)
        self.color_channel_radio2.grid(column = 1, row = 0, )        

        self.label_w_input(self.frame3, 100, 200, 25, 6, 0, "Boxes", "region", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 6, 1, "Tags", "tags", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)

        self.label(self.frame3, 600, 25, 7, 0, "", columnspan = 2)
        self.label(self.frame3, 600, 25, 8, 0, "Training parameter", columnspan = 2)

        self.vlabel_w_textarea(self.frame3, 600, 25, 100, 9, 0, "VOTT Export Files : vott-json-export >> -export.json", "votts", columnspan = 2, font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 500, 25, 10, 0, "Test folder", "testfolder", columnspan = 2, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)

        self.label_w_input(self.frame3, 100, 200, 25, 11, 0, "Anchor level", "anchor", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 11, 1, "Skip Null", "nullskip", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)

        self.label_w_input(self.frame3, 100, 200, 25, 12, 0, "Batch size", "batchsize", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 12, 1, "TrainTest split", "trainsize", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)

        self.label_w_input(self.frame3, 100, 200, 25, 13, 0, "Huber mod", "huber", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 13, 1, "Learning rate", "learning_rate", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        
        self.label_w_input(self.frame3, 100, 200, 25, 14, 0, "Train steps", "steps", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 14, 1, "Dropout", "dropout", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)

        self.label_w_input(self.frame3, 100, 200, 25, 15, 0, "Overlap", "overlap", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 15, 1, "Image Skip", "random_drop", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        
        self.label_w_input(self.frame3, 100, 200, 25, 16, 0, "NMS_IoU", "non_max_suppression_iou", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        self.label_w_input(self.frame3, 100, 200, 25, 16, 1, "Augment", "augment", font_size = self.normal_font, bind_focusout=self.update, bind_return=self.update, bind_tab=self.update)
        


    def load_pik(self):
        if os.path.isfile("tkpik.pik"):
            with open("tkpik.pik", "rb") as fio:
                self.data = pickle.load(fio)
                self.input_value_initialize()
                return True
        self.input_value_initialize()
        return False

    def input_value_initialize(self):
        if "savefile" in self.data:
            if self.data["savefile"]:
                self.inputs["savefile"].set(self.data["savefile"])

        if "port" in self.data:
            if self.data["port"]:
                self.inputs["port"].set(self.data["port"])
                    
        if "testfolder" in self.data:
            if self.data["testfolder"]:
                self.inputs["testfolder"].set(self.data["testfolder"])

        if "votts" in self.data:
            if self.data["votts"]:
                votts = self.data["votts"]
                self.objects["votts"]["input"].delete("1.0", END)
                self.objects["votts"]["input"].insert("1.0", "\n".join(votts))

        if "learning_rate" in self.data:
            if self.data["learning_rate"]:
                self.inputs["learning_rate"].set(self.data["learning_rate"])

        if "steps" in self.data:
            if self.data["steps"]:
                self.inputs["steps"].set(self.data["steps"])
            
        if "anchor" in self.data:
            if self.data["anchor"] or self.data["anchor"] == 0:
                self.inputs["anchor"].set(self.data["anchor"])
            else:
                self.inputs["anchor"].set(2)
                self.data["anchor"] = 2
        else:
            self.inputs["anchor"].set(2)
            self.data["anchor"] = 2

        if "nullskip" in self.data:
            if self.data["nullskip"] or self.data["nullskip"] == 0:
                self.inputs["nullskip"].set(self.data["nullskip"])
            else:
                self.inputs["nullskip"].set(0.3)
                self.data["nullskip"] = 0.3
        else:
            self.inputs["nullskip"].set(0.3)
            self.data["nullskip"] = 0.3

        if "dropout" in self.data:
            if self.data["dropout"] or self.data["dropout"] == 0:
                self.inputs["dropout"].set(self.data["dropout"])
            else:
                self.inputs["dropout"].set(0.2)
                self.data["dropout"] = 0.2
        else:
            self.inputs["dropout"].set(0.2)
            self.data["dropout"] = 0.2

        if "batchsize" in self.data:
            if self.data["batchsize"]:
                self.inputs["batchsize"].set(self.data["batchsize"])
            else:
                self.inputs["batchsize"].set(16)
                self.data["batchsize"] = 16
        else:
            self.inputs["batchsize"].set(16)
            self.data["batchsize"] = 16
                    
        if "trainsize" in self.data:
            if self.data["trainsize"]:
                self.inputs["trainsize"].set(self.data["trainsize"])
            else:
                self.inputs["trainsize"].set(0.7)
                self.data["trainsize"] = 0.7
        else:
            self.inputs["trainsize"].set(0.7)
            self.data["trainsize"] = 0.7

        if "huber" in self.data:
            if self.data["huber"] or self.data["huber"] == 0:
                self.inputs["huber"].set(self.data["huber"])
            else:
                self.inputs["huber"].set(20)
                self.data["huber"] = 20
        else:
            self.inputs["huber"].set(20)
            self.data["huber"] = 20

        if "backbone" in self.data:
            if self.data["backbone"]:
                self.backbone.set(self.data["backbone"])
            else:
                self.backbone.set("B1")
                self.data["backbone"] = "B1"
        else:
            self.backbone.set("B1")
            self.data["backbone"] = "B1"
                

        if "fpn_mode" in self.data:
            if self.data["fpn_mode"] in (0, 2):
                self.fpn_mode.set(self.data["fpn_mode"])
            else:
                self.fpn_mode.set(2)
                self.data["fpn_mode"] = 2
        else:
            self.fpn_mode.set(2)
            self.data["fpn_mode"] = 2
            
        if "input_size" in self.data:
            if self.data["input_size"]:
                self.input_size.set(self.data["input_size"])
            else:
                self.input_size.set(256)
                self.data["input_size"] = 256
        else:
            self.input_size.set(256)
            self.data["input_size"] = 256

        if "tags" in self.data:
            if self.data["tags"]:
                self.inputs["tags"].set(self.data["tags"])
            else:
                self.inputs["tags"].set(10)
                self.data["tags"] = 10
        else:
            self.inputs["tags"].set(10)
            self.data["tags"] = 10
                    
        if "region" in self.data:
            if self.data["region"]:
                self.inputs["region"].set(self.data["region"])
            else:
                self.inputs["region"].set(2)
                self.data["region"] = 2
        else:
            self.inputs["region"].set(2)
            self.data["region"] = 2

        if "color_channel" in self.data:
            if self.data["color_channel"]:
                self.color_channel.set(self.data["color_channel"])
            else:
                self.color_channel.set(3)
                self.data["color_channel"] = 3
        else:
            self.color_channel.set(3)
            self.data["color_channel"] = 3

        if "overlap" in self.data:
            if self.data["overlap"]:
                self.inputs["overlap"].set(self.data["overlap"])
            else:
                self.inputs["overlap"].set(0.9)
                self.data["overlap"] = 0.9
        else:
            self.inputs["overlap"].set(0.9)
            self.data["overlap"] = 0.9

        if "augment" in self.data:
            if self.data["augment"]:
                self.inputs["augment"].set(self.data["augment"])
            else:
                self.inputs["augment"].set(255)
                self.data["augment"] = 255
        else:
            self.inputs["augment"].set(255)
            self.data["augment"] = 255
            
        if "non_max_suppression_iou" in self.data:
            if self.data["non_max_suppression_iou"]:
                self.inputs["non_max_suppression_iou"].set(self.data["non_max_suppression_iou"])
            else:
                self.inputs["non_max_suppression_iou"].set(0.01)
                self.data["non_max_suppression_iou"] = 0.01
        else:
            self.inputs["non_max_suppression_iou"].set(0.01)
            self.data["non_max_suppression_iou"] = 0.01

        if "lstm" in self.data:
            if self.data["lstm"] != None:
                self.lstm.set(self.data["lstm"])
            else:
                self.lstm.set(False)
                self.data["lstm"] = False
        else:
            self.lstm.set(False)
            self.data["lstm"] = False

        if "normalization" in self.data:
            if self.data["normalization"] != None:
                self.normalization.set(self.data["normalization"])
            else:
                self.normalization.set(False)
                self.data["normalization"] = False
        else:
            self.normalization.set(False)
            self.data["normalization"] = False

        if "random_drop" in self.data:
            if self.data["random_drop"] or self.data["random_drop"] == 0:
                self.inputs["random_drop"].set(self.data["random_drop"])
            else:
                self.inputs["random_drop"].set(0.2)
                self.data["random_drop"] = 0.2
        else:
            self.inputs["random_drop"].set(0.2)
            self.data["random_drop"] = 0.2

        if "steps" in self.data:
            if self.data["steps"] or self.data["steps"] == 0:
                self.inputs["steps"].set(self.data["steps"])
            else:
                self.inputs["steps"].set(20)
                self.data["steps"] = 20
        else:
            self.inputs["steps"].set(20)
            self.data["steps"] = 20

        if "learning_rate" in self.data:
            if self.data["learning_rate"]:
                self.inputs["learning_rate"].set(self.data["learning_rate"])
            else:
                self.inputs["learning_rate"].set(0.001)
                self.data["learning_rate"] = 0.001
        else:
            self.inputs["learning_rate"].set(0.001)
            self.data["learning_rate"] = 0.001
        
        self.update()
        
            
    def load_AI(self):
        self.load_pik()
        pts = 0
        if "votts" in self.data:
            if type(self.data["votts"]) == list:
                if len(self.data["votts"]) > 0:
                    pass
        for k in ("region", "tags", "anchor", "nullskip", \
                  "batchsize", "trainsize", "huber", "backbone", "fpn_mode", \
                  "input_size", "color_channel", "dropout", "overlap", "savefile", "augment", "non_max_suppression_iou", "lstm", \
                  "normalization", "random_drop"):
            if k in self.data:
                if self.data[k]:
                    pts += 1
                elif self.data[k] == 0:
                    pts += 1
                else:
                    print(k, "not set")
            else:
                print(k, "not set")
        if pts == 19:
            self.DET_MODEL = HARR_VOTT.load_model(self.data["input_size"], self.data["color_channel"], self.data["tags"], self.data["region"], \
                                                 self.data["dropout"], self.data["fpn_mode"], self.data["backbone"], self.data["votts"], self.data["augment"])
            self.DET_MODEL.BATCH_SIZE = self.data["batchsize"]
            self.DET_MODEL.HUBER = self.data["huber"]
            self.DET_MODEL.TRAIN_SIZE = self.data["trainsize"]
            self.DET_MODEL.ANCHOR_LEVEL = self.data["anchor"]
            self.DET_MODEL.NULL_SKIP = self.data["nullskip"]
            self.DET_MODEL.OVERLAP_REQUIREMENT = self.data["overlap"]
            self.DET_MODEL.SAVENAME = self.data["savefile"]
            self.DET_MODEL.NON_MAX_SUPPRESSION_IOU = self.data["non_max_suppression_iou"]
            self.DET_MODEL.NORMALIZATION = self.data["normalization"]
            self.DET_MODEL.RANDOM_DROP = self.data["random_drop"]
            self.DET_MODEL.initialize()
            
        else:
            self.DET_MODEL = None

    
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
        reload_AI = False
        
        votts = [n for n in self.objects["votts"]["input"].get("1.0",END).split("\n") if os.path.isfile(n)]
        if "votts" in self.data:
            if self.data["votts"] != votts:
                reload_AI = True
        if self.DET_MODEL:
            if self.DET_MODEL.VOTT_PATHS != votts:
                reload_AI = True
        self.data["votts"] = votts
        
        self.objects["votts"]["input"].delete("1.0", END)
        self.objects["votts"]["input"].insert("1.0", "\n".join(votts))

        if self.data["backbone"] != self.value_type(self.backbone, str):
            reload_AI = True                
        elif self.data["fpn_mode"] != self.value_type(self.fpn_mode, int):
            reload_AI = True
        elif self.data["input_size"] != self.value_type(self.input_size, int):
            reload_AI = True
        elif self.data["color_channel"] != self.value_type(self.color_channel, int):
            reload_AI = True
        elif self.data["region"] != self.value_type(self.inputs["region"], int):
            reload_AI = True
        elif self.data["tags"] != self.value_type(self.inputs["tags"], int):
            reload_AI = True
        elif self.data["dropout"] != self.value_type(self.inputs["dropout"], float):
            reload_AI = True
        elif self.data["anchor"] != self.value_type(self.inputs["anchor"], int):
            reload_AI = True
        elif self.data["batchsize"] != self.value_type(self.inputs["batchsize"], int):
            reload_AI = True
        elif self.data["augment"] != self.value_type(self.inputs["augment"], int):
            reload_AI = True
        elif self.data["non_max_suppression_iou"] != self.value_type(self.inputs["non_max_suppression_iou"], float):
            reload_AI = True
        elif self.data["lstm"] != self.value_type(self.lstm, bool):
            reload_AI = True
        elif self.data["normalization"] != self.value_type(self.normalization, bool):
            reload_AI = True
        elif self.data["random_drop"] != self.value_type(self.inputs["random_drop"], float):
            reload_AI = True

        if self.DET_MODEL:
            if self.DET_MODEL.BACKBONE != self.value_type(self.backbone, str):
                reload_AI = True
            elif self.DET_MODEL.FPN_MODE != self.value_type(self.fpn_mode, int):
                reload_AI = True
            elif self.DET_MODEL.IMAGE_SIZE != self.value_type(self.input_size, int):
                reload_AI = True
            elif self.DET_MODEL.COLOR_CHANNEL != self.value_type(self.color_channel, int):
                reload_AI = True
            elif self.DET_MODEL.REGIONS != self.value_type(self.inputs["region"], int):
                reload_AI = True
            elif self.DET_MODEL.CLASSIFICATION_TAGS != self.value_type(self.inputs["tags"], int):
                reload_AI = True
            elif self.DET_MODEL.DROPOUT != self.value_type(self.inputs["dropout"], float):
                reload_AI = True
            elif self.DET_MODEL.ANCHOR_LEVEL != self.value_type(self.inputs["anchor"], int):
                reload_AI = True
            elif self.DET_MODEL.BATCH_SIZE != self.value_type(self.inputs["batchsize"], int):
                reload_AI = True
            elif self.DET_MODEL.AUGMENT != self.value_type(self.inputs["augment"], int):
                reload_AI = True
            elif self.DET_MODEL.NON_MAX_SUPPRESSION_IOU != self.value_type(self.inputs["non_max_suppression_iou"], float):
                reload_AI = True
            elif self.DET_MODEL.NORMALIZATION != self.value_type(self.normalization, bool):
                reload_AI = True
            elif self.DET_MODEL.RANDOM_DROP != self.value_type(self.inputs["random_drop"], float):
                reload_AI = True
                
        self.data["backbone"] = self.value_type(self.backbone, str)
        self.data["fpn_mode"] = self.value_type(self.fpn_mode, int)
        self.data["input_size"] = self.value_type(self.input_size, int)
        self.data["color_channel"] = self.value_type(self.color_channel, int)
        self.data["region"] = self.value_type(self.inputs["region"], int)
        self.data["tags"] = self.value_type(self.inputs["tags"], int)
        self.data["dropout"] = self.value_type(self.inputs["dropout"], float)
        self.data["anchor"] = self.value_type(self.inputs["anchor"], int)
        self.data["batchsize"] = self.value_type(self.inputs["batchsize"], int)
        self.data["savefile"] = self.value_type(self.inputs["savefile"], str)
        
        self.data["testfolder"] = self.value_type(self.inputs["testfolder"], str)
        self.data["nullskip"] = self.value_type(self.inputs["nullskip"], float)
        self.data["trainsize"] = self.value_type(self.inputs["trainsize"], float)
        self.data["huber"] = self.value_type(self.inputs["huber"], int)
        self.data["learning_rate"] = self.value_type(self.inputs["learning_rate"], float)
        self.data["steps"] = self.value_type(self.inputs["steps"], int)
        self.data["overlap"] = self.value_type(self.inputs["overlap"], float)
        self.data["port"] = self.value_type(self.inputs["port"], int)
        self.data["augment"] = self.value_type(self.inputs["augment"], int)
        self.data["non_max_suppression_iou"] = self.value_type(self.inputs["non_max_suppression_iou"], float)
        self.data["lstm"] = self.value_type(self.lstm, bool)
        self.data["normalization"] = self.value_type(self.normalization, bool)
        self.data["random_drop"] = self.value_type(self.inputs["random_drop"], float)

        with open("tkpik.pik", "wb") as fio:
            pickle.dump(self.data, fio)

        if self.data["savefile"]:
            if len(self.data["savefile"]) > 3:
                with open(self.data["savefile"] + "_cfg.pik", "wb") as fio:
                    pickle.dump(self.data, fio)
        self.RELOAD_AI_FLAG = reload_AI

    def train(self, early_stopping = False, cpu_training = False):
        self.update()
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
        print(EPOCH)
        for k in ("learning_rate", "steps"):
            if k not in self.data:
                print("no data", k)
                return False
            if not self.data[k]:
                print("no element", k)
                return False
            if not float(self.data[k]) > 0:
                print("no value", k, self.data[k])
                return False
        if not self.DET_MODEL or self.RELOAD_AI_FLAG:
            self.load_AI()
            self.RELOAD_AI_FLAG = False
            print("PLEASE LOAD WEIGHT")
            return False
        if self.data["trainsize"] >= 1.0:
            no_validation = True
        else:
            no_validation = False
        if cpu_training:
            v = self.DET_MODEL.cpu_train(EPOCH, self.data["steps"], self.data["learning_rate"], early_stopping = early_stopping, no_validation = no_validation)
        else:
            v = self.DET_MODEL.train(EPOCH, self.data["steps"], self.data["learning_rate"], early_stopping = early_stopping, no_validation = no_validation)
        
    def save(self):
        if not self.DET_MODEL:
            print("CANNOT SAVE, NO AI MODEL")
            return False
        print("widget.save")
        self.DET_MODEL.save(self.value_type(self.inputs["savefile"], str))
        

    def load(self):
        print("widget.load")
        #if not self.DET_MODEL or self.RELOAD_AI_FLAG:
        self.load_AI()
        self.RELOAD_AI_FLAG = False
        self.DET_MODEL.load(self.value_type(self.inputs["savefile"], str))
        

    def trial(self):
        self.update()
        if os.path.isdir(self.data["testfolder"]):
            print("widget.trial")
            self.DET_MODEL.trial(self.data["testfolder"])
        
    def generator_check(self):
        self.update()
        if not self.DET_MODEL or self.RELOAD_AI_FLAG:
            self.load_AI()
            self.RELOAD_AI_FLAG = False
        self.DET_MODEL.generator_check()

    def predict_image(self):
        self.update()
        if not self.DET_MODEL or self.RELOAD_AI_FLAG:
            self.load_AI()
            self.RELOAD_AI_FLAG = False
        filename = filedialog.askopenfilename(initialdir = self.initialdir, title = "Select a File", \
                                              filetypes = (("image files", ("*.bmp*", "*.jpg*", "*.png*")), \
                                                           ("all files", "*.*")))
        #FILE = simpledialog.askstring(title="Predict", prompt="File path:")
        if os.path.isfile(filename):
            self.DET_MODEL.predict(filename)
            self.initialdir = os.path.split(filename)[0]
        
        
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
    w.load_AI()
    w.root.mainloop()

        
def test():
    w = widget()
    label_width = 100
    input_width = 100
    height = 30
    w.label_w_input(w.frame_main, label_width, input_width, height, 0, 0, "cat", "cat", bind_return = lambda v: w.objects["dog"]["input"].focus_set())
    w.label_w_input(w.frame_main, label_width, input_width, height, 0, 1, "dog", "dog", bind_return = lambda v: w.objects["cat"]["input"].focus_set())
    w.label_w_input(w.frame_main, label_width, input_width*3, height, 1, 0, "hunam", "hunam", columnspan = 2)
    w.mainloop()




if __name__ == "__main__":
    run()
    #pass

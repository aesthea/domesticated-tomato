import csv
import os
import sys
import numpy as np
import statistics
import datetime
import math


import asyncio
import websockets

import importlib
import importlib.util

import json
import urllib

import functools
import binascii
import warnings

import HARR_VOTT_v2 as HARR_VOTT
import pickle

import base64
from io import BytesIO

warnings.filterwarnings("ignore")

predict = False
#Load this model on TKUI
if os.path.isfile("tkpik.pik"):
    with open("tkpik.pik", "rb") as fio:
        data = pickle.load(fio)


        model = HARR_VOTT.load_model(data["input_size"], data["color_channel"], data["tags"], data["region"], data["dropout"], data["fpn_mode"], data["backbone"], data["votts"])
        model.BATCH_SIZE = data["batchsize"]
        model.HUBER = data["huber"]
        model.TRAIN_SIZE = data["trainsize"]
        model.ANCHOR_LEVEL = data["anchor"]
        model.NULL_SKIP = data["nullskip"]
        model.OVERLAP_REQUIREMENT = data["overlap"]
        model.SAVENAME = data["savefile"]
        model.initialize()
            
        if os.path.isfile(data['savefile'] + ".pik"):
            model.load()
            ai_function = model.anchor
            ai_function.CROP_SIZE = (int(data['input_size']), int(data['input_size']))
            ai_function.load_tags(data['savefile'] + ".pik")
            predict = ai_function.predict
             

CONNECTED = set()
async def handler(websocket, path, predict):
    print(path)
    try:
        async for rawdata in websocket:
            client_ip, client_port = websocket.remote_address
            CONNECTED.add(websocket)
            print(client_ip, client_port, datetime.datetime.now())
            datatype = type(rawdata)
            #print("DATA RECEIVE TYPE : ", datatype)
            if datatype == bytes:
                with open(r"c:\test\byte.b", "wb") as fio:
                    fio.write(rawdata)
                    print("WRITE BINARY FILE")
            if datatype == str:
                result = {}
                try:
                    data = json.loads(rawdata)
                except Exception as e:
                    print("E114", e)
                    data = {}
                if "filename" in data and "bytes" in data:
                    fn, ext = os.path.splitext(data["filename"])
                    mimebytes = data["bytes"].split(",")
                    if len(mimebytes) == 2:
                        mime, bytedata = mimebytes
                        bytedata = base64.b64decode(bytedata)
                    fp = r"c:\test\fio" + ext
                    with open(fp, "wb") as fio:
                        fio.write(bytedata)
                    if "tensorflow"  in data:
                        if data["tensorflow"] and ext.lower() in (".bmp", ".jpg", ".png"):
                            print("TENSORFLOW", data["tensorflow"])
                            if predict:
                                pil_obj, raw_data = predict(fp, show = False, rawdata = True)

                                img_byte_arr = BytesIO()
                                pil_obj.save(img_byte_arr, format='JPEG', quality = 75, optimize = True, progressive = True)
                                img_byte_arr = img_byte_arr.getvalue()
                                
                                im_b64 = base64.b64encode(img_byte_arr).decode('utf-8', 'ignore')
                                result["result"] = ",".join(['data:image/jpeg;base64', im_b64])
                                for i, v in enumerate(raw_data):
                                    raw_data[i]["score"] = float(raw_data[i]["score"])
                                    
                                result["rawdata"] = raw_data

                                #SAVE FOR IMAGE VERIFICATION
                                with open(r"c:\test\result.png", "wb") as fio:
                                    fio.write(base64.b64decode(result["result"].split(",")[1]))
                                
                                with open(fp,"rb") as fio:
                                    test_b64 = base64.b64encode(fio.read()).decode()
                                    
                                result["test"] = ",".join(['data:image/png;base64', test_b64])
            asyncio.ensure_future(send(websocket, json.dumps(result, ensure_ascii = True)))
                            
    except websockets.exceptions.ConnectionClosedError as e:
        client_ip, client_port = websocket.remote_address
        print("E134", client_ip, e)
        
    except websockets.exceptions.ConnectionClosed as e:
        client_ip, client_port = websocket.remote_address
        print("E139", client_ip, e)

    except Exception as e:
        client_ip, client_port = websocket.remote_address
        print("E144", client_ip, e)
        
    finally:
        try:
            websocket.close()
        except Exception as e:
            pass
            
async def send(websocket, data):
    client_ip, client_port = websocket.remote_address
    disconnect_client = False
    try:
        await websocket.send(data)
    except websockets.ConnectionClosed as e:
        disconnect_client = e
    except asyncio.exceptions.CancelledError as e:
        disconnect_client = e
    except websockets.exceptions.ConnectionClosedError as e:
        disconnect_client = e
    except ConnectionResetError as e:
        disconnect_client = e
    except asyncio.exceptions.CancelledError as e:
        disconnect_client = e
    except Exception as e:
        disconnect_client = e
    if disconnect_client:
        print("E200", disconnect_client)
        try:
            websocket.close()
        except Exception as e:
            print("E204", e)
        try:
            CONNECTED.remove(websocket)
        except Exception as e:
            print("E208", e)

def broadcast(data):
    loop = asyncio.get_event_loop()
    tasks = []
    for websocket in CONNECTED:
        try:
            asyncio.ensure_future(send(websocket, data))
        except Exception as e0:
            print("E220", e0)
            try:
                websocket.close()
                CONNECTED.remove(websocket)
            except Exception as e1:
                print("E224", e1)

async def clear_session():
    while True:
        folder_dt_name = datetime.datetime.now().strftime("%Y%m%d")
        folder_target = os.path.join(WEBSERVER, folder_dt_name)
        if os.path.isdir(folder_target):
            pass
        else:
            os.mkdir(folder_target)
        tf.keras.backend.clear_session()
        await asyncio.sleep(60)
 
def run():
    #Load this model on TKUI
    if os.path.isfile("tkpik.pik"):
        with open("tkpik.pik", "rb") as fio:
            data = pickle.load(fio)
        model = HARR_VOTT.load_ai_by_pik()
        model.load()
        predict = model.anchor.predict
        start_server_00 = websockets.serve(functools.partial(handler, predict=predict), port = int(data['port']))
        asyncio.get_event_loop().run_until_complete(start_server_00)
        asyncio.get_event_loop().run_forever()
        print("RUN DATA %s SERVER" % data['port'])

    
if __name__ == "__main__":
    pass


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
import pickle
import base64
from io import BytesIO




#SPEC_LOADER = "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/DETECTOR_ALL/HARR_VOTT.py"
SPEC_LOADER = None

READ_LIMIT = 2**21
MAX_SIZE = 2**22

if not SPEC_LOADER:
    import HARR_VOTT
    PIK = "tkpik.pik"
    PORT = None
else:
    spec_name = os.path.splitext(os.path.split(SPEC_LOADER)[-1])[0]
    spec = importlib.util.spec_from_file_location(spec_name, SPEC_LOADER)
    HARR_VOTT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(HARR_VOTT)
    PIK = os.path.join(os.path.split(SPEC_LOADER)[0], "tkpik.pik")
    PORT = 8860

if not os.path.isdir("c:/test"):
    os.mkdir("c:/test")

warnings.filterwarnings("ignore") 
CONNECTED = set()
async def handler(websocket, path, model, port_no):
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
                    
                if "args" in data:
                    if "server" in data["args"]:
                        if "load_weight" in data["args"]["server"]:
                            #load_model()
                            model.load(model.LOADPATH)
                    if "anchor_size" in data["args"]:
                        result["anchor_size"] = data["args"]["anchor_size"]
                    if "nms_iou" in data["args"]:
                        result["nms_iou"] = data["args"]["nms_iou"]
                    if "image_size" in data["args"]:
                        result["image_size"] = data["args"]["image_size"]
                    if "image_format" in data["args"]:
                        result["image_format"] = data["args"]["image_format"]
                    if "segment_minimum_ratio" in data["args"]:
                        result["segment_minimum_ratio"] = data["args"]["segment_minimum_ratio"]
                    result["args"] = data["args"]
                    
                if "filename" in data and "base64" in data:
                    fn, ext = os.path.splitext(data["filename"])
                    mimebytes = data["base64"].split(",")
                    if len(mimebytes) == 2:
                        mime, bytedata = mimebytes
                        bytedata = base64.b64decode(bytedata)
                    fp = r"c:\test\fio" + ext
                    with open(fp, "wb") as fio:
                        fio.write(bytedata)
                    if "tensorflow"  in data:
                        if data["tensorflow"] and ext.lower() in (".bmp", ".jpg", ".png"):
                            print("TENSORFLOW", data["tensorflow"])
                            if model:
                                NMS_IOU = model.NON_MAX_SUPPRESSION_IOU
                                ANCHOR_SZ = model.ANCHOR_LEVEL
                                IMAGE_SZ = model.IMAGE_SIZE
                                if "nms_iou" in result:
                                    try:
                                        NMS_IOU = float(result["nms_iou"])
                                    except ValueError:
                                        NMS_IOU = model.NON_MAX_SUPPRESSION_IOU
                                    except TypeError:
                                        NMS_IOU = model.NON_MAX_SUPPRESSION_IOU
                                else:
                                    NMS_IOU = model.NON_MAX_SUPPRESSION_IOU
                                if "anchor_size" in result:
                                    try:
                                        ANCHOR_SZ = int(result["anchor_size"])
                                    except ValueError:
                                        ANCHOR_SZ = model.ANCHOR_LEVEL
                                    except TypeError:
                                        ANCHOR_SZ = model.ANCHOR_LEVEL
                                else:
                                    ANCHOR_SZ = model.ANCHOR_LEVEL
                                if "image_size" in result:
                                    try:
                                        IMAGE_SZ = int(result["image_size"])
                                    except ValueError:
                                        IMAGE_SZ = model.IMAGE_SIZE
                                    except TypeError:
                                        IMAGE_SZ = model.IMAGE_SIZE
                                else:
                                    IMAGE_SZ = model.IMAGE_SIZE
                                if "segment_minimum_ratio" in result:
                                    try:
                                        SEGMENT_MIN_RATIO = float(result["segment_minimum_ratio"])
                                    except ValueError:
                                        SEGMENT_MIN_RATIO = 0.75
                                    except TypeError:
                                        SEGMENT_MIN_RATIO = 0.75
                                else:
                                    SEGMENT_MIN_RATIO = 0.75                                  
                                #print("DEBUG 124", NMS_IOU, ANCHOR_SZ, IMAGE_SZ)
                                pil_obj, raw_data = model.predict(fp, \
                                                                  show = False, \
                                                                  rawdata = True, \
                                                                  nms_iou = NMS_IOU, \
                                                                  anchor_size = ANCHOR_SZ, \
                                                                  image_size = IMAGE_SZ, \
                                                                  segment_minimum_ratio = SEGMENT_MIN_RATIO)
                                for i, v in enumerate(raw_data):
                                    raw_data[i]["score"] = float(raw_data[i]["score"])
                                result["rawdata"] = raw_data   
                                return_image = True
                                if "rawdata_only" in data:
                                    if data["rawdata_only"]:
                                        return_image = False
                                    
                                if return_image:
                                    SAVE_FORMAT = "jpg"
                                    if "image_format" in result:
                                        if result["image_format"] == "png":
                                            SAVE_FORMAT = "png"

                                    if SAVE_FORMAT == "png":
                                        img_byte_arr = BytesIO()
                                        pil_obj.save(img_byte_arr, format='PNG', quality = 75, optimize = True, progressive = True)
                                        img_byte_arr = img_byte_arr.getvalue()
                                        im_b64 = base64.b64encode(img_byte_arr).decode('utf-8', 'ignore')
                                        result["result"] = ",".join(['data:image/png;base64', im_b64])
                                    else:
                                        img_byte_arr = BytesIO()
                                        pil_obj.save(img_byte_arr, format='JPEG', quality = 75, optimize = True, progressive = True)
                                        img_byte_arr = img_byte_arr.getvalue()
                                        im_b64 = base64.b64encode(img_byte_arr).decode('utf-8', 'ignore')
                                        result["result"] = ",".join(['data:image/jpeg;base64', im_b64]) 

                                    #SAVE FOR IMAGE VERIFICATION
                                    #with open(r"c:\test\result.png", "wb") as fio:
                                    #    fio.write(base64.b64decode(result["result"].split(",")[1]))

                                    #with open(fp,"rb") as fio:
                                    #    test_b64 = base64.b64encode(fio.read()).decode()    
                                    #result["original"] = ",".join(['data:image/png;base64', test_b64])
                                
            asyncio.ensure_future(send(websocket, json.dumps(result, ensure_ascii = True)))
            print(port_no, "SEND COMPLETE", datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                            
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
 
def run(PORT = PORT):
    #Load this model on TKUI
    if os.path.isfile(PIK):
        with open(PIK, "rb") as fio:
            data = pickle.load(fio)
        model = HARR_VOTT.load_model_by_pik(PIK)
        model.load(os.path.join(os.path.split(PIK)[0], data['savefile']))
        model.LOADPATH = os.path.join(os.path.split(PIK)[0], data['savefile'])
        if not PORT:
            PORT = int(data['port'])
        start_server_00 = websockets.serve(functools.partial(handler, model = model, port_no = PORT), \
                                           port = PORT, \
                                           max_size = MAX_SIZE, \
                                           read_limit = READ_LIMIT, \
                                           write_limit = READ_LIMIT)
        print("RUN DATA %s SERVER" % PORT)
        asyncio.get_event_loop().run_until_complete(start_server_00)
        asyncio.get_event_loop().run_forever()
        
    
if __name__ == "__main__":
    run_server = True
    if run_server:
        if os.path.isfile(PIK):
            with open(PIK, "rb") as fio:
                data = pickle.load(fio)
            model = HARR_VOTT.load_model_by_pik(PIK)
            model.load(os.path.join(os.path.split(PIK)[0], data['savefile']))
            model.LOADPATH = os.path.join(os.path.split(PIK)[0], data['savefile'])
            if not PORT:
                PORT = int(data['port'])
            start_server_00 = websockets.serve(functools.partial(handler, model = model, port_no = PORT), \
                                               port = PORT, \
                                               max_size = MAX_SIZE, \
                                               read_limit = READ_LIMIT, \
                                               write_limit = READ_LIMIT)
            print("RUN DATA %s SERVER" % PORT)     
            asyncio.get_event_loop().run_until_complete(start_server_00)
            asyncio.get_event_loop().run_forever()
               


#import csv
import os
from os import system
#import sys
import numpy as np
#import statistics
import datetime
#import math
import asyncio
import websockets
import importlib
import importlib.util
import json
import urllib
import functools
#import binascii
import warnings
import pickle
import base64
from io import BytesIO
import gc


#SPEC_LOADER = "C:/Users/CSIPIG0140/Desktop/HARR_VOTT TK/HARRVOTT_2024f/HARR_VOTT.py"
SPEC_LOADER = None

READ_LIMIT = 2**21
MAX_SIZE = 2**22

if not SPEC_LOADER:
    import HARR_VOTT
    import fixes
    PIK = "main_config.pik"
    PORT = None
else:
    spec_name = os.path.splitext(os.path.split(SPEC_LOADER)[-1])[0]
    spec = importlib.util.spec_from_file_location(spec_name, SPEC_LOADER)
    HARR_VOTT = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(HARR_VOTT)
    PIK = os.path.join(os.path.split(SPEC_LOADER)[0], "main_config.pik")
    PORT = 7788
    fixes_spec_path = os.path.join(os.path.split(SPEC_LOADER)[0], "fixes.py")
    fixes_spec = importlib.util.spec_from_file_location("fixes", fixes_spec_path)
    fixes = importlib.util.module_from_spec(fixes_spec)
    fixes_spec.loader.exec_module(fixes)

#fix asyncio OSError fail.
asyncio.windows_events.IocpProactor = fixes.IocpProactor

if not os.path.isdir("c:/test"):
    os.mkdir("c:/test")

warnings.filterwarnings("ignore") 
#CONNECTED = set()
async def handler(websocket, path, model, port_no):
    #print(path)
    try:
        async for rawdata in websocket:
            client_ip, client_port = websocket.remote_address
            print("\n", client_ip, client_port, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
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
                            model.load(model.LOADPATH)
                        if "stop_server" in data["args"]["server"]:
                            asyncio.get_event_loop().stop()
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
                            print("TENSORFLOW FUNC", data["tensorflow"])
                            if model:
                                NMS_IOU = model.NMS_IOU
                                ANCHOR_SZ = model.ANCHOR_SIZE
                                IMAGE_SZ = model.IMAGE_SHAPE[0]
                                if "nms_iou" in result:
                                    try:
                                        NMS_IOU = float(result["nms_iou"])
                                    except ValueError:
                                        NMS_IOU = model.NMS_IOU
                                    except TypeError:
                                        NMS_IOU = model.NMS_IOU
                                else:
                                    NMS_IOU = model.NMS_IOU
                                if "anchor_size" in result:
                                    try:
                                        ANCHOR_SZ = int(result["anchor_size"])
                                    except ValueError:
                                        ANCHOR_SZ = model.ANCHOR_SIZE
                                    except TypeError:
                                        ANCHOR_SZ = model.ANCHOR_SIZE
                                else:
                                    ANCHOR_SZ = model.ANCHOR_SIZE
                                if "image_size" in result:
                                    try:
                                        IMAGE_SZ = int(result["image_size"])
                                    except ValueError:
                                        IMAGE_SZ = model.IMAGE_SHAPE[0]
                                    except TypeError:
                                        IMAGE_SZ = model.IMAGE_SHAPE[0]
                                else:
                                    IMAGE_SZ = model.IMAGE_SHAPE[0]
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
                                IMAGE_SHAPE = (IMAGE_SZ, IMAGE_SZ, model.IMAGE_SHAPE[-1])
                                pil_obj, raw_data = model.predict(fp, \
                                                                  show = False, \
                                                                  input_shape = IMAGE_SHAPE, \
                                                                  anchor_size = ANCHOR_SZ, \
                                                                  nms_iou = NMS_IOU, \
                                                                  segment_minimum_ratio = SEGMENT_MIN_RATIO, \
                                                                  output_size = 480, \
                                                                  debug = False)
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
            gc.collect()
                            
    except websockets.exceptions.ConnectionClosedError as e:
        client_ip, client_port = websocket.remote_address
        print("ConnectionClosedError", client_ip, e)
        
    except websockets.exceptions.ConnectionClosed as e:
        client_ip, client_port = websocket.remote_address
        print("ConnectionClosed", client_ip, e)

    except Exception as e:
        client_ip, client_port = websocket.remote_address
        print("Exception", client_ip, e)
        
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

##def broadcast(websocket, data):
##    loop = asyncio.get_event_loop()
##    tasks = []
##    for websocket in CONNECTED:
##        try:
##            asyncio.ensure_future(send(websocket, data))
##        except Exception as e0:
##            print("E220", e0)
##            try:
##                websocket.close()
##                CONNECTED.remove(websocket)
##            except Exception as e1:
##                print("E224", e1)

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
            config = pickle.load(fio)
        MODEL_CONFIG_FILE = os.path.join(os.path.split(PIK)[0], config["savefile"] + '.pik')
        print("LOADING CONFIG", MODEL_CONFIG_FILE)
        model = HARR_VOTT.load_model_by_pik(MODEL_CONFIG_FILE)
        model.load(os.path.join(os.path.split(PIK)[0], config["savefile"]))
        model.LOADPATH = os.path.join(os.path.split(PIK)[0], config["savefile"])
        if not PORT:
            if config['port']:
                PORT = int(config['port'])
            else:
                print("PLEASE SET PORT NUMBER")
                PORT = None

        if PORT:
            start_server_00 = websockets.serve(functools.partial(handler, model = model, port_no = PORT), \
                                               port = PORT, \
                                               max_size = MAX_SIZE, \
                                               read_limit = READ_LIMIT, \
                                               write_limit = READ_LIMIT)
            print("RUN DATA %s SERVER" % PORT)
            system("title %s SERVER" % PORT)
            asyncio.get_event_loop().run_until_complete(start_server_00)
            asyncio.get_event_loop().run_forever()        
    
if __name__ == "__main__":
    run_server = True
    if run_server:
        if os.path.isfile(PIK):
            with open(PIK, "rb") as fio:
                config = pickle.load(fio)
            MODEL_CONFIG_FILE = os.path.join(os.path.split(PIK)[0], config["savefile"] + '.pik')
            print("LOADING CONFIG", MODEL_CONFIG_FILE)
            model = HARR_VOTT.load_model_by_pik(MODEL_CONFIG_FILE)
            model.load(os.path.join(os.path.split(PIK)[0], config["savefile"]))
            model.LOADPATH = os.path.join(os.path.split(PIK)[0], config["savefile"])
            if not PORT:
                if config['port']:
                    PORT = int(config['port'])
                else:
                    print("PLEASE SET PORT NUMBER")
                    PORT = None
                
            if PORT:
                start_server_00 = websockets.serve(functools.partial(handler, model = model, port_no = PORT), \
                                                   port = PORT, \
                                                   max_size = MAX_SIZE, \
                                                   read_limit = READ_LIMIT, \
                                                   write_limit = READ_LIMIT)
                print("RUN DATA %s SERVER" % PORT)
                system("title %s SERVER" % PORT)
                asyncio.get_event_loop().run_until_complete(start_server_00)
                asyncio.get_event_loop().run_forever()

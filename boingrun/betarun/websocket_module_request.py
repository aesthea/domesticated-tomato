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

warnings.filterwarnings("ignore")


CLIENTS = {}
DATA = {}
SERVER_DATA = {}
SERVER_DATA["error_logs"] = []
RECORDS = {}
MODULE_MTDATE = {}

async def handler(websocket, path, DATA = DATA, CLIENTS = CLIENTS, SERVER_DATA = SERVER_DATA, RECORDS = RECORDS, MODULE_MTDATE = MODULE_MTDATE):
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

            updated_key = []
            returned_set = []

            if client_ip not in DATA:
                DATA[client_ip] = {}

            DATA[client_ip]["active_time"] = datetime.datetime.now().timestamp()

            active_clients = [client_ip]
            
            for k in data:
                if "pipe_free" not in SERVER_DATA:
                    SERVER_DATA["pipe_free"] = True
                    
                if k == "server":
                    if data[k] == "data":
                        broadcast(active_clients, json.dumps({"data" : SERVER_DATA, "message" : "server data"}))
                    if data[k] == "clear":
                        SERVER_DATA["pipe_free"] = True
                        for pop_key in ("request_by", "request_start", "request_module"):
                            if pop_key in SERVER_DATA:
                                del SERVER_DATA[pop_key]
                        SERVER_DATA["error_logs"] = []
                        for pop_key in list(RECORDS):
                            del RECORDS[pop_key]
                        for pop_key in list(DATA):
                            del DATA[pop_key]
                        broadcast(active_clients, json.dumps({"data" : SERVER_DATA, "message" : "clear data"}))
                        
                if k == "raise_error":
                    raise
                        
                if k == "request":
                    module_list = [os.path.splitext(n)[0] for n in os.listdir("betarun\\modules") if os.path.splitext(n)[1] == ".py"]
                    if data[k] in module_list and SERVER_DATA["pipe_free"]:
                        DATA[client_ip]["request_time"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        broadcast(active_clients, json.dumps({"data" : DATA[client_ip], "message" : "processing : %s" % data[k]}))
                        print("RUN MODULE : ", data[k], datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                        #await asyncio.sleep(0.1)
                        
                        SERVER_DATA["pipe_free"] = False
                        SERVER_DATA["request_start"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        SERVER_DATA["request_by"] = client_ip
                        SERVER_DATA["request_module"] = data[k]
                        DATA[client_ip]["request_module"] = data[k]

                        #importlib.invalidate_caches()

                        LOADING_MODULE = "betarun.modules.%s" % data[k]
                        module_path_check = os.path.join("betarun\\modules", data[k] + ".py")
                        dtts = datetime.datetime.fromtimestamp(os.stat(module_path_check).st_mtime).strftime("%Y-%m-%d %H:%M:%S")
                        if LOADING_MODULE in MODULE_MTDATE:
                            if dtts == MODULE_MTDATE[LOADING_MODULE]:
                                pass
                            elif LOADING_MODULE in sys.modules:
                                del sys.modules[LOADING_MODULE]
                                print("FLUSH MODULE", LOADING_MODULE)
                                MODULE_MTDATE[LOADING_MODULE] = dtts
                            else:
                                MODULE_MTDATE[LOADING_MODULE] = dtts
                        else:
                            MODULE_MTDATE[LOADING_MODULE] = dtts

                        func = importlib.import_module(LOADING_MODULE)
                        try:
                            print("REQUIRE", func.REQUIRE)
                            require = func.REQUIRE
                        except Exception as e:
                            require = []
                            pass
                        args = {}
                        records = {}
                        if "args" in data:
                            args = data["args"]
                        if require:
                            for req in require:
                                if req in RECORDS:
                                    records[req] = RECORDS[req]
                        result = func.run(records, args)
                        if "records" in result:
                            for record_name in result["records"]:
                                RECORDS[record_name] = result["records"][record_name]

                        SERVER_DATA["pipe_free"] = True
                        for pop_key in ("request_by", "request_start", "request_module"):
                            if pop_key in SERVER_DATA:
                                del SERVER_DATA[pop_key]
                        SERVER_DATA["request_end"] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                        DATA[client_ip]["result"] = result["result"]
                        broadcast(active_clients, json.dumps({"data" : DATA[client_ip], "request": data, "message" : "request completed"}))
                        
                    elif "request_by" in SERVER_DATA:
                        broadcast(active_clients, json.dumps({"data" : DATA[client_ip], "request": data, "message" : "server busy by : %(request_by)s" % SERVER_DATA}))
                    else:
                        broadcast(active_clients, json.dumps({"data" : DATA[client_ip], "request": data, "message" : "what you requesting???"})) 
                if "__INIT__" in data:
                    broadcast(active_clients, json.dumps({"data" : DATA[client_ip], "request": data, "message" : "user log on"}))
                                   
            for k in list(DATA):
                if DATA[k]["active_time"] < (datetime.datetime.now() - datetime.timedelta(0, 600)).timestamp():
                    if k in DATA:
                        del DATA[k]
        
            print(client_ip, client_port, "SERVER pipe : ", SERVER_DATA["pipe_free"], datetime.datetime.now())
        websocket.close()
    except websockets.exceptions.ConnectionClosedError as e:
        client_ip, client_port = websocket.remote_address
        print("E222", client_ip, e)
        
    except websockets.exceptions.ConnectionClosed as e:
        client_ip, client_port = websocket.remote_address
        print("E224", client_ip, e)

    except Exception as e:
        client_ip, client_port = websocket.remote_address
        #raise e
        if client_ip in list(DATA):
            del DATA[client_ip]
            print("client not exist", client_ip)
        print("E229", client_ip, e)
        SERVER_DATA["pipe_free"] = True
        for pop_key in ("request_by", "request_start", "request_module"):
            if pop_key in SERVER_DATA:
                del SERVER_DATA[pop_key]
        SERVER_DATA["error_logs"].append([datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(e), client_ip])
    finally:
        pass
    
async def send(websocket, data):
    client_ip, client_port = websocket.remote_address
    disconnect_client = False
    try:
        await websocket.send(data)
    except websockets.ConnectionClosed as e:
        disconnect_client = True
        print(client_ip, e)
    except asyncio.exceptions.CancelledError as e:
        disconnect_client = True
        print(client_ip, e)
    except websockets.exceptions.ConnectionClosedError as e:
        disconnect_client = True
        print(client_ip, e)
    except ConnectionResetError as e:
        disconnect_client = True
        print(client_ip, e)        
    except asyncio.exceptions.CancelledError as e:
        disconnect_client = True
        print(client_ip, e)
    except Exception as e:
        disconnect_client = True
        print(client_ip, e)
    if disconnect_client:
        if client_ip in list(CLIENTS):
            try:
                CLIENTS[client_ip].close()
            except Exception as e:
                print("CLOSE ERROR", client_ip, e)
            del CLIENTS[client_ip]
            
        

def broadcast(active_clients, data):
    loop = asyncio.get_event_loop()
    tasks = []
    for client_ip in active_clients:
        if client_ip not in CLIENTS:
            pass
        else:
            websocket = CLIENTS[client_ip]
            asyncio.ensure_future(send(websocket, data))
            #asyncio.create_task(send(websocket, message))


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
    print("RUN MODULE SERVER : 8650")
    start_server_00 = websockets.serve(handler, port = 8650, max_size=2**24)
    asyncio.get_event_loop().run_until_complete(start_server_00)
    asyncio.get_event_loop().run_forever()


    
if __name__ == "__main__":
    run()


import os
import copy
import cv2
import datetime
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import importlib.util
import json
import math
import matplotlib.colors as mpcolors
import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from PIL import Image
import random
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
import tensorflow_addons as tfa
import urllib
import pandas as pd
import base64

import warnings
warnings.filterwarnings("error", category = RuntimeWarning)

try:
    import model as det
except Exception as e:
    SPEC_LOADER = os.path.join(os.path.split(__file__)[0], "model.py")
    spec_name = os.path.splitext(os.path.split(SPEC_LOADER)[-1])[0]

    spec = importlib.util.spec_from_file_location(spec_name, SPEC_LOADER)
    det = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(det)


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

OUTPUT_SIZE = 480
DEFAULT_TAGS_PICKLE = "C:/test/tags.pik"
DEFAULT_SEGMENT_PATH = "C:/test/fractured_img"

def preprocess_func(im):
    if im.ndim == 4:
        imshape = im.shape
        im = im.reshape([imshape[0], imshape[1] * imshape[2] * imshape[3]])
        immax = np.expand_dims(np.max(im, 1), 1)
        immin = np.expand_dims(np.min(im, 1), 1)
        im = (((im - immin) / (immax - immin + 1e-6)) * 255).astype(np.int16)
        return im.reshape(imshape)
    return  ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6) * 255).astype(np.int16)

def preprocess_func_ori(im):
    return  ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6) * 255).astype(np.int16)

class loader:
    def __init__(self):
        self.df = pd.DataFrame(columns = ["source", "path", "x1", "y1", "x2", "y2", "label", "color"])
        self.NULL = 999
        self.SEGMENT_FILES = None
        self.do_not_auto_save_tags = False
        self.IMAGE_FOLDER = None

    def load_from_csv(self, fp = "C:/Users/CSIPIG0140/Desktop/TF SIMPLE IMG CLASSIFIER/TRAINING/LABEL_FILE.csv"):
        column_names = ['folder', 'filename', 'timeseek', 'path', 'label', 'x1', 'y1', 'x2', 'y2', 'color', 'modified_dt']
        df = pd.read_csv(fp, delimiter = ",", names = column_names)
        df["source"] = fp
        df["path"] = df.apply(lambda d : os.path.join(os.path.split(d.source)[0], d.path), axis = 1)
        self.df = pd.concat([self.df, df[["source", "path", "x1", "y1", "x2", "y2", "label", "color"]]], ignore_index = True)

    def load_from_vott(self, fp = "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/TAPING_PROBE_PIN/type2 train/vott-json-export/TAPING-PIN-PROBE-type2-train-export.json"):
        tags = {}
        with open(fp,"r") as fio:
            obj = json.load(fio)
            
        for t in obj["tags"]:
            tags[t["name"]] = t["color"]
                    
        for k in obj["assets"]:
            asset = obj["assets"][k]
            path = os.path.join(os.path.split(fp)[0], asset["asset"]["name"])
            if os.path.isfile(path):
                for r in asset["regions"]:
                    data = {}
                    tag = r['tags'][0]
                    data["source"] = fp
                    data["path"] = path
                    data["label"] = tag
                    data["x1"] = r['boundingBox']['left']
                    data["y1"] = r['boundingBox']['top']
                    data["x2"] = data["x1"] + r['boundingBox']['width']
                    data["y2"] = data["y1"] + r['boundingBox']['height']
                    data["color"] = tags[tag]
                    self.df = pd.concat([self.df, pd.DataFrame.from_records([data])], ignore_index = True)

    def add_tag(self, label, color):
        self.tags = pd.concat([self.tags, pd.DataFrame([[label, color]], columns = ["label", "color"])], verify_integrity = True, ignore_index = True)
        self.tags.drop_duplicates("label", inplace=True, keep='first')
        self.tags.reset_index(inplace=True)
        self.tags = self.tags[["label", "color"]]

    def setup_tags(self, tagfile, nosave = False):
        if tagfile:
            hastag = False
            if os.path.isfile(tagfile):
                self.tags = pd.read_pickle(tagfile)
                print("using pickled tag file : ", tagfile)
                hastag = True
            else:
                hastag = False
                pass
            tags = self.df.copy()
            tags.drop_duplicates("label", inplace=True, keep='last')
            tags.reset_index(inplace=True)
            if hastag:
                self.tags = pd.concat([self.tags, tags[['label', 'color']]], ignore_index = True)
                self.tags.drop_duplicates("label", inplace=True, keep='last')
                self.tags.reset_index(inplace=True)
            else:
                self.tags = tags
            self.tags = self.tags[["label", "color"]]
            if not nosave:
                self.tags.to_pickle(tagfile)
                print("saved tags to >> ", tagfile)
        else:
            tags = self.df.copy()
            tags.drop_duplicates("label", inplace=True, keep='last')
            tags.reset_index(inplace=True)
            self.tags = tags[['label', 'color']].copy()
            
    def save_tags(self, tagfile):
        self.tags.to_pickle(tagfile)
            
    def prepare(self, frac = 0.7, tagfile = None):
        if self.df.source.count() > 0:
            g = self.df.groupby("path")
            if train_test_split in (0, 1):
                self.train = self.df
                self.test = pd.DataFrame(columns = ["source", "path", "x1", "y1", "x2", "y2", "label", "color"])
            else:
                self.train = g.sample(frac = frac).reset_index(drop=True)
                self.test = self.df.sample(frac = 1.0).drop(self.train.index).reset_index(drop=True)
        elif self.SEGMENT_FILES:
            fs = self.SEGMENT_FILES
            random.shuffle(fs)
            if frac != 0 or frac != 1.0:
                self.train = fs[: int(len(fs) * frac)]
                self.test = fs[int(len(fs) * frac) : ]
            else:
                self.train = fs
                self.test = pd.DataFrame(columns = ["source", "path", "x1", "y1", "x2", "y2", "label", "color"])
        self.setup_tags(tagfile, nosave = self.do_not_auto_save_tags)

    def one_file(self, batch_set):
        g = batch_set.groupby("path")
        y = g.indices
        li = list(y.keys())
        random.shuffle(li)
        for k in li:
            one_img = batch_set.iloc[y[k]].copy()
            fp = k
            one_img["tag_indice"] =  one_img.apply(lambda x : (self.tags.label == x.label).idxmax(), axis = 1)
            region = one_img[["x1", "y1", "x2", "y2", "tag_indice"]]
            if os.path.isfile(fp):
                image_io = tf.io.read_file(fp)
                if os.path.splitext(fp)[-1].lower() == ".bmp":
                    tf_img = tf.image.decode_bmp(image_io, channels=3)
                else:
                    tf_img = tf.image.decode_jpeg(image_io, channels=3)
                tf_img = tf.expand_dims(tf_img,0).numpy()
                b, h, w, c = tf_img.shape
                if not np.any(region[["x1", "y1", "x2", "y2"]] > 1.0):
                    region = region * [w, h, w, h, 1]
                yield fp, tf_img.astype(np.int16), np.expand_dims(np.array(region), 0).astype(np.int16)

    def one_file_exif(self, file_list):
        for fp in file_list:
            ff, fi, fb = self.load_exif(fp)
            yield ff, fi, fb

    def save_exif(self, fp, im, bb):
        im = tf.keras.preprocessing.image.array_to_img(im)
        e = im.getexif()
        df = pd.DataFrame(bb[0], columns= ["x1", "y1", "x2", "y2", "label_index"])
        lbc = self.tags.iloc[df.label_index].reset_index()
        ddf = pd.concat([df, lbc[["label","color"]]], axis = 1)
        ddf.drop_duplicates(None, inplace=True, keep='last')
        ddf.reset_index(inplace=True)
        #e[37510] = base64.b64encode(bb.tobytes()).decode()
        e[37510] = ddf.to_json()
        im.save(fp, exif = e)

    def load_exif(self, fp, return_boolean = False):
        im = Image.open(fp)
        e = im.getexif()
        try:
            df = pd.read_json(e[37510])
            df["tag_indice"] =  df.apply(lambda x : (self.tags.label == x.label).idxmax(), axis = 1)
            df["exist"] =  df.apply(lambda x : np.any((self.tags.label == x.label)), axis = 1)
            #ADD NEW TAG TO CURRENT LIST IF DOES NOT EXISTS.
            if np.any(df.exist == False):
                print("ADD IN NEW TAG")
                self.tags = pd.concat([self.tags, df[["label", "color"]][df.exist == False]], verify_integrity = True, ignore_index = True)
                self.tags.drop_duplicates("label", inplace=True, keep='first')
                self.tags.reset_index(inplace=True)
                self.tags = self.tags[["label", "color"]]
                
            bb = np.array(df[["x1", "y1", "x2", "y2", "tag_indice"]])[df.exist]
            if not return_boolean:
                return fp, np.expand_dims(tf.keras.utils.img_to_array(im), 0), np.expand_dims(bb, 0)
            else:
                if bb.shape[0] > 0:
                    return True
        except ValueError as e:
            print(e)
        return False

    def augment_batch(self, im, bb, seq = None):
        IN_VIEW = 0.95
        MINIMUM_SIZE = 0.10
        aug_im, aug_bbwc = augment(im, bb, seq)
        w = aug_im.shape[1]
        h = aug_im.shape[0]
        local_bbox = np.ndarray([0, 4], dtype = np.float16)
        local_label = np.ndarray([0, 1], dtype = np.int16)
        new_array_sorter = []
        for bb in aug_bbwc:
            x1 = bb.x1
            x2 = bb.x2
            y1 = bb.y1
            y2 = bb.y2
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            #get most center samples ---------------
            c = math.sqrt(pow(max(w//2, cx) - min(w//2, cx), 2) + pow(max(h//2, cy) - min(h//2, cy), 2))
            label = int(bb.label)
            if(bb_intersection([x1, y1, x2, y2], [0, 0, w, h]) > IN_VIEW) and (x2 - x1) / w > MINIMUM_SIZE and (y2 - y1) / h > MINIMUM_SIZE:
                new_array_sorter.append([x1, y1, x2, y2, label, c])
                
            if len(new_array_sorter) == 0:
                continue
        new_array_sorter.sort(key = lambda k : k[5])
        np_array = np.array(new_array_sorter, np.float16)
        if np_array.shape[0] == 0:
            np_array = np_array.reshape((0,6))
        local_bbox = np.append(local_bbox, np_array[:, [1,0,3,2]] / [h, w, h, w], 0)
        local_label = np.append(local_label, np.expand_dims(np_array[:, 4], -1), 0)
        return aug_im, local_bbox, local_label

    def batch(self, t, batch_size = 32, regions = 5, image_shape = (96, 96, 3), anchor_size = 4, skip_null = 0.5, seq = None, name = None):
        print("BATCH LEN T >> ", len(t), "NAME:", name)
        running_file = os.path.join(os.path.split(__file__)[0], "running_batch.txt")
        batch_boundary_box = np.ndarray([0, regions, 4], dtype = np.float16)
        batch_label = np.ndarray([0, regions, 1], dtype = np.int16)
        batch_image = np.ndarray([0, image_shape[0], image_shape[1], image_shape[2]], dtype = np.int16)
        anc = anchor(anchor_size, crop_size = image_shape[:2])
        need_anchor = True
        batch_bb = []
        batch_no_bb = []
        split_bb_group = True
        to_save_bb_list = True
        no_skip_null_check = False
        if skip_null >= 1:
            no_skip_null_check = True
        if type(t) == list:
            if len(t) == 2:
                if type(t[0]) == list and type(t[1]) == list:
                    batch_bb = t[0]
                    batch_no_bb = t[1]
                    split_bb_group = False
                    to_save_bb_list = False
                    
        with open(running_file, "w") as fio:
            fio.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        while os.path.isfile(running_file):
            if len(batch_bb) > 0 or len(batch_no_bb) > 0:
                if to_save_bb_list and self.IMAGE_FOLDER:
                    pickle_data = [batch_bb, batch_no_bb]
                    fp = os.path.join(self.IMAGE_FOLDER, "image_group.pik")
                    with open(fp, "wb") as fio:
                        print("SAVE PICKLE GROUP >> ", fp)
                        pickle.dump(pickle_data, fio)
                        to_save_bb_list = False
                split_bb_group = False
                random.shuffle(batch_bb)
                random.shuffle(batch_no_bb)
                t = copy.copy(batch_bb)
                t.extend(batch_no_bb[: int(len(batch_bb) * (1 - skip_null))])
                random.shuffle(t)
                
            if type(t) == pd.DataFrame:
                iter_of = self.one_file(t)
                need_anchor = True
            elif type(t) == list:
                if len(t) == 2:
                    if type(t[0]) == list and type(t[1]) == list:
                        if len(t[0]) > 0 or len(t[1]) > 0:
                            pass
                        else:
                            t = []
                print("ITER >> ", len(batch_bb), " / ", len(t), " SEGREGATE IMG.BB >> ", split_bb_group)
                iter_of = self.one_file_exif(t)
                need_anchor = False
            else:
                return False
            for fp, im, bb in iter_of:
                if need_anchor:
                    iter_anc = anc.make(im, bb)
                else:
                    iter_anc = [[im, bb]]
                    if split_bb_group:
                        if bb.shape[1] == 0:
                            batch_no_bb.append(fp)
                        else:
                            batch_bb.append(fp)
                            
                for ni, nb in iter_anc:
                    if nb.shape[1] == 0 and not no_skip_null_check:
                        rng = random.random()
                        if rng < skip_null:
                            continue

                    if image_shape[-1] == 1:
                        ni = tf.image.rgb_to_grayscale(ni).numpy()

                    if not need_anchor:
                        _, nh, nw, nc = ni.shape
                        ni = tf.image.resize(ni, image_shape[:2], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None).numpy()
                        nb = nb * [image_shape[0], image_shape[1], image_shape[0], image_shape[1], 1] // [nw, nh, nw, nh, 1]

                    ni = preprocess_func(ni)
                    aug_im, aug_bb, aug_lb = self.augment_batch(ni, nb, seq)
                    if aug_bb.shape[0] == 0 and not no_skip_null_check:
                        rng = random.random()
                        if rng < skip_null:
                            continue        
                    if batch_boundary_box.shape[1] > aug_bb.shape[0]:
                        for i in range(batch_boundary_box.shape[1] - aug_bb.shape[0]):
                            aug_bb = np.append(aug_bb, [[0, 0, 1, 1]], axis = 0)
                            aug_lb = np.append(aug_lb, [[self.NULL]], axis = 0)
                    elif batch_boundary_box.shape[1] < aug_bb.shape[0]:
                        aug_bb = aug_bb[: batch_boundary_box.shape[1], :]
                        aug_lb = aug_lb[: batch_boundary_box.shape[1], :]
                    batch_image = np.append(batch_image, np.expand_dims(aug_im, 0), axis = 0)
                    batch_boundary_box = np.append(batch_boundary_box, np.expand_dims(aug_bb, 0), axis = 0)
                    batch_label = np.append(batch_label, np.expand_dims(aug_lb, 0), axis = 0)
                    if batch_boundary_box.shape[0] == batch_size:
                        batch_boundary_box = np.where(batch_boundary_box < 0, 0, batch_boundary_box)
                        batch_boundary_box = np.where(batch_boundary_box > 1, 1, batch_boundary_box)
                        yield batch_image.astype(np.int16), (batch_label.astype(np.int16), batch_boundary_box.astype(np.float16))
                        batch_boundary_box = np.ndarray([0, regions, 4], dtype = np.float16)
                        batch_label = np.ndarray([0, regions, 1], dtype = np.int16)
                        batch_image = np.ndarray([0, image_shape[0], image_shape[1], image_shape[2]], dtype = np.int16)
                        
    def exif_prepare(self, fol = "C:/test/fractured_img", tagfile = 'C:/test/tags.pik', frac = 0.7):
        self.IMAGE_FOLDER = fol
        load_pik_fp = os.path.join(fol, "image_group.pik")
        fs = os.scandir(fol)
        fs = [n.path for n in fs if os.path.splitext(n.name)[1] == ".jpg" and n.is_file()]
        if os.path.isfile(load_pik_fp):
            with open(load_pik_fp, "rb") as fio:
                loaded_pik_group = pickle.load(fio)
            img_bb = loaded_pik_group[0]
            img_no_bb = loaded_pik_group[1]
            print("IMAGE FILES", len(fs), " -- ", len(img_bb), " - ", len(img_no_bb))
            if len(fs) == len(img_bb) + len(img_no_bb):
                print("RUN WITH GROUPED IMAGE")
                if frac != 0 or frac != 1.0:
                    self.train = [img_bb[: int(len(img_bb) * frac)], img_no_bb[: int(len(img_no_bb) * frac)]]
                    self.test = [img_bb[int(len(img_bb) * frac) : ], img_no_bb[int(len(img_no_bb) * frac) : ]]
                else:
                    self.train = [img_bb, img_no_bb]
                    self.test = pd.DataFrame(columns = ["source", "path", "x1", "y1", "x2", "y2", "label", "color"])
                return True
            else:
                pass
        random.shuffle(fs)
        self.SEGMENT_FILES = fs
        self.setup_tags(tagfile, nosave = self.do_not_auto_save_tags)
        self.df = pd.DataFrame(columns = ["source", "path", "x1", "y1", "x2", "y2", "label", "color"])
        
        if frac != 0 or frac != 1.0:
            self.train = fs[: int(len(fs) * frac)]
            self.test = fs[int(len(fs) * frac) : ]
        else:
            self.train = fs
            self.test = pd.DataFrame(columns = ["source", "path", "x1", "y1", "x2", "y2", "label", "color"])
        return True
            
    def save_as_segment_image(self, extract_path, image_shape = (96, 96, 3), anchor_size = 4, tagfile = None, with_labels_only = False):
        print("SAVE SEGMENT WITH ANCHOR LV: ", anchor_size)
        anc = anchor(anchor_size, crop_size = image_shape[:2])
        self.prepare(1.0, tagfile)
        total_files = self.train.groupby("path").path.nunique().count()
        iter_of = self.one_file(self.train)
        dtnow = datetime.datetime.now()
        dtstart = datetime.datetime.now()
        if not os.path.isdir(extract_path):
            os.mkdir(extract_path)
        for i, a in enumerate(iter_of):
            fp, im, bb = a
            iter_anc = anc.make(im, bb)
            if i % 100 == 0 or (datetime.datetime.now() - dtnow).total_seconds() > 1800:
                print(i, "/", total_files, (datetime.datetime.now() - dtnow).total_seconds())
                dtnow = datetime.datetime.now()
            for j, b in enumerate(iter_anc):
                ni, nb = b
                if nb.shape[1] > 0 or not with_labels_only:
                    p, fn = os.path.split(fp)
                    f, ext = os.path.splitext(fn)
                    savefile = os.path.join(extract_path, "%s_%03d.jpg" % (f, j))
                    self.save_exif(savefile, ni[0], nb)
        print("FINISH", total_files, "/", total_files, (datetime.datetime.now() - dtstart).total_seconds())

class detection_model:
    def __init__(self, image_shape = (64, 64, 3), detection_region = 2, classes = 1000, backbone = "B0", dropout = 0.2):
        self.m = det.edet(input_shape = image_shape, num_classes = classes, detection_region = detection_region, dropout = dropout, backbone = backbone)
        self.m.compile(optimizer = Adam(learning_rate = 0.01), \
                       loss = {'regression': det.regression_loss, 'classification': det.classification_loss})
        self.c = loader()
        self.c.NULL = classes - 1
        self.c.color_channel = image_shape[-1]
        self.c.image_size = image_shape[:2]
        self.regions = detection_region
        self.image_shape = image_shape
        self.anchor_size = 4

    def load_csv(self, path):
        self.c.load_from_csv(path)

    def load_vott(self, path):
        self.c.load_from_vott(path)

    def segment(self, extract_path = "C:/test/fractured_img", tagfile = "C:/test/tags.pik", image_shape = (128, 128, 3), anchor_size = 4):
        dt_now = datetime.datetime.now()
        self.c.save_as_segment_image(extract_path, image_shape = image_shape, anchor_size = 4, tagfile = tagfile)
        print("DONE ", (datetime.datetime.now() - dt_now).total_seconds())

    def prepare(self, tagfile, frac):
        self.c.prepare(frac = frac, tagfile = tagfile)

    def load_folder(self, path, tagfile, frac):
        self.c.exif_prepare(path, tagfile, frac)
        
    def train(self, learning_rate = 0.001, epoch = 10, steps = 10, batch_size = 32, skip_null = 0.5, augment_seq = None, callback_earlystop = False):
        self.m.optimizer.learning_rate = learning_rate
        print("PREPARE TRAIN BATCH")
        train = self.c.batch(self.c.train, \
                             batch_size = batch_size, \
                             regions = self.regions, \
                             image_shape = self.image_shape, \
                             anchor_size = self.anchor_size, \
                             skip_null = skip_null, \
                             seq = augment_seq, \
                             name = "TRAIN")
        test_alive = False
        if type(self.c.test) == pd.DataFrame:
            if self.c.test.source.count() > 0:
                test_alive = True
        elif type(self.c.test) == list:
            if len(self.c.test) > 0:
                test_alive = True
                if len(self.c.test) == 2:
                    if type(self.c.test[0]) == list and type(self.c.test[1]) == list:
                        test_alive = False
        if test_alive:
            #TEST BATCH IS AVAILABLE.
            callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, verbose = 0, mode = "min", restore_best_weights = True)
            print("PREPARE TEST BATCH")
            test = self.c.batch(self.c.test, \
                                batch_size = batch_size, \
                                regions = self.regions, \
                                image_shape = self.image_shape, \
                                anchor_size = self.anchor_size, \
                                skip_null = skip_null, \
                                seq = augment_seq, \
                                name = "TEST")
            if callback_earlystop:
                print("TRAIN WITH CALLBACK EARLYSTOP")
                self.history = self.m.fit(train, \
                                          epochs = epoch, \
                                          steps_per_epoch = steps, \
                                          validation_data = test, \
                                          validation_steps = steps // 2, \
                                          verbose = 2, \
                                          callbacks=[callback])
            else:
                self.history = self.m.fit(train, \
                                          epochs = epoch, \
                                          steps_per_epoch = steps, \
                                          validation_data = test, \
                                          validation_steps = steps // 2, \
                                          verbose = 2)
        else:
            #RUN WITHOUT TEST BATCH
            callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 20, verbose = 0, mode = "min", restore_best_weights = True)
            if callback_earlystop:
                print("TRAIN WITH CALLBACK EARLYSTOP NO TEST BATCH")
                self.history = self.m.fit(train, \
                                          epochs = epoch, \
                                          steps_per_epoch = steps, \
                                          verbose = 2, \
                                          callbacks=[callback])
            else:
                print("TRAIN WITHOUT TEST BATCH")
                self.history = self.m.fit(train, \
                                          epochs = epoch, \
                                          steps_per_epoch = steps, \
                                          verbose = 2)

    def save(self, f = None):
        if not f:
            f = "detection"
        self.m.save_weights(f)

    def load(self, f = None):
        if not f:
            f = "detection"
        try:
            self.m.load_weights(f)
        except Exception as e:
            print(e)

    def clear(self):
        if 'sanity_check_sample' in dir(self):
            del(self.sanity_check_sample)
        
    def sanity_check(self, batch_size = 36, skip_null = 0.7, seq = None):
        if 'sanity_check_sample' not in dir(self):
            self.sanity_check_sample = self.c.batch(self.c.train, batch_size = batch_size, regions = self.regions, image_shape = self.image_shape, anchor_size = self.anchor_size, skip_null = skip_null, seq = seq)
        x,(y,z) = next(self.sanity_check_sample)
        if np.max(x) <= 1.0:
            x = x * 255
        if x.shape[-1] == 1:
            x = tf.convert_to_tensor(x, tf.int16)
            x = tf.image.grayscale_to_rgb(x, name=None)
            x = x.numpy()
        x = preprocess_func(x)
        colors = [[255, 0, 0], ]
        imbb = tf.image.draw_bounding_boxes(x, z, colors)
        items = imbb.shape[0]
        w = int(math.ceil(math.sqrt(items)))
        h = math.ceil(items / w)
        fig, ax = plt.subplots(h , w)
        for i, im in enumerate(imbb):
            row = i // w
            col = i % w
            img = tf.keras.preprocessing.image.array_to_img(im)
            title = ",".join(self.c.tags.iloc[y[i][np.where(y[i] < self.c.tags.index.stop)]].label.tolist())
            if h == 1 and w == 1:
                ax.imshow(img)
                ax.axis('off')
                ax.set_title(title, fontdict= {'fontsize': 6})
            else:
                ax[row, col].imshow(img)
                ax[row, col].axis('off')
                ax[row, col].set_title(title, fontdict= {'fontsize': 6})
        plt.show()
        plt.clf()
        plt.close()
        
    def chart(self, show = True):
        plt.figure(figsize=(8, 5))
        plt.subplot(1, 2, 1)
        epochs_range = range(len(self.history.history["loss"]))
        for k in ('accuracy', 'val_accuracy', 'regression_loss', 'val_regression_loss', 'classification_loss', 'val_classification_loss'):
            if k in self.history.history:
                v = self.history.history[k]
                plt.plot(epochs_range, v, label=k)
        plt.legend(loc='upper right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        for k in ('loss', 'val_loss'):
            if k in self.history.history:
                v = self.history.history[k]
                plt.plot(epochs_range, v, label=k)                
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        
        if show:
            plt.show()
        else:
            plt.savefig("training_result.png")
        plt.clf()
        plt.close()        

    def predict(self, fp, output_size = OUTPUT_SIZE, nms_iou = None, debug = False, save_to_test = False, anchor_size = None, image_size = None):
        #print("DEBUG 594", nms_iou, anchor_size, image_size)
        raw_image = tf.io.read_file(fp)
        model_channel = self.m.input.shape[-1]
        if os.path.splitext(fp)[-1].lower() == ".bmp":
            raw_image = tf.image.decode_bmp(raw_image, channels=3)
        else:
            raw_image = tf.image.decode_jpeg(raw_image, channels=3)
        if model_channel == 1:
            image = tf.image.rgb_to_grayscale(raw_image)
        else:
            image = copy.copy(raw_image)  
        image = tf.expand_dims(image,0)
        
        raw_image = tf.expand_dims(raw_image, 0)

        if not nms_iou:
            nms_iou = 0.001
        if anchor_size:
            ANCHOR_SZ = anchor_size
        else:
            ANCHOR_SZ = self.anchor_size
        if image_size:
            IMAGE_SZ = image_size
        else:
            IMAGE_SZ = self.c.image_size
        anc = anchor(ANCHOR_SZ, crop_size = IMAGE_SZ)
        anchor_gen = anc.make(image)
        classifier_list = np.array([0])
        box_prediction_list = np.array([0])
        with tf.device("cpu:0"):
            for image_for_predict, bb in anchor_gen:
                image_for_predict = tf.image.resize(image_for_predict, self.m.input.shape[1:3])
                image_for_predict = preprocess_func(image_for_predict.numpy())
                classifier, box_prediction = self.m.predict(image_for_predict, verbose = 0)
                if classifier_list.any():
                    classifier_list = np.append(classifier_list, classifier, 0)
                    box_prediction_list = np.append(box_prediction_list, box_prediction, 0)
                else:
                    classifier_list = classifier
                    box_prediction_list = box_prediction
                
        inference = (classifier_list, box_prediction_list)
    
        if debug:
            debug_grid = tf.image.draw_bounding_boxes(image_for_predict, inference[1], [[255, 0, 0]])
            classifier = tf.argmax(inference[0], axis = -1).numpy()
            sqr_grid = math.ceil(math.sqrt(debug_grid.shape[0]))
            fig,ax = plt.subplots(sqr_grid , sqr_grid)
            for i, im in enumerate(debug_grid):
                im = im.numpy()
                row = i // sqr_grid
                col = i % sqr_grid
                ax[row, col].set_title(str(i), fontdict= {'fontsize': 6})         
                im = tf.keras.preprocessing.image.array_to_img(im)
                ax[row, col].imshow(im)
                ax[row, col].axis('off')
            plt.show()
            plt.clf()
            plt.close()

        classifier = tf.argmax(inference[0], axis = -1).numpy()
        arr = np.expand_dims(classifier,-1)
        score = np.take_along_axis(inference[0], arr, 2)[:,:,0]

        score = score[np.where(classifier < self.c.tags.index.stop)]
        inference_bb = inference[1][np.where(classifier < self.c.tags.index.stop)]
        anchor_crop = anc.boxes[np.where(classifier < self.c.tags.index.stop)[0]]
        classifier_index = classifier[np.where(classifier < self.c.tags.index.stop)]
        tags = self.c.tags.iloc[classifier[np.where(classifier < self.c.tags.index.stop)]]
        tags.index = range(tags.count().label)

        anchor_position = anchor_crop * [image.shape[2], image.shape[1], image.shape[2], image.shape[1]] #xyXY
        anchor_reposition = (anchor_position[:, [2, 3, 2, 3]] - anchor_position[:, [0, 1, 0, 1]]) / [image.shape[2], image.shape[1], image.shape[2], image.shape[1]]     

        boundary_boxes = inference_bb * anchor_reposition
        boundary_boxes = boundary_boxes + anchor_crop[:, [0, 1, 0, 1]]
        
        nms = tf.image.non_max_suppression(boundary_boxes, score, 100, nms_iou)

        nms_boundary_box = boundary_boxes[nms]

        if raw_image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(raw_image, name=None)
        else:
            image = raw_image

        output_size = min(output_size, max(image.shape[1], image.shape[2]))
        
        image = tf.image.resize(image, (output_size, output_size), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True, antialias=False, name=None)
        nms_colors = [hex2rgb(n) for n in tags.iloc[nms].color.tolist()]
        colors = [hex2rgb(n) for n in tags.color.tolist()]

        if nms_boundary_box.shape[0]:
            output_im = tf.image.draw_bounding_boxes(image, np.expand_dims(nms_boundary_box, 0), nms_colors)
        else:
            output_im = image
            
        ktags = tags.iloc[nms].to_dict('index')
        output_im = output_im.numpy()[0]

        prediction_result = []
        for k in ktags:
            pos = (boundary_boxes[k][[1, 0]] * [image.shape[2], image.shape[1]] + [5, 15]).astype(np.int16)
            output_im = cv2.putText(output_im, "%s (%0.1f%%)" % (ktags[k]['label'], score[k] * 100), pos, cv2.FONT_HERSHEY_PLAIN, 1, colors[k], 1)

            result = {}
            result["x1"] = int(boundary_boxes[k][1] * image.shape[2])
            result["y1"] = int(boundary_boxes[k][0] * image.shape[1])
            result["x2"] = int(boundary_boxes[k][3] * image.shape[2])
            result["y2"] = int(boundary_boxes[k][2] * image.shape[1])
            result["w"] = image.shape[2]
            result["h"] = image.shape[1]
            result["tag"] = ktags[k]['label']
            result["color"] = ktags[k]['color']
            result["score"] = score[k]
            prediction_result.append(result)
        im = tf.keras.preprocessing.image.array_to_img(output_im)
        original_im = tf.keras.preprocessing.image.array_to_img(image[0])
        return im, original_im, prediction_result

def read_image(fp):
    raw_image = tf.io.read_file(fp)
    if os.path.splitext(fp)[-1].lower() == ".bmp":
        raw_image = tf.image.decode_bmp(raw_image, channels=3)
    else:
        raw_image = tf.image.decode_jpeg(raw_image, channels=3)
    image = tf.expand_dims(raw_image,0)
    return image
        
class anchor:
    def __init__(self, anchor_level = 2, crop_size = (128, 128)):
        self.anchor_level = anchor_level
        self.prepare_box(anchor_level)
        if type(crop_size) in (list, tuple):
            self.crop_size = crop_size
        elif type(crop_size) == int:
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = (128, 128)
            
    def prepare_box(self, level = 2, ratio = 1.0):
        if ratio > 1:
            x = (1 / ratio) / 2
            y = 0.5
        else:
            x = 0.5
            y = (1 * ratio) / 2
        boxes = tf.constant([[0.5 - x, 0.5 - y, 0.5 + x , 0.5 + y]])
        b = []
        for i in range(level + 1):
            bx = tf.constant(self.generate_box(i, ratio))
            b.append(bx)
        b = tf.concat(b, 0)
        self.boxes = np.unique(tf.concat([boxes, b], 0), axis = 0)
        self.box_indices = tf.zeros(shape=(self.boxes.shape[0],), dtype = "int32")
        
    def generate_box(self, anc_lvl, ratio, border = 0.01):
        anc_lvl += 1
        box = []
        if ratio > 1:
            STANDARD_X = 100 * ratio * anc_lvl
            STANDARD_Y = 100 * anc_lvl
        elif ratio < 1:
            STANDARD_X = 100 * anc_lvl
            STANDARD_Y = 100 / ratio * anc_lvl
        else:
            STANDARD_X = 100 * anc_lvl
            STANDARD_Y = 100 * anc_lvl
        DIVIDE_V = 100
        CELL_Y = math.ceil(STANDARD_Y / DIVIDE_V)
        CELL_X = math.ceil(STANDARD_X / DIVIDE_V)
        if CELL_X - 1 > 0:
            X_OVERFLOW_SUBTR = (((CELL_X * DIVIDE_V) / STANDARD_X) - 1) / (CELL_X - 1)
        else:
            X_OVERFLOW_SUBTR = 0
        if CELL_Y - 1 > 0:
            Y_OVERFLOW_SUBTR = (((CELL_Y * DIVIDE_V) / STANDARD_Y) - 1) / (CELL_Y - 1)
        else:
            Y_OVERFLOW_SUBTR = 0
        FIRST_BLOCK_X = DIVIDE_V / STANDARD_X
        FIRST_BLOCK_Y = DIVIDE_V / STANDARD_Y
        for x in range(CELL_X):
            if x == 0:
                x1 = 0
                x2 = FIRST_BLOCK_X
            else:
                x2 = x1 - X_OVERFLOW_SUBTR + FIRST_BLOCK_X
            for y in range(CELL_Y):
                if y == 0:
                    y1 = 0
                    y2 = FIRST_BLOCK_Y
                else:
                    y2 = y1 - Y_OVERFLOW_SUBTR + FIRST_BLOCK_Y
                block_box = [x1, y1, x2, y2]
                box.append(block_box)
                y1 = y2
            x1 = x2
                
                
        
##        if ratio >= 1:
##            npbox = np.zeros((round(anc_lvl * ratio), anc_lvl))
##        if ratio < 1:
##            npbox = np.zeros((anc_lvl, round(anc_lvl / ratio)))
##        box = []
##        box_expanded_size = 1.5
##        if ratio >= 1.0:
##            hs = anc_lvl + 1
##            ws = math.ceil(ratio + anc_lvl)
##            wr = 1 / ratio
##            hr = 1
##        else:
##            if ratio != 0:
##                anr = anc_lvl + 1 / ratio
##            else:
##                anr = 1
##            hs = math.ceil(anr)
##            ws = anc_lvl + 1
##            wr = 1
##            hr = 1 * ratio
##        boxshape = np.zeros([hs, ws])
##        if anc_lvl > 0:
##            wr = (box_expanded_size * wr) / (anc_lvl + 1)
##            hr = (box_expanded_size * hr) / (anc_lvl + 1)
##        else:
##            wr = wr / (anc_lvl + 1)
##            hr = hr / (anc_lvl + 1)
##        for h in range(hs):
##            for w in range(ws):
##                mw = wr / 2
##                mh = hr / 2
##                pw = ((w / ws) + ((w + 1) / ws)) / 2
##                ph = ((h / hs) + ((h + 1) / hs)) / 2
##                x1 = pw - mw
##                x2 = pw + mw
##                if x1 < 0:
##                    x2 = x2 - x1
##                    x1 = 0.0
##                if x2 > 1:
##                    x1 = x1 - (x2 - 1)
##                    x2 = 1.0
##                y1 = ph - mh
##                y2 = ph + mh
##                if y1 < 0:
##                    y2 = y2 - y1
##                    y1 = 0.0
##                if y2 > 1:
##                    y1 = y1 - (y2 - 1)
##                    y2 = 1.0
##                box.append([x1, y1, x2, y2]) 
        return box

    def make(self, image, bounding_box_with_class = [[]], overlap_requirement = 0.9, max_output_box_nobb = 32, segment_min_ratio = 0.75):
        b, h, w, c = image.shape
        ratio = h / w
        self.prepare_box(self.anchor_level, ratio)
        MASK = np.all([(self.boxes[:, 2] - self.boxes[:, 0]) * h > self.crop_size[1] * segment_min_ratio,  (self.boxes[:, 3] - self.boxes[:, 1]) * w > self.crop_size[0] * segment_min_ratio], 0)
        BOXES = self.boxes[MASK]
        BOX_INDICES = self.box_indices[MASK]
        if not BOXES.shape[0]:
            BOXES = self.boxes[:9]
            BOX_INDICES = self.box_indices[: 9]
        #print("anchor.make", BOXES.shape)
        self.boxes = BOXES
        crop_iter = iter(zip(BOXES, BOX_INDICES))
        BB_NDIMS_CHECK = np.ndim(bounding_box_with_class) == 3 and BOXES.shape[0] == BOX_INDICES.shape[0] and BOXES.shape[0] > 0
        # ADD IN REMOVE BOXES WHERE IMAGE WILL BE HALF OF THE OUTPUT SIZE
        
        if BB_NDIMS_CHECK:
            bb_set_X = bounding_box_with_class[:,:,2] - bounding_box_with_class[:,:,0]
            bb_set_Y = bounding_box_with_class[:,:,3] - bounding_box_with_class[:,:,1]
        else:
            bb_set_X = False
            bb_set_Y = False
            #RETURN AS ITERATOR BECAUSE ITS SHIT LOAD OF MEMEORY!!
            for i in range(0, BOXES.shape[0], max_output_box_nobb):
                v0 = i 
                v1 = i + max_output_box_nobb
                image_outputs = tf.image.crop_and_resize(image, BOXES[v0 : v1, :], BOX_INDICES[v0 : v1], self.crop_size)
                bounding_box = np.ndarray([image_outputs.shape[0],0, 5])
                yield image_outputs, bounding_box

        if BB_NDIMS_CHECK:
            for index, box_value in enumerate(crop_iter):
                box, box_indice = box_value
                box = np.expand_dims(box, 0)
                box_indice = np.expand_dims(box_indice, 0)

                try:
                    image_outputs = tf.image.crop_and_resize(image, box, box_indice, self.crop_size)
                except Exception as e:
                    print("anchor.make.E769", e)
                    continue
                
                if image_outputs.ndim != 4:
                    del image_outputs
                    continue

                for image_output in image_outputs:
                    box_tf = BOXES[index]
                    bounding_box = np.ndarray([1,0, 5])
                    bounding_box_concat = np.ndarray([1,0, 5])
                    y1, x1, y2, x2 = box_tf * [h, w, h, w]
                    for vott_bbc in bounding_box_with_class[0]:
                        x3, y3, x4, y4, tag_class = vott_bbc
                        boxA = [x1, y1, x2, y2]
                        boxB = [x3, y3, x4, y4]
                        try:
                            intersect = bb_intersection(boxA, boxB)
                        except Exception as e:
                            print(boxA, boxB)
                            print(bounding_box_with_class)
                            raise e
                        if intersect >= overlap_requirement:
                            bx1 = max(x1, x3) - x1
                            by1 = max(y1, y3) - y1
                            bx2 = min(x2, x4) - x1
                            by2 = min(y2, y4) - y1
                            ow = image_output.shape[1]
                            oh = image_output.shape[0]
                            if x2 - x1 > 0 and y2 - y1 > 0:
                                owx21 = (ow / (x2 - x1))
                                ohy21 = (oh / (y2 - y1))
                                rx1 = math.floor(bx1 *  owx21)
                                rx2 = math.floor(bx2 *  owx21)
                                ry1 = math.floor(by1 *  ohy21)
                                ry2 = math.floor(by2 *  ohy21)
                                bounding_box_concat = np.append(bounding_box_concat, np.array([[[rx1, ry1, rx2, ry2, tag_class]]], dtype = np.float16), axis = 1)
                    bounding_box = np.append(bounding_box, bounding_box_concat, axis = 1)
                    image_output = tf.expand_dims(image_output, 0)
                    image_output = image_output.numpy()
                    if image_output.ndim == 4 and bounding_box.ndim == 3:
                        if image_output.shape[1:] == (self.crop_size[0], self.crop_size[1], c) and bounding_box.shape[2] == 5:
                            yield image_output, bounding_box

def augment(im, bbwc, seq = None):
    im = im[0]
    bbs = BoundingBoxesOnImage([BoundingBox(x1 = n[0], y1 = n[1], x2 = n[2], y2 = n[3], label = n[4]) for n in bbwc[0]], shape = im.shape[:2])
    iaaseq = [iaa.AverageBlur(k=(5, 5)),\
              iaa.AdditiveGaussianNoise(scale=10), \
              iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}), \
              iaa.Affine(scale=(0.8, 1.2)), \
              iaa.Affine(rotate=(-45, 45)), \
              iaa.Affine(shear=(-16, 16)), \
              iaa.Fliplr(0.5), \
              iaa.Flipud(0.5)]
    if not seq:
        seq = iaa.SomeOf(2,iaaseq)
    else:
        b = "0000000000000" + bin(seq)[2:]
        s = []
        for i in range(0, len(iaaseq)):
            if b[len(b) - len(iaaseq):][i] == "1":
                s.append(iaaseq[i])
        l = 2
        if len(s) < 2:
            l = len(s)
        seq = iaa.SomeOf(l, s)
    images_aug, bbs_aug_wh = seq(image = im, bounding_boxes = bbs)
    return images_aug, bbs_aug_wh


def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou

def bb_intersection(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    if min(boxAArea, boxBArea) != 0:
        i = interArea / min(boxAArea, boxBArea)
        return i
    return 0.0

def sanity_check(gen):
    x,(y,z) = gen.__next__()
    if np.max(x) <= 1.0:
        x = x * 255
    if x.shape[-1] == 1:
        x = tf.convert_to_tensor(x, tf.int16)
        x = tf.image.grayscale_to_rgb(x, name=None)
        x = x.numpy()
    colors = [[255,0,0],]
    imbb = tf.image.draw_bounding_boxes(x, z, colors)
    items = imbb.shape[0]
    w = int(math.ceil(math.sqrt(items)))
    h = math.ceil(items / w)
    fig, ax = plt.subplots(h , w)
    for i, im in enumerate(imbb):
        row = i // w
        col = i % w
        img = tf.keras.preprocessing.image.array_to_img(im)
        title = ", ".join([str(n[0]) for n in y[i].tolist()])
        if h == 1 and w == 1:
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontdict= {'fontsize': 6})
        else:
            ax[row, col].imshow(img)
            ax[row, col].axis('off')
            ax[row, col].set_title(title, fontdict= {'fontsize': 6})
    plt.show()
    plt.clf()
    plt.close()


def load_model_by_pik(f = "tkpik.pik"):
    if not os.path.isfile(f):
        print("no pik")
        return False
    else:
        with open(f, "rb") as fio:
            tkpik = pickle.load(fio)
            model = load_model(tkpik['input_size'], tkpik['color_channel'], tkpik['tags'], tkpik['region'], tkpik['dropout'], tkpik['fpn_mode'], tkpik['backbone'], tkpik['votts'], tkpik['augment'])
            model.SAVENAME = tkpik['savefile']
            model.OVERLAP_REQUIREMENT = tkpik['overlap']
            model.ANCHOR_LEVEL = tkpik['anchor']
            model.PORT = tkpik["port"]
            model.NORMALIZATION = tkpik["normalization"]
            model.RANDOM_DROP = tkpik["random_drop"]
            model.NULL_SKIP = tkpik["nullskip"]
            model.initialize(do_not_auto_save_tags = True)
            return model
        
class load_model:
    def __init__(self, IMAGE_SIZE, COLOR_CHANNEL, CLASSIFICATION_TAGS, REGIONS, DROPOUT, FPN_MODE, BACKBONE, VOTT_PATHS, AUGMENT = 255):
        self.IMAGE_SIZE = IMAGE_SIZE
        self.COLOR_CHANNEL = COLOR_CHANNEL
        self.CLASSIFICATION_TAGS = CLASSIFICATION_TAGS
        self.REGIONS = REGIONS
        self.DROPOUT = DROPOUT
        self.FPN_MODE = FPN_MODE
        self.BACKBONE = BACKBONE
        self.VOTT_PATHS = VOTT_PATHS

        self.BATCH_SIZE = 8
        self.HUBER = 10
        self.TRAIN_SIZE = 0.7
        self.ANCHOR_LEVEL = 2
        self.NULL_SKIP = 0.3
        self.OVERLAP_REQUIREMENT = 1.0
        self.AUGMENT = AUGMENT
        self.SAVENAME = "chinkosu"
        self.NON_MAX_SUPPRESSION_IOU = 0.01
        self.NORMALIZATION = False
        self.RANDOM_DROP = 0.2

    def initialize(self, do_not_auto_save_tags = False):
        print("initialize model")
        self.model = detection_model(image_shape=(self.IMAGE_SIZE, self.IMAGE_SIZE, self.COLOR_CHANNEL), \
                                  detection_region = self.REGIONS, \
                                  classes = self.CLASSIFICATION_TAGS, \
                                  backbone = self.BACKBONE,
                                  dropout = self.DROPOUT)
        b, h, w, c = self.model.m.input.shape

        self.model_compiled_height = h
        self.model_compiled_width = w
        self.model_compiled_channel = c

        self.model.anchor_size = int(self.ANCHOR_LEVEL)
        self.vott_available_paths = [n for n in self.VOTT_PATHS if os.path.isfile(n)]
        
        print(self.vott_available_paths)
        for fp in self.vott_available_paths:
            self.model.load_vott(fp)
        self.model.c.do_not_auto_save_tags = do_not_auto_save_tags
        self.model.prepare(self.SAVENAME + ".pik", self.TRAIN_SIZE)
        self.action = None

    def predict(self, fp, show = True, rawdata = False, debug = False, nms_iou = None, anchor_size = None, image_size = None):
        #print("DEBUG 1029", nms_iou, anchor_size, image_size)
        self.action = "predict"
        if not nms_iou:
            nms_iou = self.NON_MAX_SUPPRESSION_IOU
        im, original_im, result_rawdata = self.model.predict(fp, \
                                                             nms_iou = nms_iou, \
                                                             anchor_size = anchor_size, \
                                                             image_size = image_size)
        if show:
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            plt.clf()
            plt.close()
            #im.show()
        if rawdata:
            return im, result_rawdata
        else:
            return im

    def train(self, EPOCHS = 200, STEPS = 50, LR = 0.001, early_stopping = False, no_validation = False, save_on_end = False):
        self.action = "train"
        if no_validation or self.TRAIN_SIZE == 1 or self.TRAIN_SIZE == 0:
            self.model.prepare(self.SAVENAME + ".pik", 1)
        else:
            self.model.prepare(self.SAVENAME + ".pik", self.TRAIN_SIZE)
        self.model.train(learning_rate = LR, \
                         epoch = EPOCHS, \
                         steps = STEPS, \
                         batch_size = self.BATCH_SIZE, \
                         skip_null = self.NULL_SKIP, \
                         augment_seq = self.AUGMENT, \
                         callback_earlystop = early_stopping)
        if save_on_end:
            self.model.save(self.SAVENAME)
        return True

    def cpu_train(self, EPOCHS = 200, STEPS = 50, LR = 0.001, early_stopping = False, no_validation = False, save_on_end = False):
        self.model.train(EPOCHS, STEPS, LR, early_stopping, no_validation , save_on_end)

    def show_training_result(self, history):
        self.model.chart()

    def segment_image(self):
        for fp in self.vott_available_paths:
            self.model.load_vott(fp)
        self.model.prepare(self.SAVENAME + ".pik", 1.0)
        self.model.c.save_as_segment_image(self.SAVENAME, \
                                           image_shape = (128, 128, 3), \
                                           anchor_size = int(self.ANCHOR_LEVEL), \
                                           tagfile = self.SAVENAME + ".pik")

    def save(self, f = None):
        self.action = "save"
        if not f:
            f = self.SAVENAME
        colors = [[255,0,0],]
        self.model.save(f)
        self.model.c.save_tags(self.SAVENAME + ".pik")
        print("SAVED WEIGHTS")

    def load(self, f = None):
        self.action = "load"
        if not f:
            f = self.SAVENAME
        try:
            self.model.load(f)
            print("LOADED WEIGHTS")
        except Exception as e:
            print(e)

    def generator_check(self, batch_size = 36, skip_null = 0.7, seq = None):
        self.model.sanity_check(batch_size, skip_null, seq)

    def sanity_check(self):
        self.model.sanity_check()

    def folder_check(self, folder):
        #images = filter(lambda f : os.path.splitext(f)[1] in (".jpg", ".png", ".bmp") and os.path.isfile(os.path.join(folder, f)), os.listdir(folder))
        #images = [n for n in images]
        images = [n.path for n in os.scandir(folder) if os.path.splitext(n.name)[1] in (".jpg", ".png", ".bmp") and n.is_file()]
        max_len = 36
        len_images = len(images)
        if len_images < 1:
            return False
        if len_images > max_len:
            len_images = max_len
        sqr_grid = math.ceil(math.sqrt(len_images))
        fig,ax = plt.subplots(sqr_grid , sqr_grid)
        for i, fn in enumerate(images[: len_images]):
            row = i // sqr_grid
            col = i % sqr_grid
            fp = os.path.join(folder, fn)
            im = self.predict(fp, False, False)
            ax[row, col].set_title(fn, fontdict= {'fontsize': 6})
            ax[row, col].imshow(im)
            ax[row, col].axis('off')
        fig.suptitle(os.path.split(folder)[-1], fontsize=11)
        plt.show()
        plt.clf()
        plt.close()

    def trial(self, testfolder):
        for f in os.listdir(testfolder):
            fp = os.path.join(testfolder, f)
            if os.path.isdir(fp):
                self.folder_check(fp)

def from_hex(h):
    s = h.strip("#")
    return [int(s[:2],16), int(s[2:4], 16), int(s[4:], 16)]

def test_train():
    e = detection_model(image_shape=(96,96,3), detection_region = 2, classes = 1000, backbone = "B0")
    e.load_folder(r"C:\Users\CSIPIG0140\Desktop\TRAIN IMAGE\DETECTOR_ALL\DETECTOR_ALL", "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/DETECTOR_ALL/DETECTOR_ALL.pik", 0.7)
    e.train(0.01, 50, 20, 32)
    return e

def save_segmented_image():
    l = loader()
    l.load_from_csv()
    l.save_as_segment_image(r"C:\test\fractured_img", image_shape = (128, 128, 3), anchor_size = 4, tagfile = "C:/test/tags.pik")

def test():
    m = load_model_by_pik()
    return m

def debugnew():
    l = loader()
    l.load_from_csv()
    l.load_from_vott()
    l.prepare()
    v = l.one_file(l.train)
    anc = anchor(2, (96,96))
    f, x, y = next(v)
    a = anc.make(x, y)
    i, b = next(a)
    l.save_fexif("c:/test/exiv.jpg", i[0], b[0])
    ii, bb = l.load_exif("c:/test/exiv.jpg")

def hex2rgb(s):
    if len(s) == 7:
        return [int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)]
    return (0, 0, 0)

import os
import copy
import cv2
import datetime
#import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
import importlib.util
import json
import math
#import matplotlib.colors as mpcolors
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
from tensorflow.keras.optimizers import SGD
#import tensorflow_addons as tfa
import urllib


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


NAN_REPLACEMENT = "-"

##def preprocess_funcxx(im):
##    if im.ndim == 4:
##        imshape = im.shape
##        im = im.reshape([imshape[0], imshape[1] * imshape[2] * imshape[3]])
##        immax = np.expand_dims(np.max(im, 1), 1)
##        immin = np.expand_dims(np.min(im, 1), 1)
##        im = (((im - immin) / (immax - immin + 1e-6)) * 255).astype(np.int16)
##        return im.reshape(imshape)
##    return  ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6) * 255).astype(np.int16)

def preprocess_func(im):
    return  ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6) * 255).astype(np.int16)


def normalize_func(image):
    return tf.cast(image, tf.float32) / 255


class loader:
    def __init__(self):
        self.df = pd.DataFrame(columns = ["category", "source", "path", "x1", "y1", "x2", "y2", "label", "color", "segmented"])
        self.df_label = pd.DataFrame(columns = ['category', 'label', 'color'])
        self.tagfile = "dummy.csv"
        self.COLUMN = ["category", "source", "path", "x1", "y1", "x2", "y2", "label", "color", "segmented"]


    def load_from_csv(self, fp = "LABEL_FILE.csv", category = None):
        column_names = ['folder', 'filename', 'timeseek', 'path', 'label', 'x1', 'y1', 'x2', 'y2', 'color', 'modified_dt']
        df = pd.read_csv(fp, delimiter = ",", names = column_names)
        if not category:
            df["category"] = df["folder"]
        else:
            df["category"] = category
        df["source"] = fp
        df["segmented"] = 0
        df["path"] = df.apply(lambda d : os.path.join(os.path.split(d.source)[0], d.path), axis = 1)
        self.df = pd.concat([self.df, df[self.COLUMN]], ignore_index = True)
        self.df.drop_duplicates(inplace=True, keep='last')


    def load_from_vott(self, fp = "VOTT_EXPORT.json", category = None):
        tags = {}
        VOTT_LIST = []
        if category:
            pass
        elif not category:
            if os.path.isfile(fp):
                category_file = os.path.join(os.path.split(fp)[0], "category.txt")
                if os.path.isfile(category_file):
                    with open(category_file, "r") as fio:
                        category = fio.readline().strip()
                        print(fp, " >>> ", category)
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
                    data["category"] = category
                    data["source"] = fp
                    data["path"] = path
                    data["label"] = tag
                    data["x1"] = r['boundingBox']['left']
                    data["y1"] = r['boundingBox']['top']
                    data["x2"] = data["x1"] + r['boundingBox']['width']
                    data["y2"] = data["y1"] + r['boundingBox']['height']
                    data["color"] = tags[tag]
                    data["segmented"] = 0
                    VOTT_LIST.append(data)
        self.df = pd.concat([self.df, pd.DataFrame.from_records(VOTT_LIST)], ignore_index = True)
        self.df.drop_duplicates(inplace=True, keep='last')


    def load_from_exif(self, fol = "FOLDER"):
        fs = os.scandir(fol)
        fs = [n.path for n in fs if os.path.splitext(n.name)[1] == ".jpg" and n.is_file()]
        ls = []
        for fp in fs:
            d = {}
            d["category"] = None
            d["source"] = fol
            d["path"] = fp
            d["label"] = None
            d["x1"] = None
            d["y1"] = None
            d["x2"] = None
            d["y2"] = None
            d["color"] = None
            d["segmented"] = 1
            ls.append(d)
        self.df = pd.concat([self.df, pd.DataFrame.from_records(ls)], ignore_index = True)
        self.df.drop_duplicates(inplace=True, keep='last')


    def prepare_label(self, tagfile = None, save_file = True):
        column_names = ['category', 'label', 'color']
        if not tagfile:
            tagfile = ""
        if os.path.isfile(tagfile):
            self.tagfile = tagfile
            file_df = pd.read_csv(tagfile, delimiter = ",", header = 0, names = column_names)
            file_df.drop_duplicates(["category","label"], inplace=True, keep='last')
            file_df.reset_index(inplace=True)
        else:
            file_df = pd.DataFrame(columns = column_names)
        loaded_g = self.df.groupby(["category","label"], dropna=False)
        loaded_df = loaded_g.first().reset_index()[column_names]
        df = pd.concat([file_df, loaded_df], ignore_index = True)
        df.drop_duplicates(["category","label"], inplace=True, keep='last')
        df.reset_index(inplace=True)
        #save function here maybe -----
        df.fillna(value = NAN_REPLACEMENT, inplace=True)
        df = df[column_names]
        if save_file:
            df.to_csv(tagfile)
        #-----------------------
        self.df_label = df


    def frac(self, frac = 0.7):
        g = self.df.groupby("path")
        self.group_indice = g.indices
        li = list(self.group_indice.keys())
        random.shuffle(li)
        self.train = li[: int(len(li) * frac)]
        self.test = li[int(len(li) * frac) : ]        

        
    def fetch_one(self, sampling, dataframe = None, indice = None):
        if not np.any(dataframe):
            dataframe = self.df
        if not np.any(indice):
            indice = self.group_indice
        for k in sampling:
            if not os.path.isfile(k):
                continue
            ONE_FILE = dataframe.iloc[indice[k]]
            IS_SEGMENT = np.all(ONE_FILE[["segmented"]])
            IMAGE_FP = None
            if len(ONE_FILE) == 1 and IS_SEGMENT:
                #it is exif
                RAW_IM, ONE_FILE = self.load_exif(k)
            elif len(ONE_FILE) > 0 and not IS_SEGMENT:
                #it is vott or csv
                RAW_IM = self.load_image(k)
            else:
                continue
            #manage tag labels ----------
            if "category" not in ONE_FILE:
                ONE_FILE["category"] = None
            RAW_DF = ONE_FILE[["category", "x1", "y1", "x2", "y2", "label", "color"]].copy()
            RAW_DF[["category", "label"]] = RAW_DF[["category", "label"]].fillna(value = NAN_REPLACEMENT)
            RAW_DF["tag_indice"] = RAW_DF.apply(lambda x : np.all((self.df_label[["category", "label"]] == x[["category", "label"]]),1).idxmax(), axis = 1, result_type = "reduce")
            RAW_DF["tag_exist"] = RAW_DF.apply(lambda x : np.all(np.any((self.df_label[["category", "label"]] == x[["category", "label"]]),0),0), axis = 1, result_type = "reduce")
            if np.any(RAW_DF.tag_exist == False):
                #print("ADD IN NEW TAG")
                self.df_label = pd.concat([self.df_label, RAW_DF[["category", "label", "color"]][RAW_DF.tag_exist == False]], \
                                          verify_integrity = True, \
                                          ignore_index = True)
                self.df_label.fillna(value = NAN_REPLACEMENT, inplace=True)
                self.df_label.drop_duplicates("label", inplace=True, keep='first')
                self.df_label.reset_index(inplace=True)
                self.df_label = self.df_label[["category", "label", "color"]]
                try:
                    with open(self.tagfile, "wb") as fio:
                        self.df_label.to_csv(self.tagfile)
                except Exception as e:
                    print("FAILED TO SAVE TAG", e)
                RAW_DF["tag_indice"] = RAW_DF.apply(lambda x : np.all((self.df_label[["category", "label"]] == x[["category", "label"]]),1).idxmax(), axis = 1, result_type = "reduce")        
            #--------------------------
            #use only RAW_IM, RAW_DF
            #change image from 0.0 ~ 1.0 to pixel size if found
            if len(RAW_DF) > 0:
                if np.max(np.array(RAW_DF[["x2", "y2"]], np.float16) - np.array(RAW_DF[["x1", "y1"]], np.float16)) <= 1.0 and \
                   np.max(np.array(RAW_DF[["x1", "y1", "x2", "y2"]], np.float16)) <= 1.0:
                    RAW_DF[["x1", "y1", "x2", "y2"]] = np.array(RAW_DF[["x1", "y1", "x2", "y2"]], np.float16) * np.array(RAW_IM.shape)[[2,1,2,1]]   
            #---next to anchor grid -----------
            BB_INDICE = np.expand_dims(np.array(RAW_DF[["x1", "y1", "x2", "y2", "tag_indice"]]), 0).astype(np.int16)
            yield k, RAW_IM, BB_INDICE, IS_SEGMENT


    def batch(self, test_mode = False, regions = 5, null_label = 999, batch_size = 16, anchor_level = 2, input_shape = (128, 128, 3), augment_seq = 255, null_ratio = 1, normalize_image = False):
        #mode 0 = train, mode 1 = test
        if test_mode:
            sampling = self.test
        else:
            sampling = self.train
        if len(sampling) == 0:
            raise Exception("No sample to iterate")
        anc = anchor(anchor_level = anchor_level, crop_size = input_shape[:2])
        batch_boundary_box = np.ndarray([0, regions, 4], dtype = np.float16)
        batch_label = np.ndarray([0, regions, 1], dtype = np.int16)
        batch_image = np.ndarray([0, input_shape[0], input_shape[1], input_shape[2]], dtype = np.int16)
        while True:
            random.shuffle(sampling)
            null_sample = [n for n in sampling if n[-6:] == '_N.jpg']
            sampling = [n for n in sampling if n[-6:] != '_N.jpg']
            sampling.extend(null_sample[: int(len(sampling) * null_ratio)])
            random.shuffle(sampling)
            iter_file = self.fetch_one(sampling)
            for fp, RAW_IM, BB_INDICE, IS_SEGMENT in iter_file:
                RAW_IM = preprocess_func(RAW_IM)
                if not IS_SEGMENT:
                    anc_iter = anc.make(RAW_IM, BB_INDICE, null_ratio = null_ratio, fp = fp)
                else:
                    anc_iter = [[RAW_IM, BB_INDICE]]
                for ni, nb in anc_iter:
                    aug_im, aug_bb, aug_lb = self.augment_batch(ni, nb, augment_seq)
                    IMAGE_SHAPE = aug_im.shape
                    if input_shape[-1] == 1:
                        aug_im = tf.image.rgb_to_grayscale(aug_im).numpy()
                    if IMAGE_SHAPE[:2] != input_shape[:2]:
                        #aug_im = tf.image.resize(aug_im, input_shape[:2], method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None).numpy()
                        aug_im = tf.image.resize(aug_im, input_shape[:2], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR, preserve_aspect_ratio=False, antialias=False, name=None).numpy()                        
                    if batch_boundary_box.shape[1] > aug_bb.shape[0]:
                        bb_filler = np.full([batch_boundary_box.shape[1] - aug_bb.shape[0], 4], [0, 0, 1, 1])
                        label_filler = np.full([batch_boundary_box.shape[1] - aug_bb.shape[0], 1], null_label)
                        aug_bb = np.append(aug_bb, bb_filler, axis = 0)
                        aug_lb = np.append(aug_lb, label_filler, axis = 0)
                    elif batch_boundary_box.shape[1] < aug_bb.shape[0]:
                        aug_bb = aug_bb[: batch_boundary_box.shape[1], :]
                        aug_lb = aug_lb[: batch_boundary_box.shape[1], :]  
                    batch_image = np.append(batch_image, np.expand_dims(aug_im, 0), axis = 0)
                    batch_boundary_box = np.append(batch_boundary_box, np.expand_dims(aug_bb, 0), axis = 0)
                    batch_label = np.append(batch_label, np.expand_dims(aug_lb, 0), axis = 0)
                    if batch_boundary_box.shape[0] == batch_size:
                        batch_boundary_box = np.where(batch_boundary_box < 0, 0, batch_boundary_box)
                        batch_boundary_box = np.where(batch_boundary_box > 1, 1, batch_boundary_box)
                        if normalize_image:
                            batch_image = normalize_func(batch_image)
                        else:
                            batch_image = batch_image.astype(np.int16)
                        yield batch_image, (batch_label.astype(np.int16), batch_boundary_box.astype(np.float16))
                        batch_boundary_box = np.ndarray([0, regions, 4], dtype = np.float16)
                        batch_label = np.ndarray([0, regions, 1], dtype = np.int16)
                        batch_image = np.ndarray([0, input_shape[0], input_shape[1], input_shape[2]], dtype = np.int16)

            
    def load_image(self, fp):
        image_io = tf.io.read_file(fp)
        if os.path.splitext(fp)[-1].lower() == ".bmp":
            tf_img = tf.image.decode_bmp(image_io, channels=3)
        else:
            tf_img = tf.image.decode_jpeg(image_io, channels=3)
        return tf.expand_dims(tf_img,0).numpy()


    def load_exif(self, fp):
        im = Image.open(fp)
        e = im.getexif()
        try:
            return np.expand_dims(tf.keras.utils.img_to_array(im), 0), pd.read_json(e[37510])
        except:
            pass
        return np.ndarray([0, 0, 0, 3]), pd.DataFrame(columns = ["category", "source", "path", "x1", "y1", "x2", "y2", "label", "color", "segmented"])


    def save_exif(self, fp, im, bb):
        im = tf.keras.preprocessing.image.array_to_img(im)
        e = im.getexif()
        df = pd.DataFrame(bb[0], columns= ["x1", "y1", "x2", "y2", "label_index"])
        lbc = self.df_label.iloc[df.label_index].reset_index()
        ddf = pd.concat([df, lbc[["label", "color", "category"]]], axis = 1)
        ddf.drop_duplicates(None, inplace=True, keep='last')
        ddf.reset_index(inplace=True)
        e[37510] = ddf.to_json()
        im.save(fp, exif = e)


    def load_tags(self, tagfile):
        self.prepare_label(tagfile, save_file = False)
        
            
    def save_tags(self, tagfile):
        self.prepare_label(tagfile)
        

    def save_as_segment_image(self, extract_path, image_shape = (128, 128, 3), anchor_size = 4, tagfile = None, with_labels_only = False, segment_minimum_ratio = 0.25):
        print("SAVE SEGMENT WITH ANCHOR LV: ", anchor_size)
        anc = anchor(anchor_size, crop_size = image_shape[:2])
        self.prepare_label(tagfile, save_file = False)
        self.frac(1.0)
        not_segmented = self.df.query("segmented == False")
        g = not_segmented.groupby("path")
        indice = g.indices
        li = list(indice.keys())
        files = self.fetch_one(li, not_segmented, indice)
        if not os.path.isdir(extract_path):
            os.mkdir(extract_path)
        for n, f in enumerate(files):
            if n % 100 == 0:
                print("in progress > ", n)
            fp, RAW_IM, BB_INDICE, seg = f
            if seg:
                continue
            anc_iter = anc.make(RAW_IM, BB_INDICE, null_ratio = 1000, segment_minimum_ratio = 0.25)
            for i, b in enumerate(anc_iter):
                ni, nb = b
                if nb.shape[1] > 0 or not with_labels_only:
                    p, fn = os.path.split(fp)
                    f, ext = os.path.splitext(fn)
                    if nb.shape[1] > 0:
                        label_indicator = "A"
                    else:
                        label_indicator = "N"
                    savefile = os.path.join(extract_path, "%s_%03d_%s.jpg" % (f, i, label_indicator))
                    self.save_exif(savefile, ni[0], nb)        
        print("finish segmenting")

        
    def augment_batch(self, im, bb, augment_seq = None):
        aug_im, aug_bbwc = augment(im, bb, augment_seq)
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


class detection_model:
    def __init__(self, image_shape = (64, 64, 3), detection_region = 2, classes = 1000, backbone = "B0", dropout = 0.2, optimizer = "Adam"):
        self.m = det.edet(input_shape = image_shape, \
                          num_classes = classes, \
                          detection_region = detection_region, \
                          dropout = dropout, \
                          backbone = backbone)
        self.m.compile(optimizer = optimizer, \
                       #loss = {'regression': det.regression_loss, 'classification': det.classification_loss})
                       loss = [det.classification_loss, det.regression_loss], \
                       metrics = [[keras.metrics.SparseCategoricalAccuracy()], \
                                  [keras.metrics.IoU(num_classes=2, target_class_ids=[0])]])
        self.c = loader()
        self.null = classes - 1
        self.regions = detection_region
        self.image_shape = image_shape
        self.options = {}


    def load_csv(self, path):
        self.c.load_from_csv(path)


    def load_vott(self, path, category = None):
        self.c.load_from_vott(path, category)


    def load_exif(self, path):
        self.c.load_from_exif(path)


    def prepare(self, tagfile, frac):
        self.c.prepare_label(tagfile, save_file = False)
        self.c.frac(frac)

        
    def segment(self, \
                extract_path, \
                tagfile, \
                image_shape = (128, 128, 3), \
                anchor_size = 4):
        dt_now = datetime.datetime.now()
        self.c.save_as_segment_image(extract_path, \
                                     image_shape = image_shape, \
                                     anchor_size = anchor_size, \
                                     tagfile = tagfile, \
                                     with_labels_only = False, \
                                     segment_minimum_ratio = 0.25)
        print("DONE ", (datetime.datetime.now() - dt_now).total_seconds())

        
    def train(self, \
              learning_rate = 0.001, \
              epoch = 10, \
              steps = 10, \
              train_test_ratio = 0.7, \
              batch_size = 32, \
              anchor_size = 4, \
              null_ratio = 1.0, \
              augment_seq = None, \
              callback_earlystop = False):
        self.m.optimizer.learning_rate = learning_rate
        print("PREPARE TRAIN BATCH")
        self.c.frac(train_test_ratio)
        train = self.c.batch(test_mode = False, \
                             regions = self.regions, \
                             null_label = self.null, \
                             batch_size = batch_size, \
                             anchor_level = anchor_size, \
                             input_shape = self.image_shape, \
                             augment_seq = augment_seq, \
                             null_ratio = null_ratio, \
                             normalize_image = True)
        test = self.c.batch(test_mode = True, \
                            regions = self.regions, \
                            null_label = self.null, \
                            batch_size = batch_size, \
                            anchor_level = anchor_size, \
                            input_shape = self.image_shape, \
                            augment_seq = augment_seq, \
                            null_ratio = null_ratio, \
                            normalize_image = True)
        if train_test_ratio < 1 and train_test_ratio > 0:
            if callback_earlystop:
                callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 20, verbose = 0, mode = "min", restore_best_weights = True)
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
            if callback_earlystop:
                callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 20, verbose = 0, mode = "min", restore_best_weights = True)
                self.history = self.m.fit(train, \
                                          epochs = epoch, \
                                          steps_per_epoch = steps, \
                                          verbose = 2, \
                                          callbacks=[callback])
            else:
                self.history = self.m.fit(train, \
                                          epochs = epoch, \
                                          steps_per_epoch = steps, \
                                          verbose = 2)                
        
    def save(self, f = None):
        if not f:
            f = "detection"
        self.m.save_weights(f)
        tagfile = f + ".csv"
        try:
            self.c.save_tags(tagfile)
        except Exception as e:
            print(e)

            
    def load(self, f = None):
        if not f:
            f = "detection"
        tagfile = f + ".csv"
        try:
            self.c.load_tags(tagfile)
            self.m.load_weights(f)
        except Exception as e:
            print(e)

        
    def sanity_check(self, batch_size = 36, null_ratio = 1.0, anchor_size = 4, augment_seq = None):
        original_option = copy.copy(self.options)
        self.options["batch_size"] = batch_size
        self.options["anchor_size"] = anchor_size
        self.options["null_ratio"] = null_ratio
        self.options["seq"] = augment_seq
        if original_option != self.options:
            if 'sanity_check_sample' in dir(self):
                del(self.sanity_check_sample)
        if 'sanity_check_sample' not in dir(self):            
            self.c.frac(1.0)
            self.sanity_check_sample = self.c.batch(test_mode = False, \
                                                    regions = self.regions, \
                                                    null_label = self.null, \
                                                    batch_size = batch_size, \
                                                    anchor_level = anchor_size, \
                                                    input_shape = self.image_shape, \
                                                    augment_seq = augment_seq, \
                                                    null_ratio = null_ratio)
        x,(y,z) = next(self.sanity_check_sample)
        if np.max(x) <= 1.0:
            x = x * 255
        if x.shape[-1] == 1:
            x = tf.convert_to_tensor(x, tf.int16)
            x = tf.image.grayscale_to_rgb(x, name=None)
            x = x.numpy()
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
            title = ",".join(self.c.df_label.iloc[y[i][np.where(y[i] < self.c.df_label.index.stop)]].label.tolist())
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
        for k in ('accuracy', \
                  'val_accuracy', \
                  'regression_loss', \
                  'val_regression_loss', \
                  'classification_loss', \
                  'val_classification_loss',
                  'classification_sparse_categorical_accuracy', \
                  'regression_io_u', \
                  'val_classification_sparse_categorical_accuracy', \
                  'val_regression_io_u_1'):
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


    def predict(self, \
                fp, \
                input_shape = None, \
                anchor_size = 4, \
                nms_iou = 0.01, \
                segment_minimum_ratio = 0.75, \
                output_size = 480, \
                debug = False):
        model_shape = self.m.input.shape[1:]
        model_channel = model_shape[-1]
        if type(fp) != np.ndarray:
            raw_image = tf.io.read_file(fp)
            if os.path.splitext(fp)[-1].lower() == ".bmp":
                raw_image = tf.image.decode_bmp(raw_image, channels=3)
            else:
                raw_image = tf.image.decode_jpeg(raw_image, channels=3)
            if model_channel == 1:
                image = tf.image.rgb_to_grayscale(raw_image)
            else:
                image = copy.copy(raw_image)
            image = image.numpy()
        else:
            raw_image =  fp
            if model_channel == 1:
                image = tf.image.rgb_to_grayscale(raw_image)
                image = image.numpy()
            else:
                image = copy.copy(raw_image)            
        image = preprocess_func(image)
        image = tf.expand_dims(image,0)
        raw_image = tf.expand_dims(raw_image, 0)
        if not input_shape:
            input_shape = model_shape
        anc = anchor(anchor_size, crop_size = input_shape[:2])
        anchor_gen = anc.make(image, segment_minimum_ratio = segment_minimum_ratio)
        classifier_list = np.array([0])
        box_prediction_list = np.array([0])
        with tf.device("cpu:0"):
            for image_for_predict, bb in anchor_gen:
                image_for_predict = tf.image.resize(image_for_predict, self.m.input.shape[1:3])
                image_for_predict = normalize_func(image_for_predict)
                classifier, box_prediction = self.m.predict(image_for_predict, verbose = 0)
                if classifier_list.any():
                    classifier_list = np.append(classifier_list, classifier, 0)
                    box_prediction_list = np.append(box_prediction_list, box_prediction, 0)
                else:
                    classifier_list = classifier
                    box_prediction_list = box_prediction
                if debug:
                    image_for_predict = image_for_predict * 255
                    debug_grid = tf.image.draw_bounding_boxes(image_for_predict, box_prediction, [[255, 0, 0]])
                    classifier = tf.argmax(classifier, axis = -1).numpy()
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
        inference = (classifier_list, box_prediction_list)
        classifier = tf.argmax(inference[0], axis = -1).numpy()
        arr = np.expand_dims(classifier,-1)
        score = np.take_along_axis(inference[0], arr, 2)[:,:,0]
        score = score[np.where(classifier < self.c.df_label.index.stop)]
        inference_bb = inference[1][np.where(classifier < self.c.df_label.index.stop)]
        anchor_crop = anc.boxes[np.where(classifier < self.c.df_label.index.stop)[0]]
        classifier_index = classifier[np.where(classifier < self.c.df_label.index.stop)]
        tags = self.c.df_label.iloc[classifier[np.where(classifier < self.c.df_label.index.stop)]]
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
            result["category"] = ktags[k]['category']
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
            self.crop_size = np.array((crop_size, crop_size))
        else:
            self.crop_size = np.array((128, 128))

            
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
        return box


    def make(self, image, bounding_box_with_class = [[]], overlap_requirement = 0.9, max_output_box_nobb = 32, segment_minimum_ratio = 0.75, null_ratio = 1, fp = None):
        b, h, w, c = image.shape
        ratio = h / w
        if np.ndim(bounding_box_with_class) == 3:
            inbb_mask = np.all(bounding_box_with_class[:, :,[2,3]] - bounding_box_with_class[:, :,[0,1]] >= 0, 2)            
            bounding_box_with_class = bounding_box_with_class[inbb_mask]
            bounding_box_with_class = np.expand_dims(bounding_box_with_class, 0)
        self.prepare_box(self.anchor_level, ratio)
        BOXES = self.boxes
        if np.ndim(bounding_box_with_class) == 3:
            box_to_pix = BOXES * np.array(image.shape)[[1,2,1,2]]
            input_bb_box = []
            for bb in bounding_box_with_class[0]:
                input_bb_box.append(np.all(np.concatenate((box_to_pix[:,[1,0]] < bb[:2], box_to_pix[:,[3,2]] > bb[2:4]), axis = 1),1))
            bb_hit_mask = np.any(np.concatenate(np.expand_dims(input_bb_box, -1), 1), 1)
            if null_ratio > 0:
                expected_null = min(math.floor(len(bb_hit_mask) * null_ratio), len(bb_hit_mask)) - len(np.where(bb_hit_mask)[0])
                if expected_null > 0:
                    bb_hit_mask[np.random.choice(np.where(bb_hit_mask == False)[0], expected_null, replace = False)] = True
            BOXES = BOXES[bb_hit_mask]
        else:
            #print("IMAGE PREDICTION MODE")
            MASK = np.all((BOXES[:, [2,3]] - BOXES[:, [0,1]]) * np.array((h, w)) > self.crop_size * np.array(segment_minimum_ratio), 1)
            BOXES = BOXES[MASK]
        if not BOXES.shape[0]:
            BOXES = self.boxes[:9]
        self.boxes = BOXES
        crop_iter = BOXES
        BB_NDIMS_CHECK = np.ndim(bounding_box_with_class) == 3 and BOXES.shape[0] > 0
        for i in range(0, BOXES.shape[0], max_output_box_nobb):
            v0 = i 
            v1 = i + max_output_box_nobb
            box_to_crop = BOXES[v0 : v1, :]
            indice_to_crop = tf.zeros(shape=(box_to_crop.shape[0],), dtype = "int32")
            try:
                image_outputs = tf.image.crop_and_resize(image, box_to_crop, indice_to_crop, self.crop_size)
            except Exception as e:
                print("ERROR 857", e)
                continue
        
            if not BB_NDIMS_CHECK:
                bounding_box = np.ndarray([image_outputs.shape[0],0, 5])
                yield image_outputs, bounding_box
            else:
                box_to_pix = box_to_crop * np.array(image.shape)[[1,2,1,2]]
                for index, px in enumerate(box_to_pix):
                    output_bb = np.all(np.concatenate((px[[1,0]] < bounding_box_with_class[:, :, :2], px[[3,2]] > bounding_box_with_class[:, :, 2:4]), axis = -1), -1)
                    res = bounding_box_with_class[output_bb].astype(np.float16)
                    if np.any(res) != True and np.any(res) != False:
                        print("SKIN FAILED NP.ANY CHECK RES")
                        continue
                    if np.any(px) != True and np.any(px) != False:
                        print("SKIN FAILED NP.ANY CHECK PX")
                        continue
                    if np.any(image_outputs[index]) != True and np.any(image_outputs[index]) != False:
                        print("SKIN FAILED NP.ANY CHECK image_outputs[index]")
                        continue
                    if not(res.ndim == 2 and px.ndim == 1 and image_outputs[index].ndim == 3):
                        print("SKIPPER NDIMS FAILED")
                        continue
                    if not(res.shape[-1] == 5 and px.shape[-1] == 4):
                        print("SKIPPOY", res.shape[-1] == 5, px.shape[-1] == 4)
                        continue
                    res[:, :4] = (res[:, : 4] - px[[1,0,1,0]]) / (px[[3,2]] - px[[1,0]])[[0,1,0,1]] * np.array(image_outputs[index].shape)[[1,0,1,0]]
                    yield np.expand_dims(image_outputs[index], 0), np.expand_dims(res, 0).astype(np.int16)


def augment(im, bbwc, augment_seq = None):
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
    if not augment_seq:
        seq = iaa.SomeOf(2,iaaseq)
    else:
        b = "0000000000000" + bin(augment_seq)[2:]
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


def load_model_by_pik(f):
    if not os.path.isfile(f):
        print("no pik")
        return False
    else:
        with open(f, "rb") as fio:
            config = pickle.load(fio)
        model = load_model(config["MODEL_NAME"], \
                           config["IMAGE_SHAPE"], \
                           config["REGIONS"], \
                           config["CLASSES"], \
                           config["DROPOUT"], \
                           config["BACKBONE"], \
                           config["OPTIMIZER"])
        model.ANCHOR_SIZE = config["ANCHOR_SIZE"]
        model.NMS_IOU = config["NMS_IOU"]
        model.SEGMENT_MINIMUM_RATIO = config["SEGMENT_MINIMUM_RATIO"]
        model.OUTPUT_SIZE = config["OUTPUT_SIZE"]
        model.TRAIN_TEST_RATIO = config["TRAIN_TEST_RATIO"]
        model.BATCH_SIZE = config["BATCH_SIZE"]
        model.NULL_RATIO = config["NULL_RATIO"]
        model.AUGMENT_SEQ = config["AUGMENT_SEQ"]
        model.INPUT_PATHS = config["INPUT_PATHS"]
        return model

        
class load_model:
    def __init__(self, MODEL_NAME, IMAGE_SHAPE, REGIONS, CLASSES, DROPOUT, BACKBONE, OPTIMIZER):
        self.MODEL_NAME = MODEL_NAME
        self.IMAGE_SHAPE = IMAGE_SHAPE
        self.REGIONS = REGIONS
        self.CLASSES = CLASSES
        self.DROPOUT = DROPOUT
        self.BACKBONE = BACKBONE
        self.OPTIMIZER = OPTIMIZER
        self.model = detection_model(image_shape = self.IMAGE_SHAPE, \
                                     detection_region = self.REGIONS, \
                                     classes = self.CLASSES, \
                                     backbone = self.BACKBONE, \
                                     dropout = self.DROPOUT, \
                                     optimizer = self.OPTIMIZER)
        self.ANCHOR_SIZE = 4
        self.NMS_IOU = 0.01
        self.SEGMENT_MINIMUM_RATIO = 0.75
        self.OUTPUT_SIZE = 480
        self.TRAIN_TEST_RATIO = 0.7
        self.BATCH_SIZE = 16
        self.NULL_RATIO = 1.0
        self.AUGMENT_SEQ = None
        self.INPUT_PATHS = None

        
    def load_input(self, paths):
        self.INPUT_PATHS = paths
        for f in paths:
            if os.path.isdir(f):
                self.model.load_exif(f)
            elif os.path.isfile(f):
                if os.path.splitext(f)[1].lower() == ".json":
                    self.model.load_vott(f)
                elif os.path.splitext(f)[1].lower() == ".csv":
                    self.model.load_csv(f)
                elif os.path.splitext(f)[1].lower() == ".txt":
                    with open(f, "r") as fio:
                        for bf in fio:
                            l = bf.split(",")
                            if len(l) == 2:
                                category = l[0].strip()
                                fp = l[1].strip()
                                if os.path.splitext(fp)[1].lower() == ".json":
                                    self.model.load_vott(fp, category)
                                elif os.path.splitext(fp)[1].lower() == ".csv":
                                    self.model.load_csv(fp)                     
        tagfile = self.MODEL_NAME + ".csv"
        self.model.prepare(tagfile, 1.0)
        

    def predict(self, fp, show = True, input_shape = None, anchor_size = None, nms_iou = None, segment_minimum_ratio = None, output_size = None, debug = False):
        if anchor_size != None:
            self.ANCHOR_SIZE = anchor_size
        else:
            anchor_size = self.ANCHOR_SIZE
        if nms_iou:
            self.NMS_IOU = nms_iou
        else:
            nms_iou = self.NMS_IOU
        if segment_minimum_ratio:
            self.SEGMENT_MINIMUM_RATIO = segment_minimum_ratio
        else:
            segment_minimum_ratio = self.SEGMENT_MINIMUM_RATIO
        if output_size:
            self.OUTPUT_SIZE = output_size
        else:
            output_size = self.OUTPUT_SIZE
        im, original_im, result_rawdata = self.model.predict(fp, \
                                                             input_shape = input_shape, \
                                                             anchor_size = anchor_size, \
                                                             nms_iou = nms_iou, \
                                                             segment_minimum_ratio = segment_minimum_ratio, \
                                                             output_size = output_size, \
                                                             debug = debug)
        if show:
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            plt.clf()
            plt.close()
        else:
            return im, result_rawdata


    def train(self, learning_rate = 0.01, epoch = 10, steps = 10, train_test_ratio = None, batch_size = None, anchor_size = None, null_ratio = None, augment_seq = None, callback_earlystop = False):
        if anchor_size:
            self.ANCHOR_SIZE = anchor_size
        else:
            anchor_size = self.ANCHOR_SIZE
        if train_test_ratio:
            self.TRAIN_TEST_RATIO = train_test_ratio
        else:
            train_test_ratio = self.TRAIN_TEST_RATIO
        if batch_size:
            self.BATCH_SIZE = batch_size
        else:
            batch_size = self.BATCH_SIZE
        if null_ratio != None:
            self.NULL_RATIO = null_ratio
        else:
            null_ratio = self.NULL_RATIO
        if augment_seq != None:
            self.AUGMENT_SEQ = augment_seq
        else:
            augment_seq = self.AUGMENT_SEQ
        self.model.train(learning_rate = learning_rate, \
                         epoch = epoch, \
                         steps = steps, \
                         train_test_ratio = train_test_ratio, \
                         batch_size = batch_size, \
                         anchor_size = anchor_size, \
                         null_ratio = null_ratio, \
                         augment_seq = augment_seq, \
                         callback_earlystop = callback_earlystop)
        self.model.save(self.MODEL_NAME)


    def chart(self):
        self.model.chart()


    def segment_image(self, anchor_size):
        self.model.segment(self.MODEL_NAME, \
                           tagfile = self.MODEL_NAME + ".csv", \
                           image_shape = (128, 128, 3), \
                           anchor_size = anchor_size)


    def save(self):
        self.model.save(self.MODEL_NAME)
        print("SAVED WEIGHTS")


    def load(self, path = None):
        if not path:
            path = self.MODEL_NAME
        try:
            self.model.load(path)
            print("LOADED WEIGHTS")
        except Exception as e:
            print(e)


    def sanity_check(self, batch_size = None, null_ratio = None, anchor_size = None, augment_seq = None):
        if anchor_size != None:
            self.ANCHOR_SIZE = anchor_size
        else:
            anchor_size = self.ANCHOR_SIZE
        if batch_size:
            self.BATCH_SIZE = batch_size
        else:
            batch_size = self.BATCH_SIZE
        if null_ratio != None:
            self.NULL_RATIO = null_ratio
        else:
            null_ratio = self.NULL_RATIO
        if augment_seq != None:
            self.AUGMENT_SEQ = augment_seq
        else:
            augment_seq = self.AUGMENT_SEQ
        self.model.sanity_check(batch_size = batch_size, \
                                null_ratio = null_ratio, \
                                anchor_size = anchor_size, \
                                augment_seq = augment_seq)


    def folder_check(self, folder, anchor_size = None, nms_iou = None, segment_minimum_ratio = None):
        if anchor_size != None:
            self.ANCHOR_SIZE = anchor_size
        else:
            anchor_size = self.ANCHOR_SIZE
        if nms_iou:
            self.NMS_IOU = nms_iou
        else:
            nms_iou = self.NMS_IOU
        if segment_minimum_ratio:
            self.SEGMENT_MINIMUM_RATIO = segment_minimum_ratio
        else:
            segment_minimum_ratio = self.SEGMENT_MINIMUM_RATIO
        output_size = self.OUTPUT_SIZE
        images = [n.path for n in os.scandir(folder) if os.path.splitext(n.name)[1].lower() in (".jpg", ".png", ".bmp") and n.is_file()]
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
            print(i, "image trial")
            im, result_rawdata = self.predict(fp, \
                              show = False, \
                              input_shape = None, \
                              anchor_size = anchor_size, \
                              nms_iou = nms_iou, \
                              segment_minimum_ratio = segment_minimum_ratio, \
                              output_size = 480, \
                              debug = False)
            ax[row, col].set_title(fn, fontdict= {'fontsize': 6})
            ax[row, col].imshow(im)
            ax[row, col].axis('off')
        fig.suptitle(os.path.split(folder)[-1], fontsize=11)
        plt.show()
        plt.clf()
        plt.close()


    def trial(self, testfolder, anchor_size = None, nms_iou = None, segment_minimum_ratio = None):
        if anchor_size != None:
            self.ANCHOR_SIZE = anchor_size
        else:
            anchor_size = self.ANCHOR_SIZE
        if nms_iou:
            self.NMS_IOU = nms_iou
        else:
            nms_iou = self.NMS_IOU
        if segment_minimum_ratio:
            self.SEGMENT_MINIMUM_RATIO = segment_minimum_ratio
        else:
            segment_minimum_ratio = self.SEGMENT_MINIMUM_RATIO
        for f in os.listdir(testfolder):
            fp = os.path.join(testfolder, f)
            if os.path.isdir(fp):
                self.folder_check(fp, \
                                  anchor_size = anchor_size, \
                                  nms_iou = nms_iou, \
                                  segment_minimum_ratio = segment_minimum_ratio)

                
    def save_config(self):
        config = dict()
        config["MODEL_NAME"] = self.MODEL_NAME
        config["IMAGE_SHAPE"] = self.IMAGE_SHAPE
        config["REGIONS"] = self.REGIONS
        config["CLASSES"] = self.CLASSES
        config["DROPOUT"] = self.DROPOUT
        config["BACKBONE"] = self.BACKBONE
        config["OPTIMIZER"] = self.OPTIMIZER
        config["ANCHOR_SIZE"] = self.ANCHOR_SIZE
        config["NMS_IOU"] = self.NMS_IOU
        config["SEGMENT_MINIMUM_RATIO"] = self.SEGMENT_MINIMUM_RATIO
        config["OUTPUT_SIZE"] = self.OUTPUT_SIZE
        config["TRAIN_TEST_RATIO"] = self.TRAIN_TEST_RATIO
        config["BATCH_SIZE"] = self.BATCH_SIZE
        config["NULL_RATIO"] = self.NULL_RATIO
        config["AUGMENT_SEQ"] = self.AUGMENT_SEQ
        config["INPUT_PATHS"] = self.INPUT_PATHS
        pik_file = self.MODEL_NAME + ".pik"
        with open(pik_file, "wb") as fio:
            pickle.dump(config, fio)
        g = self.model.c.df.groupby(["category","source"])
        l = list(g.groups.keys())
        with open("vott_files.txt", "w") as fio:
            for category, fp in l:
                category = str(category)
                fp = str(fp)
                if os.path.splitext(fp)[-1].lower() == ".json":
                    fio.write(", ".join([category, fp]) + "\n")


def from_hex(h):
    s = h.strip("#")
    return [int(s[:2],16), int(s[2:4], 16), int(s[4:], 16)]


def hex2rgb(s):
    if len(s) == 7:
        return [int(s[1:3], 16), int(s[3:5], 16), int(s[5:7], 16)]
    return (0, 0, 0)


def model():
    #__init__(self, MODEL_NAME, IMAGE_SHAPE, REGIONS, CLASSES, DROPOUT, BACKBONE):
    m = load_model("NAME", (96, 96, 3), 2, 100, 0.2,  "MobileNetV2", "Adam")
    paths = ["C:/Users/CSIPIG0140/Desktop/TF SIMPLE IMG CLASSIFIER/TRAINING/LABEL_FILE.csv", \
             "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/AI_Auto_Cap/output/vott-json-export/OG_auto_capacitance_OK-export.json", \
             "C:/Users/CSIPIG0140/Desktop/HARR_VOTT TK/HARRVOTT_2024f/test", \
             "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_LOADTRACK/202409/output/vott-json-export/ECS_LOADTRACK_PROJECT_202409_good-export.json", \
             "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_LOADTRACK/202409/output/vott-json-export/ECS_LOADTRACK_PROJECT_202409_ng-export.json", \
             "C:/TENSORFLOW/VOTT_CONNECTION/LOADTRACK_2023_07/vott-json-export/LOADTRACK_202307-export.json", \
             "C:/TENSORFLOW/VOTT_HARR_MEECS/202307/2023-09 retrain/VOTT/vott-json-export/RETRAIN-LOADTRACK-2023-09-export.json", \
             "C:/TENSORFLOW/VOTT_HARR_MEECS/202307/fix/vott-json-export/ME_LOADTRACK_FIX_20230801-export.json", \
             "C:/TENSORFLOW/VOTT_HARR_MEECS/VOTT/vott-json-export/ME_LOADTRACK-export.json", \
             "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_LOADTRACK/202410/output/vott-json-export/LOADTRACK_202410_input_good-export.json", \
             "C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_LOADTRACK/202410/output/vott-json-export/LOADTRACK_202410_ng-export.json", \
             "C:/Users/CSIPIG0140/Desktop/HARR_VOTT TK/HARRVOTT_2024f/NAME"]
    m.load_input(paths)
    m.save_config()
    return m
    
    
def test(rate = 0.01, epoch = 100):
    d = detection_model()
    d.load()
    d.load_csv("C:/Users/CSIPIG0140/Desktop/TF SIMPLE IMG CLASSIFIER/TRAINING/LABEL_FILE.csv")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/AI_Auto_Cap/output/vott-json-export/OG_auto_capacitance_OK-export.json")
    
    d.load_exif("C:/Users/CSIPIG0140/Desktop/HARR_VOTT TK/HARRVOTT_2024f/test")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_LOADTRACK/202409/output/vott-json-export/ECS_LOADTRACK_PROJECT_202409_good-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_LOADTRACK/202409/output/vott-json-export/ECS_LOADTRACK_PROJECT_202409_ng-export.json")
    
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/TAPING_PROBE_PIN/202410/output/vott-json-export/TAPING_PROBE_PIN2_202410_ok-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/TAPING_PROBE_PIN/20240826/output/vott-json-export/taping_probe_pin_2hole_rotate_ng-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/TAPING_PROBE_PIN/20240826/output/vott-json-export/taping_pin_probe_2hole_rotate-export.json")
    
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/TRAINING/output/vott-json-export/ECS_ROLLER-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/TRAINING2/output/vott-json-export/ECS_ROLLER_2-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/roller/output/vott-json-export/ECS-ROLLER-3-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/20240829/output/vott-json-export/ecs_roller_rot_ng-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/20240829/output/vott-json-export/ecs_roller_rot-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/202410/output/vott-json-export/ROLLER_202410_dirty-export.json")
    d.load_vott("C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/ECS_ROLLER/202410/output/vott-json-export/ROLLER_202410_ok-export.json")
    
    
    d.prepare("C:/Users/CSIPIG0140/Desktop/HARR_VOTT TK/HARRVOTT_2024f/detection.csv", 0.7)
    d.train(rate, epoch, 20, train_test_ratio = 1.0, null_ratio = 1.0, anchor_size = 5)
    d.save()
    return d
    #d.predict("C:/Users/CSIPIG0140/Desktop/TF SIMPLE IMG CLASSIFIER/TRAINING/puyo world/b08.png", (64,64,3), debug = True)
    #b = d.c.batch()
    #for im,(c, bb) in b:
    #    print(im.shape, c.shape, bb.shape)

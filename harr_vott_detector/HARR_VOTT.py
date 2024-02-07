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

try:
    import efficient_det as det
except Exception as e:
    SPEC_LOADER = os.path.join(os.path.split(__file__)[0], "efficient_det.py")
    spec_name = os.path.splitext(os.path.split(SPEC_LOADER)[-1])[0]

    spec = importlib.util.spec_from_file_location(spec_name, SPEC_LOADER)
    det = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(det)

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


def preprocess_func(im):
    return  ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6) * 255).astype(np.int16)

class vott_loader:
    def __init__(self, paths, color_channel = 3, regions = 5, train_split = 0.7):
        self.regions = regions
        self.color_channel = color_channel
        self.TAGS_FORMAT, self.TAGS, self.ASSETS = self.loader(paths)
        self.TAG_KEY = {i:n for i, n in enumerate(self.TAGS_FORMAT.keys())}
        #https://stackoverflow.com/questions/60926460/can-dictionary-data-split-into-test-and-training-set-randomly
        s = pd.Series(self.ASSETS)
        if train_split < 1.0:
            self.training_data , self.validation_data  = [i.to_dict() for i in train_test_split(s, train_size = train_split, test_size = 0.95 - train_split)]
        else:
            self.training_data = s.to_dict()
            self.validation_data = s.to_dict()

    def loader(self, paths):
        TAGS = []
        TAGS_FORMAT = {}
        ASSETS = {}
        for vott_fp in paths:
            if not os.path.isfile(vott_fp):
                print("not valid file", vott_fp)
                continue
            exported_folder = os.path.split(vott_fp)[0]
            with open(vott_fp,"r") as fio:
                obj = json.load(fio)
            for t in obj["tags"]:
                tagname = t["name"]
                tagcolor = t["color"]
                if tagname not in TAGS:
                    TAGS_FORMAT[len(TAGS)] = {"name": tagname, "color": tagcolor}
                    TAGS.append(tagname)
                    
            for k in list(obj["assets"]):
                asset = obj["assets"][k]
                original_path = urllib.parse.unquote(asset["asset"]["path"].replace("file:","").replace("/","\\"))
                exported_path = os.path.join(exported_folder, asset["asset"]["name"])
                if os.path.isfile(exported_path):
                    used_path = exported_path
                elif os.path.isfile(original_path):
                    used_path = original_path
                else:
                    print("file not found", asset["asset"]["name"])
                    continue
                if len(asset["regions"]) > 0:
                    ASSETS[k] = {}
                    ASSETS[k]["original_path"] = original_path
                    ASSETS[k]["exported_path"] = exported_path
                    ASSETS[k]["path"] = used_path
                    ASSETS[k]["asset"] = asset["asset"]
                    ASSETS[k]["regions"] = asset["regions"]
        return TAGS_FORMAT, TAGS, ASSETS
            
    def prepare_image_region(self, key):
        vott_data = self.ASSETS[key]
        image_width = vott_data["asset"]["size"]["width"]
        image_height = vott_data["asset"]["size"]["height"]
        image_path = vott_data["path"]
        image_io = tf.io.read_file(image_path)
        if not os.path.isfile(image_path):
            print("nofile", image_path)
            return False, False
        if os.path.splitext(image_path)[-1].lower() == ".bmp":
            tf_img = tf.image.decode_bmp(image_io, channels=3)
        else:
            tf_img = tf.image.decode_jpeg(image_io, channels=3)
        if self.color_channel == 1:
            tf_img = tf.image.rgb_to_grayscale(tf_img)
        else:
            pass
        tf_img = tf.expand_dims(tf_img,0).numpy()
        tf_img = preprocess_func(tf_img)
        bounding_box_list = []
        for region in vott_data["regions"]:
            classification = self.TAGS.index(region["tags"][0])
            left = region['boundingBox']["left"]
            top = region['boundingBox']["top"]
            w = region['boundingBox']["width"]
            h = region['boundingBox']["height"]
            x1 = int(left)
            y1 = int(top)
            x2 = int(left + w)
            y2 = int(top + h)
            bounding_box_list.append([x1, y1, x2, y2, classification])
        bounding_box_with_class = np.array([[n[:5] for n in bounding_box_list]], dtype = np.float16)
        return tf_img, bounding_box_with_class

    def batch(self, batch_size = 36, is_validation_set = False, skip_no_bb_chance = 0.7, imported_anchor = None, augment_seq = None, null_class = 99):
        if imported_anchor:
            anc = imported_anchor
        else:
            anc = anchor()
        if is_validation_set:
            asset = self.validation_data
        else:
            asset = self.training_data

        batch_image = np.ndarray([0, anc.crop_size[1], anc.crop_size[0], self.color_channel], dtype = np.int16)
        batch_label = np.ndarray([0, self.regions, 1], dtype = np.int16)
        batch_bbox = np.ndarray([0, self.regions, 4], dtype = np.float16)
        
        while True:                
            for key in asset:
                tf_img, bounding_box_with_class = self.prepare_image_region(key)
                if type(tf_img) == np.ndarray and type(bounding_box_with_class) == np.ndarray:
                    anc_iter = anc.make(tf_img, bounding_box_with_class)                    
                    for anc_im, anc_bbc in anc_iter:
                        if len(anc_bbc) == 0 and random.random() > 1.0 - skip_no_bb_chance:
                            continue

                        aug_im, aug_bb = augment(anc_im, anc_bbc)
                        w = aug_bb.shape[1]
                        h = aug_bb.shape[0]

                        new_array = []
                        for bb in aug_bb:
                            x1 = bb.x1
                            x2 = bb.x2
                            y1 = bb.y1
                            y2 = bb.y2
                            cx = (x1 + x2) // 2
                            cy = (y1 + y2) // 2                            
                            c = math.sqrt(pow(max(w//2, cx) - min(w//2, cx), 2) + pow(max(h//2, cy) - min(h//2, cy), 2))
                            label = int(bb.label)
                            new_array.append([x1, y1, x2, y2, label, c])

                        new_array.sort(key = lambda k : 1e6 if k[4] not in self.TAG_KEY else k[5])
                        new_array = np.array(new_array, np.float16)
                        
                        if new_array.shape[0] == 0:
                            new_array = new_array.reshape((0,6))
                            
                        region_label = np.expand_dims(np.expand_dims(new_array[:, 4], -1), 0)
                        region_bb = np.expand_dims(new_array[:, [1,0,3,2]], 0)
                        
                        NULL_LABEL = np.array([[[null_class]]], np.int16)
                        NULL_BB = np.array([[[0.0, 0.0, h, w]]], np.float16)
                        
                        for i in range(batch_label.shape[1] - region_label.shape[1]):
                            region_label = np.append(region_label, NULL_LABEL, axis = 1)
                            region_bb = np.append(region_bb, NULL_BB, axis = 1)
                            
                        if batch_image.shape[0] >= batch_size:
                            batch_image = np.ndarray([0, anc.crop_size[1], anc.crop_size[0], self.color_channel], dtype = np.int16)
                            batch_label = np.ndarray([0, self.regions, 1], dtype = np.int16)
                            batch_bbox = np.ndarray([0, self.regions, 4], dtype = np.float16)
                        batch_image = np.append(batch_image, np.array([aug_im], dtype = np.int16), axis = 0)
                        batch_label = np.append(batch_label, region_label[:, :self.regions, :], axis = 0)
                        batch_bbox = np.append(batch_bbox, region_bb[:, :self.regions, :] / [h, w, h, w], axis = 0)

                        if batch_image.shape[0] >= batch_size:
                            yield batch_image.astype(np.int16), (batch_label.astype(np.int16), batch_bbox.astype(np.float16))
                        

class anchor:
    def __init__(self, anchor_level = 2, crop_size = (128, 128)):
        self.anchor_level = anchor_level
        self.prepare_box(anchor_level)
        self.crop_size = crop_size

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
            b.append(tf.constant(self.generate_box(i, ratio)))
        b = tf.concat(b, 0)
        self.boxes = tf.concat([boxes, b], 0)
        self.box_indices = tf.zeros(shape=(self.boxes.shape[0],), dtype = "int32")
        
    def generate_box(self, anc_lvl, ratio, border = 0.01):
        box = []
        box_expanded_size = 1.5
        if ratio >= 1.0:
            hs = anc_lvl + 1
            ws = math.ceil(ratio + anc_lvl)
            wr = 1 / ratio
            hr = 1
        else:
            if ratio != 0:
                anr = anc_lvl + 1 / ratio
            else:
                anr = 1
            hs = math.ceil(anr)
            ws = anc_lvl + 1
            wr = 1
            hr = 1 * ratio
        boxshape = np.zeros([hs, ws])
        if anc_lvl > 0:
            wr = (box_expanded_size * wr) / (anc_lvl + 1)
            hr = (box_expanded_size * hr) / (anc_lvl + 1)
        else:
            wr = wr / (anc_lvl + 1)
            hr = hr / (anc_lvl + 1)
        for h in range(hs):
            for w in range(ws):
                mw = wr / 2
                mh = hr / 2
                pw = ((w / ws) + ((w + 1) / ws)) / 2
                ph = ((h / hs) + ((h + 1) / hs)) / 2
                x1 = pw - mw
                x2 = pw + mw
                if x1 < 0:
                    x2 = x2 - x1
                    x1 = 0.0
                if x2 > 1:
                    x1 = x1 - (x2 - 1)
                    x2 = 1.0
                y1 = ph - mh
                y2 = ph + mh
                if y1 < 0:
                    y2 = y2 - y1
                    y1 = 0.0
                if y2 > 1:
                    y1 = y1 - (y2 - 1)
                    y2 = 1.0
                box.append([x1, y1, x2, y2]) 
        return box

    def make(self, image, bounding_box_with_class = [[]], overlap_requirement = 0.9):
        b, h, w, c = image.shape
        ratio = h / w
        self.prepare_box(self.anchor_level, ratio)
        crop_iter = iter(zip(self.boxes, self.box_indices))
        for index, box_value in enumerate(crop_iter):
            box, box_indice = box_value
            box = np.expand_dims(box, 0)
            box_indice = np.expand_dims(box_indice, 0)
            image_outputs = tf.image.crop_and_resize(image, box, box_indice, self.crop_size)
            for image_output in image_outputs:
                try:
                    box_tf = self.boxes[index]
                    bounding_box = np.ndarray([1,0, 5])
                    bounding_box_concat = np.ndarray([1,0, 5])
                    x1 = int(box_tf[1] * w)
                    y1 = int(box_tf[0] * h)
                    x2 = int(box_tf[3] * w)
                    y2 = int(box_tf[2] * h)
                    #print(box_tf[1], x1, box_tf[0], y1, box_tf[3], x2, box_tf[2], y2)
                    for vott_bbc in bounding_box_with_class[0]:
                        x3 = int(vott_bbc[0])
                        y3 = int(vott_bbc[1])
                        x4 = int(vott_bbc[2])
                        y4 = int(vott_bbc[3])
                        boxA = [x1, y1, x2, y2]
                        boxB = [x3, y3, x4, y4]
                        tag_class = int(vott_bbc[4])
                        intersect = bb_intersection(boxA, boxB)
                        if intersect >= overlap_requirement:
                            bx1 = max(x1, x3) - x1
                            by1 = max(y1, y3) - y1
                            bx2 = min(x2, x4) - x1
                            by2 = min(y2, y4) - y1
                            ow = image_output.shape[1]
                            oh = image_output.shape[0]
                            rx1 = math.floor(bx1 *  (ow / (x2 - x1)))
                            rx2 = math.floor(bx2 *  (ow / (x2 - x1)))
                            ry1 = math.floor(by1 *  (oh / (y2 - y1)))
                            ry2 = math.floor(by2 *  (oh / (y2 - y1)))
                            bounding_box_concat = np.append(bounding_box_concat, np.array([[[rx1, ry1, rx2, ry2, tag_class]]], dtype = np.float16), axis = 1)
                        else:
                            pass
                    bounding_box = np.append(bounding_box, bounding_box_concat, axis = 1)
                    image_output = tf.expand_dims(image_output, 0)
                    image_output = image_output.numpy()
                    yield image_output, bounding_box
                except Exception as e:
                    print("E285", e)
                    continue
    
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
	i = interArea / min(boxAArea,boxBArea)
	return i

def sanity_check(gen):
    x,(y,z) = gen.__next__()
    if x.shape[-1] == 1:
        x = tf.convert_to_tensor(x, tf.int16)
        x = tf.image.grayscale_to_rgb(x, name=None)
        x = x.numpy()
    colors = [[255,0,0],]
    imbb = tf.image.draw_bounding_boxes(x, z, colors)
    items = imbb.shape[0]
    h = int(math.ceil(math.sqrt(items)))
    w = int(math.ceil(math.sqrt(items)))
    fig, ax = plt.subplots(h , w)
    for i, im in enumerate(imbb):
        row = i // h
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


def load_ai_by_pik(f = "tkpik.pik"):
    if not os.path.isfile(f):
        print("no pik")
        return False
    else:
        with open(f, "rb") as fio:
            tkpik = pickle.load(fio)
            model = load_model(tkpik['input_size'], tkpik['color_channel'], tkpik['tags'], tkpik['region'], tkpik['dropout'], tkpik['fpn_mode'], tkpik['backbone'], tkpik['votts'], tkpik['augment'], tkpik['lstm'])
            model.SAVENAME = tkpik['savefile']
            model.OVERLAP_REQUIREMENT = tkpik['overlap']
            model.ANCHOR_LEVEL = tkpik['anchor']
            model.PORT = tkpik["port"]
            model.initialize()
            return model
        
class load_model:
    def __init__(self, IMAGE_SIZE, COLOR_CHANNEL, CLASSIFICATION_TAGS, REGIONS, DROPOUT, FPN_MODE, BACKBONE, VOTT_PATHS, AUGMENT = 255, LSTM = False):
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
        self.LSTM = LSTM

    def initialize(self):
        print("initialize model")
        self.model = det.edet(self.IMAGE_SIZE, self.COLOR_CHANNEL, self.CLASSIFICATION_TAGS, self.REGIONS, dropout = self.DROPOUT, bi = self.FPN_MODE, backbone = self.BACKBONE, LSTM = self.LSTM)
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss={'regression': self.iouhuloss, "classification": tf.keras.losses.SparseCategoricalCrossentropy()},)
        b, h, w, c = self.model.input.shape

        self.model_compiled_height = h
        self.model_compiled_width = w
        self.model_compiled_channel = c
        
        self.anchor = anchor(self.ANCHOR_LEVEL, crop_size = (h, w))
        self.vott_available_paths = [n for n in self.VOTT_PATHS if os.path.isfile(n)]

        print(self.vott_available_paths)

        if os.path.isfile(self.SAVENAME + ".pik"):
            with open(self.SAVENAME + ".pik", "rb") as fio:
                self.tag_format = pickle.load(fio)
        else:
            try:
                temp_gen = vott_loader(self.vott_available_paths)
                self.tag_format = temp_gen.TAGS_FORMAT
                #print("SAVE PIK", __file__)
##                with open(self.SAVENAME + ".pik", "wb") as fio:
##                    pickle.dump(temp_gen.TAGS_FORMAT, fio)
            except FileNotFoundError as e:
                print(e)

        self.action = None

    def predict(self, fp, show = True, rawdata = False):
        self.action = "predict"
        tags = [k for k in self.tag_format]
        raw_image = tf.io.read_file(fp)
        model_channel = self.model.input.shape[-1]
        if os.path.splitext(fp)[-1].lower() == ".bmp":
            raw_image = tf.image.decode_bmp(raw_image, channels=3)
        else:
            raw_image = tf.image.decode_jpeg(raw_image, channels=3)
        if model_channel == 1:
            image = tf.image.rgb_to_grayscale(raw_image)
        else:
            image = copy.copy(raw_image)
        
        image = tf.expand_dims(image,0)
        im_shape = image.shape

        raw_image = tf.expand_dims(raw_image, 0)

        image = preprocess_func(image.numpy())

        image_for_predict = np.ndarray([0, self.anchor.crop_size[0], self.anchor.crop_size[1], model_channel], np.int16)
        anchor_gen = self.anchor.make(image)
        for im, bb in anchor_gen:
            image_for_predict = np.append(image_for_predict, im, axis = 0)
        
        inference = self.model.predict(image_for_predict, verbose = 0)

        classifier = tf.argmax(inference[0], axis = -1).numpy()
        arr = np.expand_dims(classifier,-1)
        classifier = classifier.tolist()
        
        score = np.take_along_axis(inference[0], arr, 2)
        score = np.max(score, -1)
        
        bb = inference[1]

        check_within_tags = tf.argmax(inference[0], axis=-1).numpy()

        combined_boundary_boxes = []
        combined_colors = []
        combined_class = []
        color_list = []
        tag_list = []
        for cnt, img in enumerate(image_for_predict):
            within_tags = np.where(check_within_tags[cnt] < len(tags))

            cf = np.take(check_within_tags[cnt], within_tags, axis = 0)
            bf = np.take(bb[cnt], within_tags, axis = 0)

            box_h1 = self.anchor.boxes[cnt][0] * im_shape[1]
            box_h2 = self.anchor.boxes[cnt][2] * im_shape[1]
            box_w1 = self.anchor.boxes[cnt][1] * im_shape[2]
            box_w2 = self.anchor.boxes[cnt][3] * im_shape[2]
            box_hh = box_h2 - box_h1
            box_ww = box_w2 - box_w1

            bounding_boxes = bf 

            bb_actual = bounding_boxes * [box_hh, box_ww, box_hh, box_ww]
            bb_actual = bb_actual + [box_h1, box_w1, box_h1, box_w1]
            bb_actual = bb_actual / [im_shape[1], im_shape[2], im_shape[1], im_shape[2]]

            bb_actual = bb_actual.tolist()
            
            colors = [from_hex(self.tag_format[tags[n]]["color"]) for n in cf[0] if n > -1]

            for i, n in enumerate(cf[0]):
                if n < len(tags):
                    tag = tags[n]
                    tag_score = str(int(score[cnt][i] * 100)) + "%"
                    txt = str(tag) + " " + tag_score
                    tag_list.append({"sequence" : cnt, "score" : score[cnt][i], "box" : bb_actual[0][i], "tag" : tags[n], "class" : n, "text" : txt, "color" : colors[i]})

        if raw_image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(raw_image, name=None)
        else:
            image = raw_image
        image = tf.image.resize(image, (480, 480), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True, antialias=False, name=None)

        boxes = [n["box"] for n in tag_list]
        scores = [n["score"] for n in tag_list]

        result_rawdata = []
        if boxes and scores:
            res = tf.image.non_max_suppression(boxes, scores, 100, self.NON_MAX_SUPPRESSION_IOU)
            new_d = []
            for i in res:
                new_d.append(tag_list[i])

            bounding_boxes = np.array([[n["box"] for n in new_d]])
            combined_colors = np.array([n["color"] for n in new_d])
            output_im = tf.image.draw_bounding_boxes(image.numpy(), bounding_boxes, combined_colors)

            output_im = output_im[0].numpy()
            
            for i in new_d:
                if i["tag"] in self.tag_format:
                    TAGNAME = self.tag_format[i["tag"]]["name"]
                    #debug1 = 0
                else:
                    #debug1 = 1
                    TAGNAME = str(i["tag"])

                #print("PREDICTED TAG", i["tag"], TAGNAME, debug1)
                raw_result = {}
                x1 = int(i["box"][1] * output_im.shape[1])
                y1 = int(i["box"][0] * output_im.shape[0])
                x2 = int(i["box"][3] * output_im.shape[1])
                y2 = int(i["box"][2] * output_im.shape[0])
                cv2.putText(output_im, TAGNAME, (x1 + 1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 1, i["color"], 1)
                cv2.putText(output_im, "%02d%%" % (i["score"] * 100), (x1 + 1, y1 + 18), cv2.FONT_HERSHEY_PLAIN, 0.6, i["color"])

                raw_result["x1"] = x1
                raw_result["y1"] = y1
                raw_result["x2"] = x2
                raw_result["y2"] = y2
                raw_result["w"] = output_im.shape[1]
                raw_result["h"] = output_im.shape[0]
                raw_result["tag"] = TAGNAME
                raw_result["color"] = i["color"]
                raw_result["score"] = i["score"]
                result_rawdata.append(raw_result)
        else:
            output_im = image[0]
        
        im = tf.keras.preprocessing.image.array_to_img(output_im)
        if show:        
            plt.imshow(im)
            plt.axis('off')
            plt.show()
            plt.clf()
            plt.close()

        if rawdata:
            return im, result_rawdata
        else:
            return im

    def iouhuloss(self, y_true, y_pred):
        fl = tf.keras.losses.Huber()
        gl = tfa.losses.GIoULoss()
        f = fl(y_true, y_pred)
        g = gl(y_true, y_pred)
        return g + f * self.HUBER

    def train(self, EPOCHS = 200, STEPS = 50, LR = 0.001, early_stopping = False):
        if not len(self.vott_available_paths):
            print("NO VOTT PATHS AVAILABLE")
            return False

        self.action = "train"
        tf.keras.backend.clear_session()
        gen = vott_loader(self.vott_available_paths, color_channel = self.model_compiled_channel, regions = self.REGIONS, train_split = self.TRAIN_SIZE)

        NULL_VALUE = self.model.output[0].shape[-1] - 1

        train_data = gen.batch(batch_size = self.BATCH_SIZE,
                               is_validation_set = False,
                               skip_no_bb_chance = self.NULL_SKIP,
                               imported_anchor = self.anchor,
                               augment_seq = self.AUGMENT,
                               null_class = NULL_VALUE)
        
        validation_data = gen.batch(batch_size = self.BATCH_SIZE,
                               is_validation_set = True,
                               skip_no_bb_chance = self.NULL_SKIP,
                               imported_anchor = self.anchor,
                               augment_seq = self.AUGMENT,
                               null_class = NULL_VALUE)
        
        self.model.optimizer.learning_rate = LR
        dtnow = datetime.datetime.now()

        if early_stopping:
            callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", patience = 10, verbose = 0, mode = "min", restore_best_weights = True)
            history = self.model.fit(train_data, validation_data = validation_data, epochs = EPOCHS, steps_per_epoch = STEPS, validation_steps = int(STEPS * 0.3), verbose = 2, callbacks=[callback])
        else:
            history = self.model.fit(train_data, validation_data = validation_data, epochs = EPOCHS, steps_per_epoch = STEPS, validation_steps = int(STEPS * 0.3), verbose = 2)
        
        loss = history.history['loss']
        classification_loss = history.history['classification_loss']
        regression_loss = history.history['regression_loss']

        if 'val_loss' in history.history:
            val_loss = history.history['val_loss']
        if 'val_classification_loss' in history.history:
            val_classification_loss = history.history['val_classification_loss']
        if 'val_regression_loss' in history.history:
            val_regression_loss = history.history['val_regression_loss']
        
        epochs_range = range(EPOCHS)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='loss')
        if 'val_loss' in history.history:
            plt.plot(epochs_range, val_loss, label='val_loss')
        plt.legend(loc='lower right')
        plt.title('LOSS')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, classification_loss, label='classification_loss')
        plt.plot(epochs_range, regression_loss, label='regression_loss')
        if 'val_classification_loss' in history.history:
            plt.plot(epochs_range, val_classification_loss, label='val_classification_loss')
        if 'val_regression_loss' in history.history:
            plt.plot(epochs_range, val_regression_loss, label='val_regression_loss')
        plt.legend(loc='upper right')
        plt.title('Classification and regression')

        plt.legend()
        plt.show()
        plt.clf()
        plt.close()

        with open(self.SAVENAME + ".pik", "wb") as fio:
            pickle.dump(gen.TAGS_FORMAT, fio)

    def save(self, f = None):
        self.action = "save"
        if not f:
            f = self.SAVENAME
        colors = [[255,0,0],]
        self.model.save_weights(f)
        print("SAVED WEIGHTS")

    def load(self, f = None):
        self.action = "load"
        if not f:
            f = self.SAVENAME
        try:
            self.model.load_weights(f)
            print("LOADED WEIGHTS")
        except Exception as e:
            print(e)

    def generator_check(self):
        self.sanity_check()

    def sanity_check(self):
        if self.action != "sanity_check":
            NULL_VALUE = self.model.output[0].shape[-1] - 1
            vottgen = vott_loader(self.vott_available_paths, color_channel = self.model_compiled_channel, regions = self.REGIONS, train_split = self.TRAIN_SIZE)
            self.gen = vottgen.batch(batch_size = self.BATCH_SIZE,
                                     is_validation_set = False,
                                     skip_no_bb_chance = self.NULL_SKIP,
                                     imported_anchor = self.anchor,
                                     augment_seq = self.AUGMENT,
                                     null_class = NULL_VALUE)
            self.TAGS_FORMAT = vottgen.TAGS_FORMAT
        
        self.action = "sanity_check"
        x,(y,z) = self.gen.__next__()
        if x.shape[-1] == 1:
            x = tf.convert_to_tensor(x, tf.int16)
            x = tf.image.grayscale_to_rgb(x, name=None)
            x = x.numpy()
        colors = [[255,0,0],]
        imbb = tf.image.draw_bounding_boxes(x, z, colors)
        items = imbb.shape[0]
        h = int(math.ceil(math.sqrt(items)))
        w = int(math.ceil(math.sqrt(items)))
        fig, ax = plt.subplots(h , w)
        for i, im in enumerate(imbb):
            row = i // h
            col = i % w
            img = tf.keras.preprocessing.image.array_to_img(im)
            title = ", ".join([self.TAGS_FORMAT[n[0]]['name'] if n[0] in self.TAGS_FORMAT else str(n[0]) for n in y[i].tolist()])
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

    def folder_check(self, folder):
        images = filter(lambda f : os.path.splitext(f)[1] in (".jpg", ".png", ".bmp") , os.listdir(folder))
        cnt = 0
        fig,ax = plt.subplots(3 , 5)
        for row in ax:
            for col in row:
                col.axis("off")
        while cnt < 15:
            try:
                fn = images.__next__()
            except Exception as e:
                print(e)
                cnt = 100
                break
            row = cnt // 5
            col = cnt % 5
            fp = os.path.join(folder, fn)
            im = self.predict(fp, False, False)
            ax[row, col].set_title(fn, fontdict= {'fontsize': 6})
            ax[row, col].imshow(im)
            ax[row, col].axis('off')
            cnt += 1
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
    model = load_ai_by_pik()
    model.VOTT_PATHS = ['C:/PROJECT/HARR_VOTT/vott-json-export/OCR-export.json',]
    model.initialize()
    model.load()
    model.train(10, 10, 0.001)
    return model

def test_predict():
    model = load_ai_by_pik()
    model.VOTT_PATHS = ['C:/PROJECT/HARR_VOTT/vott-json-export/OCR-export.json',]
    model.initialize()
    model.load()
    return model

def test():
    a = anchor()
    g = vott_loader(['C:/Users/CSIPIG0140/Desktop/TRAIN IMAGE/TAPING_PROBE_PIN/output/vott-json-export/TAPING-PROBE-PIN-export.json',])
    b = g.batch()
    sanity_check(b)

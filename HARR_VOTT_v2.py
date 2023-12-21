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
from tensorflow.keras.optimizers.legacy import Adam
import tensorflow_addons as tfa
import urllib
import efficient_det_LSTM as det



#AI SPEC---------------------------------------------------------------------------
##SPEC_LOADER = "efficient_det_S.py"
##spec_name = os.path.splitext(os.path.split(SPEC_LOADER)[-1])[0]
##spec = importlib.util.spec_from_file_location(spec_name, SPEC_LOADER)
##det = importlib.util.module_from_spec(spec)
##spec.loader.exec_module(det)

SAVENAME = "noname"

VOTT3 = r"C:\TENSORFLOW\VOTT_HARR_MEECS\202307\fix\vott-json-export\ME_LOADTRACK_FIX_20230801-export.json"
VOTT2 = r"C:\TENSORFLOW\VOTT_HARR_MEECS\VOTT\vott-json-export\ME_LOADTRACK-export.json"
VOTT = r"C:\TENSORFLOW\VOTT_CONNECTION\LOADTRACK_2023_07\vott-json-export\LOADTRACK_202307-export.json"
VOTT4 = r"C:\TENSORFLOW\VOTT_HARR_MEECS\202307\hole_fix_202308\vott-json-export\ME_LOADTRACK_HOLE_FIX_202308-export.json"
VOTT5 = r"C:\TENSORFLOW\VOTT_HARR_MEECS\202307\2023-09 retrain\VOTT\vott-json-export\RETRAIN-LOADTRACK-2023-09-export.json"
VOTT6 = r"C:\TENSORFLOW\VOTT_HARR_MEECS\202307\LT_202310_RETRAIN\output\vott-json-export\LT_202310_RETRAIN-export.json"

VOTTS = [VOTT, VOTT2, VOTT3, VOTT4, VOTT5, VOTT6]

TESTFOLDER = r"C:\Users\CSIPIG0140\Desktop\TEST"

#main training parameter ------------------------------------------------------------
ANCHOR_LEVEL = 2
overlap_for_true = 0.9
REGIONS = 1
CLASSIFICATION_TAGS = 10
IMAGE_SIZE = 256
COLOR_CHANNEL = 1
NULL_SKIP = 0.8
DEBUG = False
BACKBONE = "B0"
FPN_MODE = 0
DROPOUT = 0.1


BATCH_SIZE = 4
EPSILON = 1e-6
TRAIN_SIZE = 0.7
# HUBER MODIFIER - SET HIGHER ON NEW TRAIN, LOWER ON GOOD LOSS.
HUBER = 2


def preprocess_func(im):
    return  ((im - np.min(im)) / (np.max(im) - np.min(im) + 1e-6) * 255).astype(np.int16)

seq = iaa.SomeOf(2,[
    iaa.AdditiveGaussianNoise(scale=10),
    iaa.Affine(translate_px={"y": (-25, 25)}),
    iaa.Affine(translate_px={"x": (-25, 25)}),
    iaa.Affine(rotate=(-45, 45)),
    iaa.Affine(shear=(-16, 16)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5)
])


seq_null = iaa.SomeOf(5,[
    iaa.AdditiveGaussianNoise(scale=99),
    iaa.Affine(translate_px={"y": (-25, 25)}),
    iaa.Affine(translate_px={"x": (-25, 25)}),
    iaa.Affine(rotate=(-50, 50)),
    iaa.Affine(shear=(-16, 16)),
    iaa.Fliplr(0.5),
    iaa.Flipud(0.5)
])

def colormesh(v):
    plt.pcolormesh(v,norm=mpcolors.SymLogNorm(linthresh=0.5, linscale=1, base=10),  cmap= 'PRGn', shading = 'gouraud')
    plt.show()

def vott_loader(paths, silent = True):
    TAGS = []
    TAGS_FORMAT = {}
    ASSETS = {}
    for vott_fp in paths:
        if not os.path.isfile(vott_fp):
            print("not valid file", vott_fp)
            continue
        if not silent:
            print(vott_fp)
        exported_folder = os.path.split(vott_fp)[0]
        with open(vott_fp,"r") as fio:
            obj = json.load(fio)
        for t in obj["tags"]:
            tagname = t["name"]
            tagcolor = t["color"]
            if tagname not in TAGS:
                TAGS.append(tagname)
                TAGS_FORMAT[tagname] = {"name": tagname, "color": tagcolor}
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

class vott_generator:
    def __init__(self, model, paths, ANCHOR, regions_setting = 5, train_size = 0.7, null_skip_chance = NULL_SKIP, augment_seq = None):
        self.NULL_CLASS = model.outputs[0].shape[2] - 1
        self.TAGS_FORMAT, self.TAGS, self.ASSETS = vott_loader(paths, False)

        self.COLOR_CHANNEL = model.input.shape[3]
        self.ANCHOR = ANCHOR
        self.null_skip_chance = null_skip_chance
        self.image_size = (model.input.shape[1], model.input.shape[2])
        self.regions_setting = regions_setting
        self.augment_seq = augment_seq
        #https://stackoverflow.com/questions/60926460/can-dictionary-data-split-into-test-and-training-set-randomly
        #print("TRAIN SIZE",  train_size)
        s = pd.Series(self.ASSETS)
        
        
        
        if train_size < 1:
            self.training_data , self.test_data  = [i.to_dict() for i in train_test_split(s, train_size=train_size)]
        else:
            #self.test_data = self.training_data
            #pass
            self.training_data = s.to_dict()
            self.test_data = s.to_dict()
        #print(len(self.training_data), len(self.test_data))
        
    def batch(self, batch_size = 8, test_set = False):
        batch = 0
        if test_set:
            asset = self.test_data
        else:
            asset = self.training_data
        while True:
            for key in asset:
                if not os.path.isfile(asset[key]["path"]):
                    #print("DEBUG 165", asset[key]["path"])
                    continue
                try:
                    augmented = self.prepare_image_region(asset[key])
                    if augmented:
                        for augmented_im in augmented:
                            if not augmented_im:
                                #print("DEBUG 172")
                                continue
                            images_aug, labels, bounding_box = augmented_im
                            if images_aug.shape[1:3] == self.image_size and images_aug.shape[-1] in (1,3):
                                if batch == 0:
                                    batch_images = images_aug
                                    batch_class = labels
                                    batch_boundary_boxes = bounding_box
                                    batch += 1
                                elif batch < batch_size:
                                    if np.min(labels, -1)[0] == self.NULL_CLASS and random.random() > 1- self.null_skip_chance:
                                        pass
                                    else:
                                        batch_images = np.append(batch_images, images_aug, 0)
                                        batch_class = np.append(batch_class, labels, 0)
                                        batch_boundary_boxes = np.append(batch_boundary_boxes, bounding_box, 0)
                                        batch += 1
                                else:
                                    batch = 0
                                    yield batch_images, (batch_class, batch_boundary_boxes)
                except Exception as e:
                    if DEBUG:
                        print("DEBUG RAISE")
                        raise e
                    else:
                        pass

    def prepare_image_region(self, ASSET_DATA):
        seq = self.augment_seq
        image_width = ASSET_DATA["asset"]["size"]["width"]
        image_height = ASSET_DATA["asset"]["size"]["height"]
        image_path = ASSET_DATA["path"]
        image_io = tf.io.read_file(image_path)
        if not os.path.isfile(image_path):
            print("nofile", image_path)
            return False
        
        if os.path.splitext(image_path)[-1].lower() == ".bmp":
            tf_img = tf.image.decode_bmp(image_io, channels=3)
        else:
            tf_img = tf.image.decode_jpeg(image_io, channels=3)

        if self.COLOR_CHANNEL == 1:
            tf_img = tf.image.rgb_to_grayscale(tf_img)
        else:
            pass
            
        image_before_augment = tf.expand_dims(tf_img,0).numpy()
        image_before_augment = preprocess_func(image_before_augment)
        batch, height, width, channel = image_before_augment.shape
        bounding_box_list = []
        image_center_distance = math.sqrt(pow(image_width // 2,2) + pow(image_height // 2, 2))
        for region in ASSET_DATA["regions"]:
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
        try:
            anc_res = self.ANCHOR.make(image_before_augment, bounding_box_with_class)
        except Exception as e:
            return False

        if not anc_res:
            return False
            
        for anc_result in anc_res:
            image_before_augment, bounding_box_with_class = anc_result
            try:
                image_before_augment = image_before_augment[tf.newaxis, ...].numpy()
            except Exception as e:
                continue

            bounding_box_with_class = np.array([bounding_box_with_class])
        
            images_aug, bbs_aug_wh = augment(image_before_augment, bounding_box_with_class, seq)
            new_array = []
            
            w = bbs_aug_wh.shape[1]
            h = bbs_aug_wh.shape[0]
            for i in bbs_aug_wh:
                x1 = i.x1
                x2 = i.x2
                y1 = i.y1
                y2 = i.y2
                xz = (x1 + x2) // 2
                yz = (y1 + y2) // 2
            
                c = math.sqrt(pow(max(w//2, xz) - min(w//2, xz),2) + pow(max(h//2, yz) - min(h//2, yz),2))
                label = i.label
                new_array.append([x1, y1, x2, y2, label, c])
            new_array.sort(key = lambda k : 1e6 if k[4] == self.NULL_CLASS else k[5])
            
            bounding_box = np.zeros((1, self.regions_setting, 4), dtype = np.float16)
            bounding_box[:,:,2:] = 1
            labels = np.full((1, self.regions_setting), self.NULL_CLASS, dtype = np.int16)
            for index, i in enumerate(new_array):
                if index >= self.regions_setting:
                    break
                label = i[4]
                if label == self.NULL_CLASS:
                    x1, y1, x2, y2 = (0, 0, 1, 1)
                else:
                    x1 = i[0] / (bbs_aug_wh.shape[1] + 1e-4)
                    y1 = i[1] / (bbs_aug_wh.shape[0] + 1e-4)
                    x2 = i[2] / (bbs_aug_wh.shape[1] + 1e-4)
                    y2 = i[3] / (bbs_aug_wh.shape[0] + 1e-4)
                bounding_box[:, index, :] = np.array([y1, x1, y2, x2], dtype = np.float16)
                bounding_box = np.where(bounding_box < 0, 0, bounding_box)
                bounding_box = np.where(bounding_box > 1, 1, bounding_box)
                labels[:, index] = label
            images_aug = tf.image.resize(images_aug, self.image_size, method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=False, antialias=False, name=None)
            images_aug = np.expand_dims(images_aug, 0)
            yield images_aug, labels, bounding_box
        

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

def from_hex(h):
    s = h.strip("#")
    return [int(s[:2],16), int(s[2:4], 16), int(s[4:], 16)]

class anchor:
    def __init__(self, anchor_level, tag_path, model, overlap_requirement = 1.0, savename="pik", non_max_supression_iou = 0.01):
        self.boxes = self._boxes(anchor_level)
        self.box_indices = tf.zeros(shape=(self.boxes.shape[0],), dtype = "int32")
        self.CROP_SIZE = (model.input.shape[1], model.input.shape[2])
        self.OVERLAP_REQUIREMENT = overlap_requirement
        self.NON_MAX_SUPPRESSION_IOU = non_max_supression_iou
        try:
            self.load_tags(tag_path, savename)
        except Exception as e:
            print("anchor.load_tags error", e)
        self.model = model

    def load_tags(self, votts = VOTTS, savename = SAVENAME):
        if os.path.isfile(savename + ".pik"):
            with open(savename + ".pik", "rb") as fio:
                self.tag_format = pickle.load(fio)
        else:
            try:
                self.tag_format, tags, _ = vott_loader(votts)
                with open(savename + ".pik", "wb") as fio:
                    pickle.dump(self.tag_format, fio)
            except FileNotFoundError as e:
                print(e)
        
    def _boxes(self, level = 2):
        boxes = tf.constant([[0.01, 0.01, 0.99, 0.99]])
        if level > 0:
            b = []
            for i in range(level):
                b.append(tf.constant(self.generate_box(2 + i)))
            b = tf.concat(b, 0)
            boxes = tf.concat([boxes, b], 0)
        return boxes

    def generate_box(self, size, border = 0.01):
        a = [n/size for n in range(size + 1)]
        box = []
        for b1 in range(size):
            for b2 in range(size):
                x1 = a[b1] + border
                y1 = a[b2] + border
                x2 = a[b1 + 1] - border
                y2 = a[b2 + 1] - border                
                box.append([x1, y1, x2, y2])
        for b1 in range(size - 1):
            for b2 in range(size):
                x1 = (a[b1] + a[b1 + 1]) / 2 + border
                y1 = a[b2] + border
                x2 = (a[b1 + 1] + a[b1 + 2]) / 2 - border
                y2 = a[b2 + 1] - border               
                box.append([x1, y1, x2, y2])

        for b1 in range(size):
            for b2 in range(size - 1):               
                x1 = a[b1] + border
                y1 = (a[b2] + a[b2 + 1]) / 2 + border
                x2 = a[b1 + 1] - border
                y2 = (a[b2 + 1] + a[b2 + 2]) / 2 - border            
                box.append([x1, y1, x2, y2])
                
        for b1 in range(size - 1):
            for b2 in range(size - 1):
                x1 = (a[b1] + a[b1 + 1]) / 2 + border
                y1 = (a[b2] + a[b2 + 1]) / 2 + border
                x2 = (a[b1 + 1] + a[b1 + 2]) / 2 - border
                y2 = (a[b2 + 1] + a[b2 + 2]) / 2 - border
                box.append([x1, y1, x2, y2])        
        return box

    def make(self, image, bounding_box_with_class):
        im_shape = image.shape
        output = tf.image.crop_and_resize(image, self.boxes, self.box_indices, self.CROP_SIZE)
        bounding_boxes = []
        for index, i in enumerate(self.boxes):
            x1 = int(i[1] * im_shape[2])
            y1 = int(i[0] * im_shape[1])
            x2 = int(i[3] * im_shape[2])
            y2 = int(i[2] * im_shape[1])

            sub_box = []
 
            for j in bounding_box_with_class[0]:
                x3 = int(j[0])
                y3 = int(j[1])
                x4 = int(j[2])
                y4 = int(j[3])

                boxA = [x1, y1, x2, y2]
                boxB = [x3, y3, x4, y4]

                tag_class = int(j[4])

                intersect = bb_intersection(boxA, boxB)

                if intersect >= self.OVERLAP_REQUIREMENT:

                    bx1 = max(x1, x3) - x1
                    by1 = max(y1, y3) - y1
                    bx2 = min(x2, x4) - x1
                    by2 = min(y2, y4) - y1

                    ow = output.shape[2]
                    oh = output.shape[1]
                    rx1 = math.floor(bx1 *  (ow / (x2 - x1)))
                    rx2 = math.floor(bx2 *  (ow / (x2 - x1)))
                    ry1 = math.floor(by1 *  (oh / (y2 - y1)))
                    ry2 = math.floor(by2 *  (oh / (y2 - y1)))
                    sub_box.append([rx1, ry1, rx2, ry2, tag_class])
                else:
                    pass          
            bounding_boxes.append(sub_box)
        #print(len(output), len(self.boxes), len(bounding_boxes))
        return zip(output, bounding_boxes)

    def predict(self, fp, show = True, rawdata = False):
        tags = [k for k in self.tag_format]
        raw_image = tf.io.read_file(fp)
        if os.path.splitext(fp)[-1].lower() == ".bmp":
            raw_image = tf.image.decode_bmp(raw_image, channels=3)
        else:
            raw_image = tf.image.decode_jpeg(raw_image, channels=3)
        if self.model.input.shape[-1] == 1:
            image = tf.image.rgb_to_grayscale(raw_image)
        else:
            image = copy.copy(raw_image)
        
        image = tf.expand_dims(image,0)
        im_shape = image.shape

        raw_image = tf.expand_dims(raw_image, 0)
        
        output = tf.image.crop_and_resize(image, self.boxes, self.box_indices, self.CROP_SIZE)
        output = preprocess_func(output.numpy())
        
        inference = self.model.predict(output, verbose = 0)

        classifier = tf.argmax(inference[0], axis = -1).numpy()
        arr = np.expand_dims(classifier,-1)
        classifier = classifier.tolist()
        
        score = np.take_along_axis(inference[0], arr, 2)
        score = np.max(score,-1)
        
        bb = inference[1]

        check_within_tags = tf.argmax(inference[0], axis=-1).numpy()

        combined_boundary_boxes = []
        combined_colors = []
        combined_class = []
        color_list = []
        tag_list = []
        for cnt, img in enumerate(output):
            within_tags = np.where(check_within_tags[cnt] < len(tags))

            cf = np.take(check_within_tags[cnt], within_tags, axis = 0)
            bf = np.take(bb[cnt], within_tags, axis = 0)

            box_h1 = self.boxes[cnt][0] * im_shape[1]
            box_h2 = self.boxes[cnt][2] * im_shape[1]
            box_w1 = self.boxes[cnt][1] * im_shape[2]
            box_w2 = self.boxes[cnt][3] * im_shape[2]
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
                    txt = tag + " " + tag_score
                    tag_list.append({"sequence" : cnt, "score" : score[cnt][i], "box" : bb_actual[0][i], "tag" : tags[n], "class" : n, "text" : txt, "color" : colors[i]})

        if raw_image.shape[-1] == 1:
            image = tf.image.grayscale_to_rgb(raw_image, name=None)
        else:
            image = raw_image
        image = tf.image.resize(image, (512, 384), method=tf.image.ResizeMethod.BILINEAR, preserve_aspect_ratio=True, antialias=False, name=None)

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
                raw_result = {}
                x1 = int(i["box"][1] * output_im.shape[1])
                y1 = int(i["box"][0] * output_im.shape[0])
                x2 = int(i["box"][3] * output_im.shape[1])
                y2 = int(i["box"][2] * output_im.shape[0])
                cv2.putText(output_im, i["tag"], (x1 + 1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 1, i["color"], 1)
                cv2.putText(output_im, "%02d%%" % (i["score"] * 100), (x1 + 1, y1 + 18), cv2.FONT_HERSHEY_PLAIN, 0.6, i["color"])

                raw_result["x1"] = x1
                raw_result["y1"] = y1
                raw_result["x2"] = x2
                raw_result["y2"] = y2
                raw_result["tag"] = i["tag"]
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

    def predict2(self, fp):
        tags = [k for k in self.tag_format]
        image = tf.io.read_file(fp)
        if os.path.splitext(fp)[-1].lower() == ".bmp":
            image = tf.image.decode_bmp(image, channels=3)
        else:
            image = tf.image.decode_jpeg(image, channels=3)
            
        if self.model.input.shape[-1] == 1:
            image = tf.image.rgb_to_grayscale(image)
        else:
            image = copy.copy(raw_image)
        image = tf.expand_dims(image,0)
        im_shape = image.shape
        print("IMAGE SHAPE", image.shape)
        output = tf.image.crop_and_resize(image, self.boxes, self.box_indices, self.CROP_SIZE)
        output = preprocess_func(output.numpy())
                
        inference = model.predict(output, verbose = 0)

        classifier = tf.argmax(inference[0], axis = -1).numpy()
        arr = np.expand_dims(classifier,-1)
        classifier = classifier.tolist()
        
        score = np.take_along_axis(inference[0], arr, 2)
        score = np.max(score,-1)
        
        bb = inference[1]

        check_within_tags = tf.argmax(inference[0], axis=-1).numpy()

        fig_sq = int(math.sqrt(self.boxes.shape[0])) + 1
        fig,ax = plt.subplots(fig_sq , fig_sq)
        for row in ax:
            for col in row:
                col.axis("off")

        combined_boundary_boxes = []
        combined_colors = []
        combined_class = []
        color_list = []
        tag_list = []
        for cnt, img in enumerate(output):
            row = cnt // fig_sq
            col = cnt % fig_sq

            output_im = tf.image.grayscale_to_rgb(img, name=None)
            output_im = output_im.numpy()[tf.newaxis, ...]
            within_tags = np.where(check_within_tags[cnt] < len(tags))

            cf = np.take(check_within_tags[cnt], within_tags, axis = 0)
            bf = np.take(bb[cnt], within_tags, axis = 0)

            box_h1 = self.boxes[cnt][0] * im_shape[1]
            box_h2 = self.boxes[cnt][2] * im_shape[1]
            box_w1 = self.boxes[cnt][1] * im_shape[2]
            box_w2 = self.boxes[cnt][3] * im_shape[2]
            box_hh = box_h2 - box_h1
            box_ww = box_w2 - box_w1

            bounding_boxes = bf 

            bb_actual = bounding_boxes * [box_hh, box_ww, box_hh, box_ww]
            bb_actual = bb_actual + [box_h1, box_w1, box_h1, box_w1]
            bb_actual = bb_actual / [im_shape[1], im_shape[2], im_shape[1], im_shape[2]]

            bb_actual = bb_actual.tolist()
            colors = [from_hex(self.tag_format[tags[n]]["color"]) for n in cf[0] if n > -1]
            if bb_actual:
                combined_boundary_boxes.extend(bb_actual[0])
                combined_colors.extend(colors)    
                colors.extend([[255,255,255]])

            output_im = tf.image.draw_bounding_boxes(output_im, bounding_boxes, colors)
            output_im = output_im[0].numpy()

            tx = []
            for i, n in enumerate(cf[0]):
                if n < len(tags):
                    tag = tags[n]
                    tag_score = str(int(score[cnt][i] * 100)) + "%"
                    y1, x1, y2, x2 = bb[cnt][i]
                    x = int(x1 * self.CROP_SIZE[1])
                    y = int(y1 * self.CROP_SIZE[0])
                    txt = tag + " " + tag_score
                    tag_list.append({"sequence" : cnt, "score" : score[cnt][i], "box" : bb_actual[0][i], "tag" : tags[n], "class" : n, "text" : txt, "x" : bb_actual[0][i][1], "y" : bb_actual[0][i][0], "color" : colors[i]})
                    tx.append(cv2.putText(output_im, txt, (x, y + 20), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[i]))

            if cnt < (fig_sq * fig_sq):   
                ax[row, col].set_title(cnt, fontdict= {'fontsize': 6})
                im = tf.keras.preprocessing.image.array_to_img(output_im)
                ax[row, col].imshow(im)
                ax[row, col].axis('off')

        #final image ---------------------
        ax[fig_sq - 1, fig_sq - 1].set_title("finale", fontdict= {'fontsize': 6})
        image = tf.image.grayscale_to_rgb(image, name=None)


        boxes = [n["box"] for n in tag_list]
        scores = [n["score"] for n in tag_list]

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
                x1 = int(i["box"][1] * output_im.shape[1])
                y1 = int(i["box"][0] * output_im.shape[0])
                x2 = int(i["box"][3] * output_im.shape[1])
                y2 = int(i["box"][2] * output_im.shape[0])
                cv2.putText(output_im, i["tag"], (x1 + 1, y1 + 10), cv2.FONT_HERSHEY_PLAIN, 1, i["color"], 1)
                cv2.putText(output_im, "%02d%%" % (i["score"] * 100), (x1 + 1, y1 + 18), cv2.FONT_HERSHEY_PLAIN, 0.6, i["color"])
        else:
            output_im = image[0]

        im = tf.keras.preprocessing.image.array_to_img(output_im)
        ax[fig_sq - 1, fig_sq - 1].imshow(im)
        ax[fig_sq - 1, fig_sq - 1].axis('off')
        
        plt.show()
        plt.clf()
        plt.close()
        return tag_list

def augment(im, bbwc, seq = None):
    im = im[0]
    bbs = BoundingBoxesOnImage([BoundingBox(x1 = n[0], y1 = n[1], x2 = n[2], y2 = n[3], label = n[4]) for n in bbwc[0]], shape = im.shape[:2])
    iaaseq = [iaa.AdditiveGaussianNoise(scale=10), \
              iaa.Affine(translate_px={"y": (-25, 25)}), \
              iaa.Affine(translate_px={"x": (-25, 25)}), \
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
        seq = iaa.SomeOf(2,s)
    images_aug, bbs_aug_wh = seq(image = im, bounding_boxes = bbs)
    return images_aug, bbs_aug_wh

def augment_null(im, bb_hw):
    im = im[0]
    images_aug, bbs_aug_wh = seq_null(image = im, bounding_boxes = bb_hw)
    images_aug = np.expand_dims(images_aug, 0)
    bbs_aug_hw = np.zeros((1,1,4))
    return images_aug, bbs_aug_hw

def sanity_check(gen):
    x,(y,z) = gen.__next__()
    colors = [[255,0,0],]
    imbb = tf.image.draw_bounding_boxes(x, z, colors)
    for i in range(x.shape[0]):
        plt.imshow(tf.keras.preprocessing.image.array_to_img(imbb[i]))
        try:
            plt.title(",".join([tags[n] for n in y[i] if n < len(tags)]))
        except Exception as e:
            print(e)
        print(y[i])
        plt.show()
        plt.clf()
        plt.close()

def test():
    #IMAGE_SIZE, COLOR_CHANNEL, CLASSIFICATION_TAGS, REGIONS, DROPOUT, FPN_MODE, BACKBONE, VOTT_PATHS
    m = load_model(128, 1, 10, 2, 0.2, 2, "B1", VOTTS, 255)
    m.BATCH_SIZE = 8
    m.HUBER = 100
    m.TRAIN_SIZE = 0.7
    m.ANCHOR_LEVEL = 2
    m.NULL_SKIP = 0.3
    m.OVERLAP_REQUIREMENT = 1.0
    m.SAVENAME = "pookeymack"
    
    m.initialize()
    return m

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
        self.SAVENAME = "pookeymack"
        self.NON_MAX_SUPPRESSION_IOU = 0.01
        self.LSTM = LSTM

    def initialize(self):
        print("initialize model")
        self.model = det.edet(self.IMAGE_SIZE, self.COLOR_CHANNEL, self.CLASSIFICATION_TAGS, self.REGIONS, dropout = self.DROPOUT, bi = self.FPN_MODE, backbone = self.BACKBONE, LSTM = self.LSTM)
        self.model.compile(optimizer=Adam(learning_rate=1e-3), loss={'regression': self.iouhuloss, "classification": tf.keras.losses.SparseCategoricalCrossentropy()},)
        self.anchor = anchor(self.ANCHOR_LEVEL, self.VOTT_PATHS, self.model, self.OVERLAP_REQUIREMENT, self.SAVENAME, self.NON_MAX_SUPPRESSION_IOU)
        self.traindata = None
        try:
            self.gen = vott_generator(self.model, self.VOTT_PATHS, self.anchor, regions_setting = self.REGIONS, train_size = self.TRAIN_SIZE, null_skip_chance = self.NULL_SKIP, augment_seq = self.AUGMENT)
        except Exception as e:
            print(e)
            self.gen = False
            print("generator will be off, not able to train")

    def iouhuloss(self, y_true, y_pred):
        fl = tf.keras.losses.Huber()
        gl = tfa.losses.GIoULoss()
        f = fl(y_true, y_pred)
        g = gl(y_true, y_pred)
        return g + f * self.HUBER

    def train(self, EPOCHS = 200, STEPS = 50, LR = 0.001):
        tf.keras.backend.clear_session()
        gen = vott_generator(self.model, self.VOTT_PATHS, self.anchor, regions_setting = self.REGIONS, train_size = self.TRAIN_SIZE, null_skip_chance = self.NULL_SKIP, augment_seq = self.AUGMENT)
        train_data = gen.batch(self.BATCH_SIZE, test_set = False)
        validation_data = gen.batch(self.BATCH_SIZE, test_set = True)
        self.model.optimizer.learning_rate = LR
        dtnow = datetime.datetime.now()
        history = self.model.fit(train_data, validation_data = validation_data, epochs = EPOCHS, steps_per_epoch = STEPS, validation_steps = int(STEPS * 0.5), verbose = 2)
        
        loss = history.history['loss']
        classification_loss = history.history['classification_loss']
        regression_loss = history.history['regression_loss']

        val_loss = history.history['val_loss']
        val_classification_loss = history.history['val_classification_loss']
        val_regression_loss = history.history['val_regression_loss']
        
        epochs_range = range(EPOCHS)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, loss, label='loss')
        plt.plot(epochs_range, val_loss, label='val_loss')
        plt.legend(loc='lower right')
        plt.title('LOSS')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, classification_loss, label='classification_loss')
        plt.plot(epochs_range, regression_loss, label='regression_loss')
        plt.plot(epochs_range, val_classification_loss, label='val_classification_loss')
        plt.plot(epochs_range, val_regression_loss, label='val_regression_loss')
        plt.legend(loc='upper right')
        plt.title('Classification and regression')

        plt.legend()
        plt.show()
        plt.clf()
        plt.close()

    def save(self, f = None):
        if not f:
            f = self.SAVENAME
        colors = [[255,0,0],]
        self.model.save_weights(f)
        print("SAVED WEIGHTS")

    def load(self, f = None):
        if not f:
            f = self.SAVENAME
        try:
            self.model.load_weights(f)
            print("LOADED WEIGHTS")
        except Exception as e:
            print(e)

    def generator_check(self):
        if not self.traindata:
            gen = vott_generator(self.model, self.VOTT_PATHS, self.anchor, regions_setting = self.REGIONS, train_size = self.TRAIN_SIZE, null_skip_chance = self.NULL_SKIP, augment_seq = self.AUGMENT)
            self.traindata = gen.batch(self.BATCH_SIZE, test_set = False)
        self.sanity_check(self.traindata)


    def sanity_check(self, gen):
        x,(y,z) = gen.__next__()
        colors = [[255,0,0],]
        items = x.shape[0]
        h = int(math.ceil(math.sqrt(items)))
        w = int(math.ceil(math.sqrt(items)))
        fig,ax = plt.subplots(h , w)
        for row in ax:
            for col in row:
                col.axis("off")
                
        imbb = tf.image.draw_bounding_boxes(x, z, colors)
        for i in range(x.shape[0]):
            row = i // h
            col = i % w
            im = tf.keras.preprocessing.image.array_to_img(imbb[i])
            #plt.imshow(im)
            ax[row, col].imshow(im)
            try:
                title = ",".join([self.gen.TAGS[n] for n in y[i] if n < len(self.gen.TAGS)])
                #plt.title(title)
                ax[row, col].set_title(title, fontdict= {'fontsize': 6})
            except Exception as e:
                print(e)
            ax[row, col].axis('off')
            #print(y[i])
        plt.show()
        plt.clf()
        plt.close()


    def folder_check(self, folder):
        jpgs = filter(lambda f : os.path.splitext(f)[1] in (".jpg", ".png", ".bmp") , os.listdir(folder))
        cnt = 0
        fig,ax = plt.subplots(3 , 5)
        for row in ax:
            for col in row:
                col.axis("off")
        while cnt < 15:
            try:
                im = jpgs.__next__()
                print(im)
            except Exception as e:
                print(e)
                cnt = 100
                break
            row = cnt // 5
            col = cnt % 5
            fp = os.path.join(folder, im)
            print(cnt, fp)
            im = self.anchor.predict(fp, False)
            ax[row, col].set_title(im, fontdict= {'fontsize': 6})
            ax[row, col].imshow(im)
            ax[row, col].axis('off')
            cnt += 1
        
        plt.show()
        plt.clf()
        plt.close()

    def trial(self, testfolder):
        for f in os.listdir(testfolder):
            fp = os.path.join(testfolder,f)
            if os.path.isdir(fp):
                self.folder_check(fp)




def run(start_val_loss = None , rate = 0.001, endless = True, EPOCHS = 50, STEPS = 15,  VALIDATION_STEPS = 10):
    load()
    if start_val_loss == None:
        with open("%s_run.pik" % SAVENAME,"rb") as fio:
            start_val_loss = pickle.load(fio)
            print("LOAD PIK",start_val_loss)
    RUN = True
    model.optimizer.learning_rate = rate
    dtnow = datetime.datetime.now()
    lowest_val = 9999
    LR = (rate, rate * 2)
    while RUN:
        tf.keras.backend.clear_session()
        gen = vott_generator(paths = VOTTS, regions_setting = REGIONS, train_size = TRAIN_SIZE, color_channel = COLOR_CHANNEL)
        train_data = gen.batch(BATCH_SIZE, test_set = False)
        validation_data = gen.batch(BATCH_SIZE, test_set = True)
        #logger = tf.keras.callbacks.CSVLogger("efficient.csv", separator=",", append=True)
        callback = tf.keras.callbacks.EarlyStopping(monitor = "val_loss", min_delta = 0, patience = 15, verbose = 0, mode = "min", baseline = start_val_loss, restore_best_weights = True)
        #callback = tf.keras.callbacks.EarlyStopping(monitor = "loss", min_delta = 0, patience = 15, verbose = 0, mode = "min", baseline = start_val_loss, restore_best_weights = True)
        model_history = model.fit(train_data, validation_data = validation_data, epochs = EPOCHS, steps_per_epoch = STEPS, validation_steps = VALIDATION_STEPS, verbose = 2, callbacks=[callback])
        min_val_loss = min(model_history.history['val_loss'])
        loss = min(model_history.history['loss'])
        print(min_val_loss)
        if min_val_loss <= start_val_loss and min_val_loss <= lowest_val and min_val_loss >= loss * 0.5:
            print("SAVED", min_val_loss)
            start_val_loss = min_val_loss
            save()
            with open("%s_run.pik" % SAVENAME,"wb") as fio:
                pickle.dump(start_val_loss, fio)
        else:
            print("No progress", min_val_loss, "BEST : ", start_val_loss)
            if min_val_loss / start_val_loss < 1.1:
                print("ELSA : LET IT GO!!")
            else:
                print("LOAD")
                load()
            rate = random.sample(LR, 1)[0]
            model.optimizer.learning_rate = rate
            print("UPDATE rate : ",rate)
        if min_val_loss < lowest_val:
            lowest_val = min_val_loss
        if not endless:
            RUN = False
            print("STOPPING", lowest_val)

def image_resize(fol, newsize = 640):
    for f in os.listdir(fol):
        fp = os.path.join(fol, f)
        fn, ext = os.path.splitext(f)
        if ext in (".jpg", ".png", ".bmp"):
            newfp = os.path.join(fol, fn + ".jpg")
            im = Image.open(fp)
            ratio = min(im.size) / newsize
            imn = im.resize((int(im.size[0]//ratio), int(im.size[1]//ratio)), Image.Resampling.NEAREST)
            imn.save(newfp, "jpeg")
 
if __name__ == "__main__":
    pass
    #bi = False, True, "full"
##    model = det.edet(IMAGE_SIZE, COLOR_CHANNEL, CLASSIFICATION_TAGS, REGIONS, dropout = DROPOUT, bi = FPN_MODE, backbone = BACKBONE)
##    try:
##        tag_format, tags, _ = vott_loader(VOTTS)
##        with open(SAVENAME + ".pik", "wb") as fio:
##            pickle.dump(tag_format, fio)
##    except FileNotFoundError as e:
##        print(e)
##        with open(SAVENAME + ".pik", "rb") as fio:
##            tag_format = pickle.load(fio)        
##    model.compile(optimizer=Adam(learning_rate=1e-3), loss={'regression': iouhuloss, "classification":cl},)
##
    


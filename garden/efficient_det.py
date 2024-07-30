import tensorflow as tf
import tensorflow_addons as tfa

def edet(input_shape = (256, 256, 3), num_classes = 1000, detection_region = 5, dropout = 0.2, backbone = "B0"):
    x_in = tf.keras.Input(shape = input_shape)
    if backbone == "B0":
        backbone = tf.keras.applications.efficientnet.EfficientNetB0(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "B1":
        backbone = tf.keras.applications.efficientnet.EfficientNetB1(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)    
    elif backbone == "B2":
        backbone = tf.keras.applications.efficientnet.EfficientNetB2(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "B3":
        backbone = tf.keras.applications.efficientnet.EfficientNetB3(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "B4":
        backbone = tf.keras.applications.efficientnet.EfficientNetB4(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "B5":
        backbone = tf.keras.applications.efficientnet.EfficientNetB5(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "B6":
        backbone = tf.keras.applications.efficientnet.EfficientNetB6(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "B7":
        backbone = tf.keras.applications.efficientnet.EfficientNetB7(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "V2S":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "V2B0":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "V2B1":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B1(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "V2B2":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B2(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    elif backbone == "V2B3":
        backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2B3(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    else:
        backbone = tf.keras.applications.efficientnet.EfficientNetB0(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)

    layer_block = []
    cur_shape = None
    for i, j in enumerate(backbone.layers):
        if "add" in j.name:
            #print(j.name, j.output.shape)
            if j.output.shape != cur_shape:
                cur_shape = j.output.shape
                layer_block.append(j.output)
            else:
                layer_block[-1] = j.output

    bn_activation = []
    for i, j in enumerate(layer_block):
        c = tf.keras.layers.BatchNormalization(axis=-1, name = "before_FPN_BN_%s" % i)(j)
        c = tf.keras.layers.Activation('relu', name = "before_FPN_Activation_%s" % i)(c)
        c = tf.keras.layers.Dropout(dropout)(c)
        bn_activation.append(c)
    #bn_activation.append(backbone.layers[-1].output)
        
    #f1 = biFPN(layer_block, sequence = 1)
    f1 = biFPN(bn_activation, sequence = 1)
    f2 = biFPN(f1, sequence = 2)
    f3 = biFPN(f2, sequence = 3)

    for i, j in enumerate(layer_block):
        print("layer block", i, j.name, j.shape)
        
    global_avg_pool = []
    global_avg_pool_classification = []
    global_avg_pool_regression = []
    for i, j in enumerate(f3):
        if j.shape[1] > 3:
            s2 = tf.keras.layers.Conv2D(filters = j.shape[-1] * 2, kernel_size= 3, strides = 2, padding = 'valid', activation = "relu", name = "gp_Conv2D_%02d_%02d" % (j.shape[1], j.shape[3]))(j)
            s2 = tf.keras.layers.BatchNormalization(name = "gp_BN_%02d_%02d" % (j.shape[1], j.shape[3]))(s2)
            s2 = tf.keras.layers.Activation("relu", name = "gp_Activation_%02d_%02d" % (j.shape[1], j.shape[3]))(s2)
            pooling = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling2d_%02d_%02d" % (j.shape[1], j.shape[3]))(s2)
        else:
            pooling = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling2d_%02d_%02d" % (j.shape[1], j.shape[3]))(j)
        global_avg_pool.append(pooling)
        
    r = tf.keras.layers.Concatenate(axis=-1)(global_avg_pool)

    print(r.shape)
    
    b = tf.keras.layers.Dense(4 * detection_region, name = "dense_regression")(r)
    b = tf.keras.layers.Reshape([detection_region, 4], name = "reshape_regression")(b)
    b = tf.keras.layers.Activation('sigmoid', name = "activation_regression")(b)
    regression = tf.keras.layers.Layer(name = "regression")(b)

    c = tf.keras.layers.Dense(num_classes * detection_region,  name = "dense_classification")(r)
    c = tf.keras.layers.Reshape([detection_region, num_classes], name = "reshape_classification")(c)
    c = tf.keras.layers.BatchNormalization(axis=-1, name = "BN_classification")(c)
    c = tf.keras.layers.Activation('softmax', name = "activation_classification")(c)
    classification = tf.keras.layers.Layer(name = "classification")(c)
    
    model = tf.keras.Model(x_in, [classification, regression])
    return model

def biFPN(layers, KERNELS = 3, end_activation = "relu", dropout = 0.2, sequence = 1):
    f1_block = layers
    f2_block = []
    f3_block = []
    f4_block = []
    f5_block = []

    #F2
    for i, j in enumerate(f1_block):
        if i == 0 or i == len(f1_block) - 1:
            #F2_0, #F2_3
            c = tf.keras.layers.Layer(name = "fpn_%02d_f2_%02d" % (sequence, i))(j)
            f2_block.append(c)
        elif i == 1:
            #F2_1
            strides = c.shape[1] // j.shape[1]
            #x = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=1, padding='same', activation = "relu", name = "fpn_%02d_f2_%02d_Conv2D_x" % (sequence, i))(j)
            x = j
            y = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = "relu", name = "fpn_%02d_f2_%02d_Conv2D_y" % (sequence, i - 1))(c)
            y = tf.keras.layers.BatchNormalization(name = "fpn_%02d_f2_%02d_BN" % (sequence, i - 1))(y)
            y = tf.keras.layers.Activation("relu", name = "fpn_%02d_f2_%02d_Activation" % (sequence, i - 1))(y)             
            c = tf.keras.layers.Add(name = "fpn_%02d_f2_%02d_ADD" % (sequence, i))([x, y])
            f2_block.append(c)
        else:
            #F2_2        
            strides = c.shape[1] // j.shape[1]
            #x = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=1, padding='same', activation = "relu", name = "fpn_%02d_f2_%02d_Conv2D_x" % (sequence, i))(j)
            x = j
            y = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = "relu", name = "fpn_%02d_f2_%02d_Conv2D_y" % (sequence, i - 1))(c)
            y = tf.keras.layers.BatchNormalization(name = "fpn_%02d_f2_%02d_BN" % (sequence, i - 1))(y)
            y = tf.keras.layers.Activation("relu", name = "fpn_%02d_f2_%02d_Activation" % (sequence, i - 1))(y)             
            c = tf.keras.layers.Add(name = "fpn_%02d_f2_%02d_ADD" % (sequence, i))([x, y])
            f2_block.append(c)

    #F3
    for i, j in enumerate(f2_block):
        #F3_0
        if i == 0:
            c = tf.keras.layers.Layer(name = "fpn_%02d_f3_%02d" % (sequence, i))(j)
            f3_block.append(c)
        #F3_1, F3_2
        elif i != len(f2_block) - 1:
            x = j
            #x = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=1, padding='same', activation = "relu", name = "fpn_%02d_f3_%02d_Conv2D_x" % (sequence, i))(j)
            #skip layer
            y = f1_block[i]
            c = tf.keras.layers.Add(name = "fpn_%02d_f3_%02d_ADD" % (sequence, i))([x, y])
            f3_block.append(c)
        #F3_3
        else:
            strides = f2_block[i - 1].shape[1] // j.shape[1]
            x = j
            y = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = "relu", name = "fpn_%02d_f3_%02d_Conv2D_y" % (sequence, i - 1))(f2_block[i - 1])
            y = tf.keras.layers.BatchNormalization(name = "fpn_%02d_f3_%02d_BN" % (sequence, i - 1))(y)
            y = tf.keras.layers.Activation("relu", name = "fpn_%02d_f3_%02d_Activation" % (sequence, i - 1))(y)            
            c = tf.keras.layers.Add(name = "fpn_%02d_f3_%02d_ADD" % (sequence, i))([x, y])
            f3_block.append(c)

    #F4
    f3_block.reverse()
    for ri, j in enumerate(f3_block):
        i = len(f3_block) - (ri + 1)
        #F4_3
        if ri == 0:
            c = tf.keras.layers.Layer(name = "fpn_%02d_f4_%02d" % (sequence, i))(j)
            f4_block.append(c)
        #F4_2, F4_1, F4_0
        else:
            strides = j.shape[1] // c.shape[1]
            x = j
            y = tf.keras.layers.Conv2DTranspose(j.shape[-1], KERNELS, strides=(strides, strides), padding="same", activation = "relu", name = "fpn_%02d_f4_%02d_Conv2D" % (sequence, i + 1))(c)
            c = tf.keras.layers.Add(name = "fpn_%02d_f4_%02d_ADD" % (sequence, i))([x, y])
            f4_block.append(c)

    #F5
    f4_block.reverse()
    for i, j in enumerate(f4_block):
        x = tf.keras.layers.Layer(name = "fpn_%02d_f5_%02d" % (sequence, i))(j)
        #x = tf.keras.layers.Conv2D(filters=j.shape[-1], kernel_size=KERNELS, strides=1, padding='same', activation = "relu", name = "fpn_%02d_f5_%02d_Conv2D" % (sequence, i))(j)
        x = tf.keras.layers.BatchNormalization(name = "fpn_%02d_f5_%02d_BN" % (sequence, i))(x)
        x = tf.keras.layers.Activation("relu", name = "fpn_%02d_f5_%02d_Activation" % (sequence, i))(x)
        f5_block.append(x)

    return f5_block


fl = tf.keras.losses.Huber(reduction = 'sum')
gl = tfa.losses.GIoULoss()
def regression_loss(y_true, y_pred):
    f = fl(y_true, y_pred, 1.0)
    g = gl(y_true, y_pred, 1.0)
    return g + f

classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()


def experiment_loss(L1, L2):
    print(L1, L2)

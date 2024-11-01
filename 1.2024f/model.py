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
    elif backbone == "MobileNetV2":
        backbone = tf.keras.applications.MobileNetV2(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)
    else:
        backbone = tf.keras.applications.efficientnet.EfficientNetB0(input_tensor = x_in, include_top = False, weights = None, input_shape = x_in.shape[1:], classes = num_classes)

    layer_block = []
    cur_shape = [0, 0, 0]
    for i, j in enumerate(backbone.layers):
        if "add" in j.name:
            #print(j.name, j.output.shape)
            if j.output.shape[1:] != cur_shape:
                cur_shape = j.output.shape[1:]
                layer_block.append(j.output)
            else:
                layer_block[-1] = j.output

    bn_activation = []
    for i, j in enumerate(layer_block):
        c = tf.keras.layers.BatchNormalization(axis=-1, name = "before_FPN_BN_%s" % i)(j)
        c = tf.keras.layers.Activation('relu', name = "before_FPN_Activation_%s" % i)(c)
        c = tf.keras.layers.Dropout(dropout)(c)
        bn_activation.append(c)

    f1 = biFPN(bn_activation, end_activation = tf.keras.layers.LeakyReLU(), sequence = 1)
    f2 = biFPN(f1, end_activation = tf.keras.layers.LeakyReLU(), sequence = 2)
    f3 = biFPN(f2, end_activation = tf.keras.layers.LeakyReLU(), sequence = 3)


    for i, j in enumerate(layer_block):
        print("layer block", i, j.name, j.shape)
        
    global_avg_pool = []
    global_avg_pool_classification = []
    global_avg_pool_regression = []
    for i, j in enumerate(f3):
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
    input_block = layers #input block
    skip_block = [] #skip block
    downsample_block = [] #downsample block
    dn_add_block = [] #downsample add block
    upsample_block = [] #upsample block
    up_add_block = [] #upsample add block
    output_block = [] #output block

    #F2
    for i, k in enumerate(input_block):
        if i < len(input_block) - 1:
            skip_block.append(tf.keras.layers.Layer(name = "fpn_%02d_input_%02d" % (sequence, i))(k))
        else:
            skip_block.append(None)
        upsample_block.append(None)
        up_add_block.append(None)
        output_block.append(None)
        dn_add_block.append(None)
        downsample_block.append(None)
        
    #F3 - DOWNSAMPLE and ADD
    for i, k in enumerate(input_block):
        if i == 0:
            k2 = input_block[i + 1]
            strides = int(k.shape[1] // k2.shape[1])
            downblock = tf.keras.layers.Conv2D(filters=k2.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = end_activation, name = "fpn_%02d_downsample_%02d_Conv2D" % (sequence, i))(k)
            downblock = tf.keras.layers.BatchNormalization(name = "fpn_%02d_downsample_%02d_BN" % (sequence, i))(downblock)
            downblock = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_downsample_%02d_Activation" % (sequence, i))(downblock)
            downsample_block[i + 1] = downblock
        elif i < len(input_block) - 1:
            k2 = input_block[i + 1]
            strides = int(k.shape[1] // k2.shape[1])
            k1 = downblock #downsample_block[i]
            addblock = tf.keras.layers.Add(name = "fpn_%02d_dn_add_%02d" % (sequence, i))([k, k1])
            dn_add_block[i] = addblock
            downblock = tf.keras.layers.Conv2D(filters=k2.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = end_activation, name = "fpn_%02d_downsample_%02d_Conv2D" % (sequence, i))(addblock)
            downblock = tf.keras.layers.BatchNormalization(name = "fpn_%02d_downsample_%02d_BN" % (sequence, i))(downblock)
            downblock = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_downsample_%02d_Activation" % (sequence, i))(downblock)
            downsample_block[i + 1] = downblock
        else:
            k1 = downblock #downsample_block[i]
            addblock = tf.keras.layers.Add(name = "fpn_%02d_dn_add_%02d" % (sequence, i))([k, k1])
            dn_add_block[i] = addblock
            
    #F4 - UPSAMPLE and ADD
    for i, k in reversed(list(enumerate(dn_add_block))):
        if i == len(dn_add_block) - 1:
            k1 = addblock #dn_add_block[i]
            k2 = input_block[i - 1]
            strides = k2.shape[1] // k.shape[1]
            upblock = tf.keras.layers.Conv2DTranspose(k2.shape[-1], KERNELS, strides=(strides, strides), padding="same", activation = end_activation, name = "fpn_%02d_upsample_block_%02d_Conv2DTranspose" % (sequence, i))(k1)
            upblock = tf.keras.layers.BatchNormalization(name = "fpn_%02d_upsample_%02d_BN" % (sequence, i))(upblock)
            upblock = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_upsample_%02d_Activation" % (sequence, i))(upblock)
            upsample_block[i - 1] = upblock
        elif i > 0:
            k2 = input_block[i - 1]
            strides = k2.shape[1] // k.shape[1]
            k1 = upblock
            addblock = tf.keras.layers.Add(name = "fpn_%02d_up_add_%02d" % (sequence, i))([input_block[i], k1, dn_add_block[i]])
            up_add_block[i] = addblock
            upblock = tf.keras.layers.Conv2DTranspose(k2.shape[-1], KERNELS, strides=(strides, strides), padding="same", activation = end_activation, name = "fpn_%02d_upsample_block_%02d_Conv2DTranspose" % (sequence, i))(addblock)
            upblock = tf.keras.layers.BatchNormalization(name = "fpn_%02d_upsample_%02d_BN" % (sequence, i))(upblock)
            upblock = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_upsample_%02d_Activation" % (sequence, i))(upblock)
            upsample_block[i - 1] = upblock
        else:
            k1 = upblock
            addblock = tf.keras.layers.Add(name = "fpn_%02d_up_add_%02d" % (sequence, i))([input_block[i], k1])
            up_add_block[i] = addblock

    #F5 - OUTPUT
    for i, k in enumerate(up_add_block):
        if i < len(up_add_block) - 1:
            c = tf.keras.layers.Dropout(dropout)(k)
        else:
            c = tf.keras.layers.Dropout(dropout)(dn_add_block[i])
        c = tf.keras.layers.BatchNormalization(name = "fpn_%02d_output_%02d_BN" % (sequence, i))(c)
        c = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_output_%02d_Activation" % (sequence, i))(c)            
        output_block[i] = c
    #return input_block, skip_block, downsample_block, dn_add_block, upsample_block, up_add_block, output_block
    return output_block
            

fl = tf.keras.losses.Huber(reduction = 'sum')
gl = tfa.losses.GIoULoss()
def regression_loss(y_true, y_pred):
    f = fl(y_true, y_pred, 1.0)
    g = gl(y_true, y_pred, 1.0)
    return g + f

classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()


def experiment_loss(L1, L2):
    print(L1, L2)
    
def checkFPN(model):
    for i in model.layers:
        if "fpn" in i.name:
            if type(i.input) == list:
                print([(n.name, n.shape) for n in i.input], ">>", i.output.name, i.output.shape)
            else:
                print([i.input.name], i.input.shape, ">>", i.output.name, i.output.shape)

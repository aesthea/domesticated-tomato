import tensorflow as tf
import tensorflow_addons as tfa
import layer
import copy

#V11 2024-11-12

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
    cur_shape = None
    has_add = False
    for i, j in enumerate(backbone.layers):
        #print(i, j.name, j.output.shape)
        if "add" in j.name.lower():
            has_add = True
            if j.output.shape != cur_shape:
                cur_shape = j.output.shape
                layer_block.append(j.output)
            else:
                layer_block[-1] = j.output
        if "project_bn" in j.name.lower() and has_add:
            if j.output.shape != cur_shape:
                cur_shape = j.output.shape
                layer_block.append(j.output)
            else:
                layer_block[-1] = j.output
                
    beforeFPN = []
    for i, j in enumerate(layer_block):
        print("layer block", i, j.name, j.shape)
        c = tf.keras.layers.Dropout(dropout)(j)
        beforeFPN.append(c)
        
    #BiFPN
    end_activation = tf.keras.layers.LeakyReLU()
    KERNELS = 3
    sequence = 3
    bn_axis = 3
    output_block = [[None for n in beforeFPN] for m in range(sequence)]
    #downsampling
    for seq in range(sequence):
        if seq == 0:
            input_block = beforeFPN
        else:
            input_block = output_block[seq - 1]
        for i, k in enumerate(input_block):
            if i < len(input_block) - 1:
                k2 = input_block[i + 1]
                strides = int(k.shape[1] // k2.shape[1])
                if i == 0:
                    carry_block = tf.keras.layers.Conv2D(filters=k2.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = end_activation, name = "fpn_%02d_downsample_%02d_Conv2D" % (seq, i))(k)
                    output_block[seq][i] = k
                else:
                    carry_block = tf.keras.layers.Add(name = "fpn_%02d_dn_add_%02d" % (seq, i))([k, carry_block])
                    output_block[seq][i] = carry_block
                    carry_block = tf.keras.layers.Conv2D(filters=k2.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = end_activation, name = "fpn_%02d_downsample_%02d_Conv2D" % (seq, i))(carry_block)
                    carry_block = tf.keras.layers.BatchNormalization(axis=bn_axis, name = "fpn_%02d_downsampling_%02d_BN" % (seq, i))(carry_block)
            else:
                carry_block = tf.keras.layers.Add(name = "fpn_%02d_dn_add_%02d" % (seq, i))([k, carry_block])
                to_output_block = tf.keras.layers.BatchNormalization(axis=bn_axis, name = "fpn_%02d_output_%02d_BN" % (seq, i))(carry_block)
                #to_output_block = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_output_%02d_Activation" % (seq, i))(to_output_block)
                output_block[seq][i] = to_output_block
        #upsampling
        output_block[seq].reverse()
        for ri, k in enumerate(output_block[seq]):
            i = len(output_block[seq]) - (ri + 1)
            if i > 0:
                if i < len(input_block) - 1:
                    carry_block = tf.keras.layers.Add(name = "fpn_%02d_up_add_%02d" % (seq, i))([input_block[i], carry_block, output_block[seq][ri]])
                    to_output_block = tf.keras.layers.BatchNormalization(axis=bn_axis, name = "fpn_%02d_output_%02d_BN" % (seq, i))(carry_block)
                    #to_output_block = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_output_%02d_Activation" % (seq, i))(to_output_block)
                    output_block[seq][ri] = to_output_block
                k2 = input_block[i - 1]
                strides = k2.shape[1] // k.shape[1]
                carry_block = tf.keras.layers.Conv2DTranspose(k2.shape[-1], KERNELS, strides=(strides, strides), padding="same", activation = end_activation, name = "fpn_%02d_upsample_%02d_Conv2DTranspose" % (seq, i))(carry_block)
                carry_block = tf.keras.layers.BatchNormalization(axis=bn_axis, name = "fpn_%02d_upsampling_%02d_BN" % (seq, i))(carry_block)
                carry_block = tf.keras.layers.Dropout(dropout)(carry_block)
            else:
                carry_block = tf.keras.layers.Add(name = "fpn_%02d_up_add_%02d" % (seq, i))([input_block[i], carry_block])
                to_output_block = tf.keras.layers.BatchNormalization(axis=bn_axis, name = "fpn_%02d_output_%02d_BN" % (seq, i))(carry_block)
                #to_output_block = tf.keras.layers.Activation(end_activation, name = "fpn_%02d_output_%02d_Activation" % (seq, i))(to_output_block)
                output_block[seq][ri] = carry_block
        output_block[seq].reverse()
    
    global_avg_pool = []
    for i, j in enumerate(output_block[-1]):        
        pooling = tf.keras.layers.GlobalAveragePooling2D(name = "global_average_pooling2d_%02d_%02d" % (j.shape[1], j.shape[3]))(j)
        global_avg_pool.append(pooling)
    r = tf.keras.layers.Concatenate(axis=-1)(global_avg_pool)
    r = tf.keras.layers.Dropout(dropout)(r)

    print(r.shape)

    b = tf.keras.layers.Dense(4 * detection_region, activation = "sigmoid", name = "dense_regression")(r)
    b = tf.keras.layers.Reshape([detection_region, 4], name = "reshape_regression")(b)
    regression = tf.keras.layers.Layer(name = "regression")(b)

    c = tf.keras.layers.Dense(num_classes * detection_region,  activation = "softmax", name = "dense_classification")(r)
    c = tf.keras.layers.Reshape([detection_region, num_classes], name = "reshape_classification")(c)
    classification = tf.keras.layers.Layer(name = "classification")(c)
    
    model = tf.keras.Model(x_in, [classification, regression])
    return model

fl = tf.keras.losses.Huber(reduction = 'sum')
gl = tfa.losses.GIoULoss()
def regression_loss(y_true, y_pred):
    f = fl(y_true, y_pred, 1.0)
    g = gl(y_true, y_pred, 1.0)
    return g + f

classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()

def checkFPN(model):
    for i in model.layers:
        if "fpn" in i.name:
            if type(i.input) == list:
                print([(n.name, n.shape) for n in i.input], ">>", i.output.name, i.output.shape)
            else:
                print([i.input.name], i.input.shape, ">>", i.output.name, i.output.shape)

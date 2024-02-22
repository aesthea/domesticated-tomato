
import tensorflow as tf
import tensorflow_addons as tfa

def conv_block(x, filters, strides, KERNELS, activation = 'relu'):
    x = tf.keras.layers.Conv2D(filters=filters, kernel_size=KERNELS, strides=strides, padding='same', activation = activation)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x

def res_block(x_in, KERNELS, strides = 1, activation = 'relu'):
    x = tf.keras.layers.Conv2D(filters=x_in.shape[-1], kernel_size=KERNELS, strides=strides, padding='same', activation = activation)(x_in)
    x = tf.keras.layers.Activation(activation)(x)

    x_add = tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=strides, padding='same')(x_in)
    x = tf.keras.layers.Add()([x, x_add])
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation(activation)(x)
    return x
    
def edet(input_size = 256, num_channels = 1, num_classes = 100, items = 3, dropout = 0.0, bi = 0, backbone = "B1", LSTM = False):
    print("input", input_size, "channel", num_channels, "classtags", num_classes, "region", items, "fpn mode", bi, "backbone", backbone)
    with tf.device("cpu:0"):
        x_in = tf.keras.Input(shape=[input_size, input_size, num_channels])

        if backbone == "B0":
            backbone = tf.keras.applications.efficientnet.EfficientNetB0(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(None, 256, 256, 1)
            p1 = backbone.layers[16].output #(None, 128, 128, 16)
            p2 = backbone.layers[45].output #(None, 64, 64, 24)
            p3 = backbone.layers[74].output #(None, 32, 32, 40)
            p4 = backbone.layers[118].output #(None, 16, 16, 80)
            p5 = backbone.layers[161].output #(None, 16, 16, 112)
            p6 = backbone.layers[220].output #(None, 8, 8, 192)
            p7 = backbone.layers[233].output #(None, 8, 8, 320)
        elif backbone == "B1":
            backbone = tf.keras.applications.efficientnet.EfficientNetB1(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output 
            p1 = backbone.layers[28].output
            p2 = backbone.layers[72].output 
            p3 = backbone.layers[116].output
            p4 = backbone.layers[175].output
            p5 = backbone.layers[233].output
            p6 = backbone.layers[277].output
            p7 = backbone.layers[335].output
        elif backbone == "B2":
            backbone = tf.keras.applications.efficientnet.EfficientNetB2(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
            p1 = backbone.layers[28].output #(28, 'block1b_add', TensorShape([None, 128, 128, 16]))
            p2 = backbone.layers[72].output #(72, 'block2c_add', TensorShape([None, 64, 64, 24]))
            p3 = backbone.layers[116].output #(116, 'block3c_add', TensorShape([None, 32, 32, 48]))
            p4 = backbone.layers[175].output #(175, 'block4d_add', TensorShape([None, 16, 16, 88]))
            p5 = backbone.layers[233].output #(233, 'block5d_add', TensorShape([None, 16, 16, 120]))
            p6 = backbone.layers[307].output #(307, 'block6e_add', TensorShape([None, 8, 8, 208]))
            p7 = backbone.layers[335].output #(335, 'block7b_add', TensorShape([None, 8, 8, 352]))
        elif backbone == "B3":
            backbone = tf.keras.applications.efficientnet.EfficientNetB3(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
            p1 = backbone.layers[28].output #(28, 'block1b_add', TensorShape([None, 128, 128, 24]))
            p2 = backbone.layers[72].output #(72, 'block2c_add', TensorShape([None, 64, 64, 32]))
            p3 = backbone.layers[116].output #(116, 'block3c_add', TensorShape([None, 32, 32, 48]))
            p4 = backbone.layers[190].output #(190, 'block4e_add', TensorShape([None, 16, 16, 96]))
            p5 = backbone.layers[263].output #(263, 'block5e_add', TensorShape([None, 16, 16, 136]))
            p6 = backbone.layers[352].output #(352, 'block6f_add', TensorShape([None, 8, 8, 232]))
            p7 = backbone.layers[380].output #(380, 'block7b_add', TensorShape([None, 8, 8, 384]))
        elif backbone == "B4":
            backbone = tf.keras.applications.efficientnet.EfficientNetB4(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
            p1 = backbone.layers[28].output #(28, 'block1b_add', TensorShape([None, 128, 128, 24]))
            p2 = backbone.layers[87].output #(87, 'block2d_add', TensorShape([None, 64, 64, 32]))
            p3 = backbone.layers[146].output #(146, 'block3d_add', TensorShape([None, 32, 32, 56]))
            p4 = backbone.layers[235].output #(235, 'block4f_add', TensorShape([None, 16, 16, 112]))
            p5 = backbone.layers[323].output #(323, 'block5f_add', TensorShape([None, 16, 16, 160]))
            p6 = backbone.layers[442].output #(442, 'block6h_add', TensorShape([None, 8, 8, 272]))
            p7 = backbone.layers[470].output #(470, 'block7b_add', TensorShape([None, 8, 8, 448]))
        elif backbone == "B5":
            backbone = tf.keras.applications.efficientnet.EfficientNetB5(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
            p1 = backbone.layers[40].output #(40, 'block1c_add', TensorShape([None, 128, 128, 24]))
            p2 = backbone.layers[114].output #(114, 'block2e_add', TensorShape([None, 64, 64, 40]))
            p3 = backbone.layers[188].output #(188, 'block3e_add', TensorShape([None, 32, 32, 64]))
            p4 = backbone.layers[292].output #(292, 'block4g_add', TensorShape([None, 16, 16, 128]))
            p5 = backbone.layers[395].output #(395, 'block5g_add', TensorShape([None, 16, 16, 176]))
            p6 = backbone.layers[529].output #(529, 'block6i_add', TensorShape([None, 8, 8, 304]))
            p7 = backbone.layers[572].output #(572, 'block7c_add', TensorShape([None, 8, 8, 512]))
        elif backbone == "B6":
            backbone = tf.keras.applications.efficientnet.EfficientNetB6(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
            p1 = backbone.layers[40].output #(40, 'block1c_add', TensorShape([None, 128, 128, 32]))
            p2 = backbone.layers[129].output #(129, 'block2f_add', TensorShape([None, 64, 64, 40]))
            p3 = backbone.layers[218].output #(218, 'block3f_add', TensorShape([None, 32, 32, 72]))
            p4 = backbone.layers[337].output #(337, 'block4h_add', TensorShape([None, 16, 16, 144]))
            p5 = backbone.layers[455].output #(455, 'block5h_add', TensorShape([None, 16, 16, 200]))
            p6 = backbone.layers[619].output #(619, 'block6k_add', TensorShape([None, 8, 8, 344]))
            p7 = backbone.layers[662].output #(662, 'block7c_add', TensorShape([None, 8, 8, 576]))
        elif backbone == "B7":
            backbone = tf.keras.applications.efficientnet.EfficientNetB7(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #(0, 'input_1', TensorShape([None, 256, 256, 3]))
            p1 = backbone.layers[52].output #(52, 'block1d_add', TensorShape([None, 128, 128, 32]))
            p2 = backbone.layers[156].output #(156, 'block2g_add', TensorShape([None, 64, 64, 48]))
            p3 = backbone.layers[260].output #(260, 'block3g_add', TensorShape([None, 32, 32, 80]))
            p4 = backbone.layers[409].output #(409, 'block4j_add', TensorShape([None, 16, 16, 160]))
            p5 = backbone.layers[557].output #(557, 'block5j_add', TensorShape([None, 16, 16, 224]))
            p6 = backbone.layers[751].output #(751, 'block6m_add', TensorShape([None, 8, 8, 384]))
            p7 = backbone.layers[809].output #(809, 'block7d_add', TensorShape([None, 8, 8, 640]))
        elif backbone == "V2S":
            backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #0 input_2 (None, 256, 256, 3)
            p1 = backbone.layers[13].output #13 block1b_add (None, 128, 128, 24)
            p2 = backbone.layers[39].output #39 block2d_add (None, 64, 64, 48)
            p3 = backbone.layers[65].output #65 block3d_add (None, 32, 32, 64)
            p4 = backbone.layers[153].output #153 block4f_add (None, 16, 16, 128)
            p5 = backbone.layers[286].output #286 block5i_add (None, 16, 16, 160)
            p6 = backbone.layers[494].output #494 block6n_add (None, 8, 8, 256)
            p7 = backbone.layers[509].output #509 block6o_add (None, 8, 8, 256)
        elif backbone == "V2B0":
            backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #0 input_2 (None, 256, 256, 3)
            p1 = backbone.layers[8].output #8 block1a_project_activation (None, 128, 128, 16)
            p2 = backbone.layers[20].output #20 block2b_add (None, 64, 64, 32)
            p3 = backbone.layers[32].output #32 block3b_add (None, 32, 32, 48)
            p4 = backbone.layers[75].output #75 block4c_add (None, 16, 16, 96)
            p5 = backbone.layers[148].output #148 block5e_add (None, 16, 16, 112)
            p6 = backbone.layers[251].output #251 block6g_add (None, 8, 8, 192)
            p7 = backbone.layers[266].output #266 block6h_add (None, 8, 8, 192)
        elif backbone == "V2B1":
            backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #0 input_2 (None, 256, 256, 3)
            p1 = backbone.layers[13].output #13 block1b_add (None, 128, 128, 16)
            p2 = backbone.layers[32].output #32 block2c_add (None, 64, 64, 32)
            p3 = backbone.layers[51].output #51 block3c_add (None, 32, 32, 48)
            p4 = backbone.layers[109].output #109 block4d_add (None, 16, 16, 96)
            p5 = backbone.layers[197].output #197 block5f_add (None, 16, 16, 112)
            p6 = backbone.layers[315].output #315 block6h_add (None, 8, 8, 192)
            p7 = backbone.layers[330].output #330 block6i_add (None, 8, 8, 192)
        elif backbone == "V2B2":
            backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #0 input_2 (None, 256, 256, 3)
            p1 = backbone.layers[13].output #13 block1b_add (None, 128, 128, 16)
            p2 = backbone.layers[32].output #32 block2c_add (None, 64, 64, 32)
            p3 = backbone.layers[51].output #51 block3c_add (None, 32, 32, 56)
            p4 = backbone.layers[109].output #109 block4d_add (None, 16, 16, 104)
            p5 = backbone.layers[197].output #197 block5f_add (None, 16, 16, 120)
            p6 = backbone.layers[330].output #330 block6i_add (None, 8, 8, 208)
            p7 = backbone.layers[345].output #345 block6j_add (None, 8, 8, 208)
        elif backbone == "V2B3":
            backbone = tf.keras.applications.efficientnet_v2.EfficientNetV2S(input_tensor = x_in, include_top = False, weights = None,input_shape = (input_size, input_size, num_channels), classes = num_classes)
            p0 = backbone.layers[0].output #0 input_2 (None, 256, 256, 3)
            p1 = backbone.layers[13].output #13 block1b_add (None, 128, 128, 16)
            p2 = backbone.layers[32].output #32 block2c_add (None, 64, 64, 40)
            p3 = backbone.layers[51].output #51 block3c_add (None, 32, 32, 56)
            p4 = backbone.layers[124].output #124 block4e_add (None, 16, 16, 112)
            p5 = backbone.layers[227].output #227 block5g_add (None, 16, 16, 136)
            p6 = backbone.layers[390].output #390 block6k_add (None, 8, 8, 232)
            p7 = backbone.layers[405].output #405 block6l_add (None, 8, 8, 232)
            
        o1, o2, o3, o4, o5 = FPN(p1, p2, p3, p5, p7, 3, 'relu', dropout, bi)
        o1, o2, o3, o4, o5 = FPN(o1, o2, o3, o4, o5, 3, 'relu', dropout, bi)
        c1, c2, c3, c4, c5 = FPN(o1, o2, o3, o4, o5, 3, 'relu', dropout, bi)

        L1 = tf.keras.layers.GlobalAveragePooling2D()(c1)
        L2 = tf.keras.layers.GlobalAveragePooling2D()(c2)
        L3 = tf.keras.layers.GlobalAveragePooling2D()(c3)
        L4 = tf.keras.layers.GlobalAveragePooling2D()(c4)
        L5 = tf.keras.layers.GlobalAveragePooling2D()(c5)

        r = tf.keras.layers.Concatenate(axis=-1)([L1, L2, L3, L4, L5])
        r = tf.keras.layers.Dropout(dropout)(r)

        b = tf.keras.layers.Dense(4 * items)(r)
        b = tf.keras.layers.Reshape([items, 4])(b)
        b = tf.keras.layers.Activation('sigmoid')(b)
        regression = tf.keras.layers.Layer(name = "regression")(b)

        c = tf.keras.layers.Dense(num_classes * items)(r)
        c = tf.keras.layers.Reshape([items, num_classes])(c)
        c = tf.keras.layers.BatchNormalization(axis=-1)(c)
        c = tf.keras.layers.Activation('softmax')(c)
        classification = tf.keras.layers.Layer(name = "classification")(c)
        
        model = tf.keras.Model(x_in, [classification, regression])
        return model


def FPN(i1, i2, i3, i4, i5, KERNELS = 3, end_activation = "relu", dropout = 0.2, bi = False):
    pool_size_2_1 = i1.shape[1] // i2.shape[1]
    pool_size_3_2 = i2.shape[1] // i3.shape[1]
    pool_size_4_3 = i3.shape[1] // i4.shape[1]
    pool_size_5_4 = i4.shape[1] // i5.shape[1]

    u2_1 = tf.keras.layers.Conv2DTranspose(i1.shape[-1], KERNELS, strides=(pool_size_2_1, pool_size_2_1), padding="same", activation = "relu")(i2)
    u3_2 = tf.keras.layers.Conv2DTranspose(i2.shape[-1], KERNELS, strides=(pool_size_3_2, pool_size_3_2), padding="same", activation = "relu")(i3)
    u4_3 = tf.keras.layers.Conv2DTranspose(i3.shape[-1], KERNELS, strides=(pool_size_4_3, pool_size_4_3), padding="same", activation = "relu")(i4)
    u5_4 = tf.keras.layers.Conv2DTranspose(i4.shape[-1], KERNELS, strides=(pool_size_5_4, pool_size_5_4), padding="same", activation = "relu")(i5)

    c1 = tf.keras.layers.Add()([i1, u2_1])
    c2 = tf.keras.layers.Add()([i2, u3_2])
    c3 = tf.keras.layers.Add()([i3, u4_3])
    c4 = tf.keras.layers.Add()([i4, u5_4])

    c1 = conv_block(c1, c1.shape[-1], 1, KERNELS, activation = 'relu')
    c2 = conv_block(c2, c2.shape[-1], 1, KERNELS, activation = 'relu')
    c3 = conv_block(c3, c3.shape[-1], 1, KERNELS, activation = 'relu')
    c4 = conv_block(c4, c4.shape[-1], 1, KERNELS, activation = 'relu')

    d2 = tf.keras.layers.Add()([i2, c2])
    d3 = tf.keras.layers.Add()([i3, c3])
    d4 = tf.keras.layers.Add()([i4, c4])

    if bi:
        u2 = conv_block(c1, i2.shape[-1], pool_size_2_1, KERNELS, activation = 'relu')
        u3 = conv_block(c2, i3.shape[-1], pool_size_3_2, KERNELS, activation = 'relu')
        u4 = conv_block(c3, i4.shape[-1], pool_size_4_3, KERNELS, activation = 'relu')
        if bi > 1:
            u5 = conv_block(c4, i5.shape[-1], pool_size_5_4, KERNELS, activation = 'relu')

        e2 = tf.keras.layers.Add()([i2, u2])
        e3 = tf.keras.layers.Add()([i3, u3])
        e4 = tf.keras.layers.Add()([i4, u4])
        if bi > 1:
            e5 = tf.keras.layers.Add()([i5, u5])

        o1 = conv_block(c1, c1.shape[-1], 1, KERNELS, activation = end_activation)
        o2 = conv_block(e2, e2.shape[-1], 1, KERNELS, activation = end_activation)
        o3 = conv_block(e3, e3.shape[-1], 1, KERNELS, activation = end_activation)
        o4 = conv_block(e4, e4.shape[-1], 1, KERNELS, activation = end_activation)
        if bi > 1:
            o5 = conv_block(e5, e5.shape[-1], 1, KERNELS, activation = end_activation)
        else:
            o5 = conv_block(i5, i5.shape[-1], 1, KERNELS, activation = end_activation)

    else:
        o1 = conv_block(c1, c1.shape[-1], 1, KERNELS, activation = end_activation)
        o2 = conv_block(d2, d2.shape[-1], 1, KERNELS, activation = end_activation)
        o3 = conv_block(d3, d3.shape[-1], 1, KERNELS, activation = end_activation)
        o4 = conv_block(i4, i4.shape[-1], 1, KERNELS, activation = end_activation)
        o5 = conv_block(i5, i5.shape[-1], 1, KERNELS, activation = end_activation)

    o1 = tf.keras.layers.Dropout(dropout)(o1)
    o2 = tf.keras.layers.Dropout(dropout)(o2)
    o3 = tf.keras.layers.Dropout(dropout)(o3)
    o4 = tf.keras.layers.Dropout(dropout)(o4)
    o5 = tf.keras.layers.Dropout(dropout)(o5)
    
    return o1, o2, o3, o4, o5

def regression_loss(y_true, y_pred):
    fl = tf.keras.losses.Huber()
    gl = tfa.losses.GIoULoss()
    f = fl(y_true, y_pred)
    g = gl(y_true, y_pred)
    return g + f

classification_loss = tf.keras.losses.SparseCategoricalCrossentropy()

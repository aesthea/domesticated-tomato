import tensorflow as tf

class bbiou(tf.keras.metrics.Metric):
    def __init__(self, name = 'bbiou', **kwargs):
        super(bbiou, self).__init__(**kwargs)
        self.tp = self.add_weight('tp', initializer = 'zeros')
        self.count = self.add_weight('count', initializer = 'zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = 1e-6
        yA = tf.reduce_max(tf.concat([[tf.reshape(y_true[..., 0], (-1, 1))], \
                                        [tf.reshape(y_pred[..., 0], (-1, 1))]], 0), 0)
        xA = tf.reduce_max(tf.concat([[tf.reshape(y_true[..., 1], (-1, 1))], \
                                        [tf.reshape(y_pred[..., 1], (-1, 1))]], 0), 0)
        yB = tf.reduce_min(tf.concat([[tf.reshape(y_true[..., 2], (-1, 1))], \
                                        [tf.reshape(y_pred[..., 2], (-1, 1))]], 0), 0)
        xB = tf.reduce_min(tf.concat([[tf.reshape(y_true[..., 3], (-1, 1))], \
                                        [tf.reshape(y_pred[..., 3], (-1, 1))]], 0), 0)
        interArea = (xB - xA + epsilon) * (yB - yA + epsilon)
        
        boxAArea = (tf.reshape(y_true[..., 2], (-1, 1)) - tf.reshape(y_true[..., 0], (-1, 1)) + epsilon) * \
                   (tf.reshape(y_true[..., 3], (-1, 1)) - tf.reshape(y_true[..., 1], (-1, 1)) + epsilon)
        boxBArea = (tf.reshape(y_pred[..., 2], (-1, 1)) - tf.reshape(y_pred[..., 0], (-1, 1)) + epsilon) * \
                   (tf.reshape(y_pred[..., 3], (-1, 1)) - tf.reshape(y_pred[..., 1], (-1, 1)) + epsilon)
        
        iou = tf.math.reduce_mean(interArea / (boxAArea + boxBArea - interArea))

        self.tp.assign_add(iou)
        #self.tp.assign(iou)
        self.count.assign_add(1)
        
    def reset_state(self):
        self.tp.assign(0)
        self.count.assign(0)

    @tf.function
    def result(self):
        epsilon = 1e-4
        res = self.tp / (self.count + 1.0)
        res = tf.cond(tf.greater(res, 1.0), lambda: 1.0, lambda: res)
        res = tf.cond(tf.greater(0.0, res), lambda: 0.0, lambda: res)
        return res

class custom_metric(tf.keras.metrics.Metric):
    def __init__(self, name = 'custom_metric', **kwargs):
        super(custom_metric, self).__init__(**kwargs)
        #self.m = tf.keras.metrics.MeanSquaredError()
        #self.m = SumSquaredError()
        self.m = bbiou()

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = 1e-6
##        mask = y_true[..., 2:3] - y_true[..., 0:1] < 1.0
##        mask = tf.reduce_all(tf.reduce_all(mask, -1), -1)
##        self.m.update_state(tf.boolean_mask(y_true, mask), tf.boolean_mask(y_pred, mask))
        self.m.update_state(y_true, y_pred)
        
    def reset_state(self):
        self.m.reset_state()

    def result(self):
        return self.m.result()

class SumSquaredError(tf.keras.metrics.Metric):
    def __init__(self, name='sum_squared_error', **kwargs):
        super(SumSquaredError, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name='total', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        errors = tf.square(y_true - y_pred)
        self.total.assign_add(tf.reduce_sum(errors))

    def result(self):
        return self.total

    def reset_state(self):
        self.total.assign(0.)


x0 = tf.constant([[[0,0,1,1],[0,0,1,1]],[[0.1,0.1,0.9,0.9],[0.1,0.1,0.9,0.9]],[[0,0,1,1],[0,0,1,1]],[[0.1,0.1,0.9,0.9],[0.1,0.1,0.9,0.9]]])
y0 = tf.constant([[[0,0,1,1],[0,0,1,1]],[[0.1,0.1,0.9,0.9],[0.1,0.1,0.9,0.9]],[[0,0,1,1],[0,0,1,1]],[[0.1,0.1,0.9,0.9],[0.1,0.1,0.9,0.9]]])
y1 = tf.constant([[[0.1,0.1,.1,.9],[0,0,1,1]],[[0.1,0.1,0.8,0.8],[0.1,0.1,0.9,0.9]],[[0,0,1,1],[0,0,1,0.8]],[[0.1,0.1,0.8,0.9],[0.1,0.1,0.9,0.9]]])
y2 = tf.constant([[[1,1,0,0],[1,1,0,0]],[[0.9,0.9,0.1,0.1],[0.9,0.9,0.1,0.1]],[[1,1,0,0],[1,1,0,0]],[[0.9,0.9,0.1,0.1],[0.9,0.9,0.1,0.1]]])

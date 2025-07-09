from keras import backend as K
def recall(y_true,y_pred):                          
    #y_true = K.ones_like(y_true)                              
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall = true_positives / (all_positives + K.epsilon())
    return recall

def precision(y_true,y_pred):
    #y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision
def dice(y_true,y_pred):
    #y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    dice = true_positives*2 / (predicted_positives + all_positives + K.epsilon())
    return dice
    #y_true = K.ones_like(y_true)
def f1(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def iou(y_true,y_pred):    
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))   
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))

    iou = true_positives / (predicted_positives + all_positives - true_positives + K.epsilon())
    return iou
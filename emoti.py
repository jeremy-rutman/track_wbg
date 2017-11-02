__author__ = 'jeremy'

import os
import numpy as np
import matplotlib.pyplot as plt
import caffe
from PIL import Image

categories = [ 'Angry' , 'Disgust' , 'Fear' , 'Happy'  , 'Neutral' ,  'Sad' , 'Surprise']
DEMO_DIR = '.'
cur_net_dir = '/data/jeremy/image_dbs/variant/hasner/VGG_S_rgb'

mean_filename=os.path.join(DEMO_DIR,cur_net_dir,'mean.binaryproto')
proto_data = open(mean_filename, "rb").read()
a = caffe.io.caffe_pb2.BlobProto.FromString(proto_data)
mean  = caffe.io.blobproto_to_array(a)[0][0]

mean = np.zeros([256,256],dtype=np.uint8)
print('mean shape {}'.format(mean.shape))

net_pretrained = os.path.join(DEMO_DIR,cur_net_dir,'EmotiW_VGG_S.caffemodel')
net_model_file = os.path.join(DEMO_DIR,cur_net_dir,'deploy.prototxt')

VGG_S_Net = caffe.Classifier(net_model_file, net_pretrained,
#                       mean=mean,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))


def emotion_det(image_file):
    input_image = caffe.io.load_image(image_file)
#xinput_image=input_image.resize((256,256), Image.ANTIALIAS)
    prediction = VGG_S_Net.predict([input_image],oversample=False)
    print 'predicted category is {0}'.format(categories[prediction.argmax()])
    print prediction[0]
#    return prediction[0]
    return categories[prediction.argmax()]





import tensorflow as tf
import data_utils
import run
import os
import cv2
import numpy as np
import pathlib
import argparse
from PIL import Image
import numpy
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # numbers
    parser.add_argument('--scale', type=int, help='Scaling factor of the model', default=2)


    # paths
    parser.add_argument('--input_type', help='input type is image or folder', default="folder")
    parser.add_argument('--input_dir', help='Path to input images')
    parser.add_argument('--output_dir', help='Path to store images')
    parser.add_argument('--ckpt_path', help='Path to model weight')
    args = parser.parse_args()

    # INIT
    scale = args.scale


    # Set gpu
    # os.environ["CUDA_VISIBLE_DEVICES"] = "4"
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    print("start Testing......")
    if args.input_type == "folder":
        testfiles = [f for f in os.listdir(args.input_dir)]
        for i in range(len(testfiles)):
            testfiles[i] = os.path.join(args.input_dir, testfiles[i])
        print('input_file: ', testfiles)
    else:
        testfiles = []
        testfiles.append(args.input_dir)
        print('input_file: ', testfiles)

    meanbgr = [91.26855492, 114.36102148, 123.20773159]
    print('Mean bgr: ', meanbgr)
    for filename in testfiles:
        fullimg = cv2.imread(filename, 3)
        img = cv2.resize(fullimg, (fullimg.shape[1]*args.scale, fullimg.shape[0]**args.scale), interpolation=cv2.INTER_CUBIC)
        width = img.shape[0]
        height = img.shape[1]

        cropped = img[0:(width - (width % args.scale)), 0:(height - (height % args.scale)), :]
        img = cv2.resize(cropped, None, fx=1. / args.scale, fy=1. / args.scale, interpolation=cv2.INTER_CUBIC)
        floatimg = img.astype(np.float32) - meanbgr

        LR_input_ = floatimg.reshape(1, floatimg.shape[0], floatimg.shape[1], 3)


        with tf.compat.v1.Session(config=config) as sess:
            # load the model
            ckpt_name = args.ckpt_path + "/" + "edsr_ckpt" + ".meta"
            saver = tf.compat.v1.train.import_meta_graph(ckpt_name)
            saver.restore(sess, tf.train.latest_checkpoint(args.ckpt_path))
            graph_def = sess.graph
            LR_tensor = graph_def.get_tensor_by_name("IteratorGetNext:0")
            HR_tensor = graph_def.get_tensor_by_name("NHWC_output:0")

            output = sess.run(HR_tensor, feed_dict={LR_tensor: LR_input_})

            Y = output[0]
            HR_image = (Y + meanbgr).clip(min=0, max=255)
            HR_image = (HR_image).astype(np.uint8)

            bicubic_image = cv2.resize(img, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_CUBIC)

            saveimg_path = args.output_dir + "/" + os.path.basename(filename)
            cv2.imwrite(saveimg_path , HR_image)

        sess.close()
        print('Testing done.')

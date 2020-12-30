from yolov4tool.utils import *
from yolov4tool.torch_utils import *
from yolov4tool.darknet2pytorch import Darknet
import cv2
import argparse

"""hyper parameters"""
use_cuda = False

def detect_cv2(cfgfile, weightfile, imgfile):
    import cv2
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

    num_classes = m.num_classes
    if num_classes == 20:
        namesfile = 'data/voc.names'
    elif num_classes == 80:
        namesfile = 'data/coco.names'
    else:
        namesfile = 'wheat_data/obj.names'
    class_names = load_class_names(namesfile)

    img = cv2.imread(imgfile)
    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)


    start = time.time()
    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
    finish = time.time()
    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

    width = img.shape[1]
    height = img.shape[0]
    ans = []

    b = boxes[0]
    for i in range(len(b)):
        t = []
        box = b[i]
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        w = x2 - x1
        h = y2 - y1
        t.append(box[4])
        t.append(x1)
        t.append(y1)
        t.append(w)
        t.append(h)
        ans.append(t)
    for i in ans:
        print(i)
    return ans



    # plot_boxes_cv2(img, boxes[0], savename='predictions.jpg', class_names=class_names)

def get_args():
    parser = argparse.ArgumentParser('Test your image or video by trained model.')
    parser.add_argument('-cfgfile', type=str, default='./cfg/yolov4.cfg',
                        help='path of cfg file', dest='cfgfile')
    parser.add_argument('-weightfile', type=str,
                        default='./checkpoints/Yolov4_epoch1.pth',
                        help='path of trained model.', dest='weightfile')
    parser.add_argument('-imgfile', type=str,
                        default='./data/mscoco2017/train2017/190109_180343_00154162.jpg',
                        help='path of your image file.', dest='imgfile')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = get_args()
    if args.imgfile:
        detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_imges(args.cfgfile, args.weightfile)
        # detect_cv2(args.cfgfile, args.weightfile, args.imgfile)
        # detect_skimage(args.cfgfile, args.weightfile, args.imgfile)
    else:
        detect_cv2_camera(args.cfgfile, args.weightfile)

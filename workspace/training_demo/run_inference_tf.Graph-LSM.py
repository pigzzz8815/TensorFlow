# -*- coding: cp949 -*-
import os
import time
import tensorflow as tf
tf.compat.v1.disable_v2_behavior()



PRETRAINED_MODEL_PATH = './pretrained_model'
file_list = os.listdir(PRETRAINED_MODEL_PATH)

model_list = {}
num_model = 1
for file in file_list:
    if 'tar.gz' in file:
        continue

    model_list[num_model] = file
    num_model += 1

print('[ model list ]')
for key, value in model_list.items():
    print(str(key) + '.', value)

user_input = int(input('사용할 모델을 고르세요 : '))
model = model_list[user_input]

start_time_first = time.perf_counter()
# PATH_TO_FROZEN_GRAPH = PRETRAINED_MODEL_PATH + '/' + model
PATH_TO_FROZEN_GRAPH = PRETRAINED_MODEL_PATH + '/' + model + '/frozen_inference_graph.pb'

detection_graph = tf.Graph()

with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()

    # 'rb'
    with tf.io.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as f:
        serialized_graph = f.read()
        od_graph_def.ParseFromString(serialized_graph)

        tf.import_graph_def(od_graph_def, name="")
ellpased_time1 = time.perf_counter()

grpah_load_time = (ellpased_time1 - start_time_first)

str999 = "Graph Load Time1 : %0.9f" % grpah_load_time
print(str999)
print('Graph Load Time2 : %0.9f" % grpah_load_time')

print('계산 그래프 로드 완료...')



import cv2
import numpy as np

iInference_cnt = 0
def run_inference_for_single_image(image, graph, sess):
    # with tf.compat.v1.Session(graph=graph) as sess:
        inferect_time1 = time.perf_counter()

        input_tensor = graph.get_tensor_by_name('image_tensor:0')

        target_operation_names = ['num_detections', 'detection_boxes',
                              'detection_scores', 'detection_classes', 'detection_masks']
        tensor_dict = {}
        for key in target_operation_names:
            op = None
            try:
                op = graph.get_operation_by_name(key)

            except:
                continue

            tensor = graph.get_tensor_by_name(op.outputs[0].name)
            tensor_dict[key] = tensor

        if 'detection_masks' in tensor_dict:
            detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])


        output_dict = sess.run(tensor_dict, feed_dict={input_tensor: [image]})
        # output_dict  = []
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        inferect_time2 = time.perf_counter()
        session_load_time = (inferect_time2 - inferect_time1)


        global iInference_cnt
        iInference_cnt += 1



        print("Inference time%d : %0.9f" % (iInference_cnt, session_load_time))

        return output_dict

        # str999 = "Inference time%d" % iInference_cnt
        # str999 = "Inference time%d : %0.9f" % (iInference_cnt, session_load_time)
        # print(str999)

def draw_bounding_boxes(img, output_dict, class_info):
    height, width, _ = img.shape

    obj_index = output_dict['detection_scores'] > 0.5

    scores = output_dict['detection_scores'][obj_index]
    boxes = output_dict['detection_boxes'][obj_index]
    classes = output_dict['detection_classes'][obj_index]

    for box, cls, score in zip(boxes, classes, scores):
        # draw bounding box
        img = cv2.rectangle(img,
                            (int(box[1] * width), int(box[0] * height)),
                            (int(box[3] * width), int(box[2] * height)), class_info[cls][1], 8)

        # put class name & percentage
        object_info = class_info[cls][0] + ': ' + str(int(score * 100)) + '%'
        text_size, _ = cv2.getTextSize(text=object_info,
                                       fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                       fontScale=0.9, thickness=2)
        img = cv2.rectangle(img,
                            (int(box[1] * width), int(box[0] * height) - 25),
                            (int(box[1] * width) + text_size[0], int(box[0] * height)),
                            class_info[cls][1], -1)
        img = cv2.putText(img,
                          object_info,
                          (int(box[1] * width), int(box[0] * height)),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    return img

PATH_TO_TEST_IMAGE = './images'
n_images = len(os.listdir(PATH_TO_TEST_IMAGE))

TEST_IMAGE_PATHS = [os.path.join(PATH_TO_TEST_IMAGE, 'jpg_image_{}.jpg'.format(i+1)) for i in range(n_images)]
print('분석 대상 이미지 경로 지정 완료...')

class_info = {}
f = open('class_info.txt', 'r')
for line in f:
    info = line.split(', ')

    class_index = int(info[0])
    class_name = info[1]
    color = (int(info[2][1:]), int(info[3]), int(info[4].strip()[:-1]))

    class_info[class_index] = [class_name, color]
f.close()

with tf.compat.v1.Session(graph=detection_graph) as sess:
    ellpased_time2 = time.perf_counter()
    session_load_time = (ellpased_time2 - ellpased_time1)

    str999 = "Session Load Time1 : %0.9f" % session_load_time
    str999 = "Session Load Time1 : %0.9f" % session_load_time
    print(str999)

    for image_path in TEST_IMAGE_PATHS:
        for_loop_time_start = time.perf_counter()
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)

        # start_time = time.perf_counter()

        output_dict = run_inference_for_single_image(img, detection_graph, sess)

        result = draw_bounding_boxes(img, output_dict, class_info)

        # end_time = time.perf_counter()

        for_loop_time_end = time.perf_counter()
        loop_time = (for_loop_time_end - for_loop_time_start)

        str99 = "For loop Time : %0.9f" % loop_time
        cv2.putText(img, str99, (1,32), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0))

        test = cv2.resize(result, (960, 1280))
        cv2.imshow(os.path.basename(image_path), test)

        if cv2.waitKey() & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            continue
        else:
            break

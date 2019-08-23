import cv2


# def detect_video(self, video_path):
#     cap = cv2.VideoCapture(video_path)
#     while (cap.isOpened()):
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         # pnet_boxes = self.__pnet_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#         # rnet_boxes = self.__rnet_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pnet_boxes)
#         # onet_boxes = self.__onet_detect(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), rnet_boxes)
#         # if onet_boxes is None:
#         #     cv2.imshow('Video', frame)
#         #     if cv2.waitKey(25) & 0xFF == ord('q'):
#         #         break
#         #     continue
#         # # Detected_image = self.draw(frame, onet_boxes)
#         # for i, box in enumerate(onet_boxes):
#         #     x1 = int(box[0])
#         #     y1 = int(box[1])
#         #     x2 = int(box[2])
#         #     y2 = int(box[3])
#         #     font = cv2.FONT_HERSHEY_COMPLEX
#         #     cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=1)
#         #     cv2.putText(frame, str(box[4]), (x1, y1 - 8), font, 0.3, (255, 0, 0), 1)
#         cv2.imshow('Video', frame)
#         cv2.waitKey(1)
#         if cv2.waitKey(40) & 0xFF == ord('q'):
#             break
#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()
#
# detect_video(r"G:\CloudMusic\MV\testvi.mp4")


import cv2

capture = cv2.VideoCapture(r"G:\CloudMusic\MV\videoplayback.mp4")

if capture.isOpened():
    while True:
        ret, prev = capture.read()
        if ret == True:
            cv2.imshow('video', prev)
        else:
            break
        print(cv2.waitKey(20))
        if cv2.waitKey(40) == 27:
            break
cv2.destroyAllWindows()

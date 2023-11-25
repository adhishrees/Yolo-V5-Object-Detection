import torch
import cv2

class YoloV5N6:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5n6.pt')
        self.size = 416
        self.color = (0, 0, 255)
    
    def vision(self):
        cap = cv2.VideoCapture(0)
        while True:
            ret, img = cap.read()
            results = self.model(img, self.size)
            for index, row in results.pandas().xyxy[0].iterrows():
                x1 = int(row['xmin'])
                x2 = int(row['xmax'])
                y1 = int(row['ymin'])
                y2 = int(row['ymax'])
                d = row['class']
                
                cv2.rectangle(img, (x1, y1), (x2, y2), self.color, 2)
                rectx1, recty1 = (x1 + x2) / 2, (y1 + y2) / 2
                rectcenter = int(rectx1), int(recty1)
                cv2.putText(img, str(self.model.names[d]), (x1, y1), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

            cv2.imshow("IMG", img)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            
        cap.release()
        cv2.destroyAllWindows()

if '__main__' == __name__:
    yolo = YoloV5N6()
    yolo.vision()
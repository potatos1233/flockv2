import cv2
import os
import time
import math
import threading
import urllib.request
import numpy as np
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, render_template_string

try:
    from ultralytics import YOLO
    from paddleocr import PaddleOCR
    LIBRARIES_AVAILABLE = True
except ImportError:
    LIBRARIES_AVAILABLE = False
    print("Missing libraries! Please run: pip install ultralytics paddleocr paddlepaddle flask")

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False
    print("Warning: yt-dlp not installed. YouTube links will not work.\nRun: pip install yt-dlp")

app = Flask(__name__)

class CentroidTracker:
    def __init__(self, max_disappeared=40, max_distance=60):
        self.next_id = 0
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.paths = {}

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.paths[self.next_id] = [centroid]
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        if object_id in self.paths:
            del self.paths[object_id]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects, self.paths

        new_centroids = []
        for (x, y, w, h) in rects:
            cx = x + w // 2
            cy = y + h // 2
            new_centroids.append((cx, cy))

        if len(self.objects) == 0:
            for c in new_centroids:
                self.register(c)
        else:
            object_ids = list(self.objects.keys())
            old_centroids = list(self.objects.values())

            dist_matrix = []
            for oc in old_centroids:
                row = []
                for nc in new_centroids:
                    d = math.dist(oc, nc)
                    row.append(d)
                dist_matrix.append(row)

            used_rows = set()
            used_cols = set()
            pairs = []
            for r in range(len(object_ids)):
                for c in range(len(new_centroids)):
                    pairs.append((dist_matrix[r][c], r, c))
            pairs.sort()

            for dist, r, c in pairs:
                if r in used_rows or c in used_cols: continue
                if dist > self.max_distance: continue
                object_id = object_ids[r]
                self.objects[object_id] = new_centroids[c]
                self.disappeared[object_id] = 0
                self.paths[object_id].append(new_centroids[c])
                if len(self.paths[object_id]) > 40:
                    self.paths[object_id] = self.paths[object_id][-40:]
                used_rows.add(r)
                used_cols.add(c)

            unused_rows = set(range(len(object_ids))) - used_rows
            unused_cols = set(range(len(new_centroids))) - used_cols

            for r in unused_rows:
                oid = object_ids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for c in unused_cols:
                self.register(new_centroids[c])

        return self.objects, self.paths

class FeedProcessor:
    def __init__(self, feed_id, source, cookie_source, yolo, face_net, ocr, save_dir):
        self.feed_id = feed_id
        self.source = source
        self.cookie_source = cookie_source
        self.yolo = yolo
        self.face_net = face_net
        self.ocr = ocr
        self.save_dir = save_dir
        self.session_dir = save_dir / f"feed{feed_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_dir.mkdir(exist_ok=True)
        self.track_dir = self.session_dir / "crops"
        self.track_dir.mkdir(exist_ok=True)
        self.text_dir = self.session_dir / "texts"
        self.text_dir.mkdir(exist_ok=True)

        self.tracker_people = CentroidTracker(max_disappeared=40, max_distance=60)
        self.tracker_cars = CentroidTracker(max_disappeared=40, max_distance=80)
        self.logged_ids = set()
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        
        try:
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            self.trained_ids = set()
            self.training_data = []
            self.training_labels = []
            self.learning_enabled = True
        except AttributeError:
            print("Warning: opencv-contrib-python not installed. Face learning disabled.\nRun: pip install opencv-contrib-python")
            self.learning_enabled = False

        self.cap = self.get_cap()
        if not self.cap or not self.cap.isOpened():
            print(f"Error: Could not open source for Feed {feed_id}")
            self.running = False

    def get_cap(self):
        if self.source.startswith("http") or self.source.startswith("www.") or self.source.startswith("youtube"):
            if not YT_DLP_AVAILABLE: return None
            try:
                ydl_opts = {'format': 'best[ext=mp4][height<=720]/best[height<=720]/best', 'quiet': True, 'no_warnings': True}
                if self.cookie_source != "None":
                    ydl_opts['cookiesfrombrowser'] = (self.cookie_source,)
                    
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(self.source, download=False)
                    stream_url = info['url']
                return cv2.VideoCapture(stream_url)
            except Exception as e:
                print(f"Stream Error: {e}")
                return None
        else:
            return cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)

    def draw_corner_rect(self, img, pt1, pt2, color, thickness, length=15):
        x1, y1 = pt1
        x2, y2 = pt2
        cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
        cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
        cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
        cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
        cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
        cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
        cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
        cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

    def draw_label(self, img, text, pt, color):
        x, y = pt
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(img, (x, y - h - 10), (x + w + 10, y), color, -1)
        cv2.putText(img, text, (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def get_color_for_id(self, track_id):
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255),
            (255, 255, 0), (0, 255, 255), (255, 0, 255),
            (255, 128, 0), (128, 0, 255), (0, 128, 255), (128, 255, 0)
        ]
        return colors[track_id % len(colors)]

    def save_face(self, frame, x, y, w, h, track_id):
        pad = 15
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0: return
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = self.track_dir / f"face_ID{track_id}_{timestamp}.jpg"
        cv2.imwrite(str(filepath), face_img)

        # TRAINING LOGIC: Sketch the face and update the brain
        if self.learning_enabled and track_id not in self.trained_ids:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face_roi = gray[y1:y2, x1:x2]
            if face_roi.size > 0:
                try:
                    face_roi_resized = cv2.resize(face_roi, (100, 100))
                    self.training_data.append(face_roi_resized)
                    self.training_labels.append(track_id)
                    self.trained_ids.add(track_id)
                    if len(self.training_data) > 1:
                        self.recognizer.update(self.training_data, np.array(self.training_labels))
                        print(f"Brain updated: Studied ID {track_id}")
                except: pass

    def save_car(self, frame, x, y, w, h, track_id):
        pad = 10
        h_img, w_img = frame.shape[:2]
        x1, y1 = max(0, x - pad), max(0, y - pad)
        x2, y2 = min(w_img, x + w + pad), min(h_img, y + h + pad)
        car_img = frame[y1:y2, x1:x2]
        if car_img.size == 0: return
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = self.track_dir / f"car_ID{track_id}_{timestamp}.jpg"
        cv2.imwrite(str(filepath), car_img)

    def save_plate_text(self, track_id, text):
        timestamp = datetime.now().strftime("%H%M%S")
        filepath = self.text_dir / f"plate_ID{track_id}_{timestamp}.txt"
        with open(filepath, "w") as f:
            f.write(f"ID: {track_id}\nPlate: {text}\nTime: {timestamp}\n")

    def process(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break

            display = frame.copy()
            (h, w) = frame.shape[:2]
            
            results = self.yolo(frame, classes=[0, 2, 5, 7], verbose=False)
            people, cars = [], []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls = int(box.cls[0])
                    bw, bh = x2 - x1, y2 - y1
                    if bw > 30 and bh > 30:
                        if cls == 0:
                            if bh > bw * 0.8: people.append((x1, y1, bw, bh))
                        else:
                            cars.append((x1, y1, bw, bh))

            # people
            people_objects, people_paths = self.tracker_people.update(people)
            id_to_person = {}
            for oid, centroid in people_objects.items():
                best_rect, best_dist = None, 99999
                for (px, py, pw, ph) in people:
                    cx, cy = px + pw // 2, py + ph // 2
                    d = math.dist((cx, cy), centroid)
                    if d < best_dist and d < 60:
                        best_dist, best_rect = d, (px, py, pw, ph)
                if best_rect: id_to_person[oid] = best_rect

            for oid, centroid in people_objects.items():
                color = self.get_color_for_id(oid)
                if oid in id_to_person:
                    (px, py, pw, ph) = id_to_person[oid]
                    if len(people_paths.get(oid, [])) >= 5:
                        self.draw_corner_rect(display, (px, py), (px + pw, py + ph), color, 2, 15)
                        label = f"Person ID: {oid}"
                        if oid not in self.logged_ids:
                            top_half = frame[py : py + int(ph/2), px : px + pw]
                            if top_half.size > 0:
                                blob = cv2.dnn.blobFromImage(cv2.resize(top_half, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
                                self.face_net.setInput(blob)
                                detections = self.face_net.forward()
                                for i in range(detections.shape[2]):
                                    if detections[0, 0, i, 2] > 0.4:
                                        box = detections[0, 0, i, 3:7] * np.array([pw, int(ph/2), pw, int(ph/2)])
                                        (fx, fy, fx2, fy2) = box.astype("int")
                                        abs_fx, abs_fy = px + fx, py + fy
                                        abs_fx2, abs_fy2 = px + fx2, py + fy2
                                        cv2.rectangle(display, (abs_fx, abs_fy), (abs_fx2, abs_fy2), color, 1)
                                        label = f"Face ID: {oid}"
                                        self.save_face(frame, abs_fx, abs_fy, abs_fx2-abs_fx, abs_fy2-abs_fy, oid)
                                        self.logged_ids.add(oid)
                                        break
                        self.draw_label(display, label, (px, py), color)

            for oid, path_points in people_paths.items():
                if oid not in people_objects or len(path_points) < 5: continue
                color = self.get_color_for_id(oid)
                for i, pt in enumerate(path_points):
                    if i % 3 == 0:
                        alpha = (i + 1) / len(path_points)
                        cv2.circle(display, pt, int(2 + alpha * 4), color, -1)

            # cars 
            car_objects, car_paths = self.tracker_cars.update(cars)
            id_to_car = {}
            for oid, centroid in car_objects.items():
                best_rect, best_dist = None, 99999
                for (cx1, cy1, cw, ch) in cars:
                    cx, cy = cx1 + cw // 2, cy1 + ch // 2
                    d = math.dist((cx, cy), centroid)
                    if d < best_dist and d < 80:
                        best_dist, best_rect = d, (cx1, cy1, cw, ch)
                if best_rect: id_to_car[oid] = best_rect

            for oid, centroid in car_objects.items():
                color = self.get_color_for_id(oid + 100)
                if oid in id_to_car:
                    (cx1, cy1, cw, ch) = id_to_car[oid]
                    if len(car_paths.get(oid, [])) >= 5:
                        self.draw_corner_rect(display, (cx1, cy1), (cx1 + cw, cy1 + ch), color, 2, 15)
                        label = f"Car ID: {oid}"
                        if oid not in self.logged_ids:
                            self.save_car(frame, cx1, cy1, cw, ch, oid)
                            self.logged_ids.add(oid)
                            crop = frame[cy1:cy1+ch, cx1:cx1+cw]
                            if crop.size > 0:
                                try: ocr_result = self.ocr.ocr(crop, cls=True)
                                except: ocr_result = None
                                if ocr_result and ocr_result[0]:
                                    for line in ocr_result[0]:
                                        text = line[1][0].strip()
                                        if len(text) > 3:
                                            label += f" | {text}"
                                            self.save_plate_text(oid, text)
                                            break
                        self.draw_label(display, label, (cx1, cy1), color)

            for oid, path_points in car_paths.items():
                if oid not in car_objects or len(path_points) < 5: continue
                color = self.get_color_for_id(oid + 100)
                for i, pt in enumerate(path_points):
                    if i % 3 == 0:
                        alpha = (i + 1) / len(path_points)
                        cv2.circle(display, pt, int(2 + alpha * 5), color, -1)

            ts = datetime.now().strftime("%H:%M:%S")
            cv2.putText(display, f"Feed {self.feed_id} | {ts}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            with self.lock:
                self.frame = display

        if self.cap: self.cap.release()

    def get_frame(self):
        with self.lock:
            if self.frame is None:
                return None
            ret, jpeg = cv2.imencode('.jpg', self.frame)
            return jpeg.tobytes()

# flask shit idk what im fucking
processors = {}

@app.route('/')
def index():
    num_feeds = len(processors)
    # Calculate grid layout (e.g., 1x1, 2x1, 2x2, 3x2, 3x3)
    cols = math.ceil(math.sqrt(num_feeds))
    rows = math.ceil(num_feeds / cols)
    
    return render_template_string('''
    <!DOCTYPE html>
    <html>
    <head>
        <title>potatos1233</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            html, body { width: 100vw; height: 100vh; background: #000; overflow: hidden; }
            .grid-container {
                display: grid;
                width: 100%;
                height: 100%;
                grid-template-columns: repeat({{ cols }}, 1fr);
                grid-template-rows: repeat({{ rows }}, 1fr);
                gap: 2px; /* A tiny gap to separate the monitors */
                background: #111;
            }
            .feed-cell {
                width: 100%;
                height: 100%;
                background: #000;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }
            .feed {
                max-width: 100%;
                max-height: 100%;
                object-fit: contain; /* Adapts to video size without cropping! */
                display: block;
            }
        </style>
    </head>
    <body>
        <div class="grid-container">
            {% for i in range(1, num_feeds + 1) %}
            <div class="feed-cell">
                <img class="feed" src="/video_feed/{{ i }}">
            </div>
            {% endfor %}
        </div>
    </body>
    </html>
    ''', num_feeds=num_feeds, cols=cols, rows=rows)

@app.route('/video_feed/<int:feed_id>')
def video_feed(feed_id):
    def generate(feed_id):
        while True:
            processor = processors.get(feed_id)
            if not processor or not processor.running:
                # Serve a black frame if feed is offline/loading
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, f"Feed {feed_id} Offline", (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
                ret, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')
                time.sleep(1)
                continue
                
            frame = processor.get_frame()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
            time.sleep(0.03)
    return Response(generate(feed_id), mimetype='multipart/x-mixed-replace; boundary=frame')

def main():
    if not LIBRARIES_AVAILABLE:
        return

    print("\potatos1233")
    num_feeds_str = input("How many feeds do you want? (e.g., 1, 2, 4, 6, 9): ").strip()
    try:
        num_feeds = int(num_feeds_str)
        if num_feeds < 1: num_feeds = 1
    except:
        num_feeds = 1

    cookie_source = input("enter browser for cookies").strip()
    if not cookie_source:
        cookie_source = "None"

    sources = []
    for i in range(num_feeds):
        print(f"\nFeed {i+1} source (Youtube URL, file path, or webcam number):")
        src = input("> ").strip()
        sources.append(src)

    print("\nwaking up models.")
    yolo = YOLO('yolov8n.pt')
    
    # Load face model
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    proto_path = model_dir / "deploy.prototxt"
    model_path = model_dir / "res10_300x300_ssd_iter_140000.caffemodel"
    if not proto_path.exists():
        url = "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt"
        urllib.request.urlretrieve(url, proto_path)
    if not model_path.exists():
        url = "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel"
        urllib.request.urlretrieve(url, model_path)
    face_net = cv2.dnn.readNetFromCaffe(str(proto_path), str(model_path))

    try: ocr = PaddleOCR(use_angle_cls=True, lang='en')
    except TypeError: ocr = PaddleOCR(lang='en')

    save_dir = Path("detected_items")
    save_dir.mkdir(exist_ok=True)

    global processors
    for i, src in enumerate(sources):
        feed_id = i + 1
        processors[feed_id] = FeedProcessor(feed_id, src, cookie_source, yolo, face_net, ocr, save_dir)
        threading.Thread(target=processors[feed_id].process, daemon=True).start()

    print("\nstarted")
    print("http://127.0.0.1:5000")
    print("CTRL+C to stop.\n")

    app.run(host='0.0.0.0', port=5000, threaded=True)

if __name__ == "__main__":
    main()
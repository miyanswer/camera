import cv2
import numpy as np

# YOLOの設定ファイルとモデルのパス
model_cfg = "yolov3-tiny.cfg"
model_weights = "yolov3-tiny.weights"
labels_path = "coco.names"

# ラベルの読み込み
with open(labels_path, "r") as f:
    labels = f.read().strip().split("\n")

# YOLOモデルをロード
net = cv2.dnn.readNetFromDarknet(model_cfg, model_weights)

# 使用する出力層を取得
layer_names = net.getLayerNames()
# 出力レイヤーのインデックスを取得
try:
    # OpenCV 4.x の場合
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
except TypeError:
    # OpenCV 4.5.3 以降の仕様変更の場合
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

# カメラを初期化
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("カメラを開くことができませんでした")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("フレームを取得できませんでした")
        break

    # フレームをリサイズして解像度を下げる
    resized_frame = cv2.resize(frame, (320, 320))

    # フレームのサイズを取得
    (H, W) = resized_frame.shape[:2]

    # フレームをBlobに変換
    blob = cv2.dnn.blobFromImage(resized_frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    # 推論
    layer_outputs = net.forward(output_layers)

    # 結果を解析
    boxes = []
    confidences = []
    class_ids = []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:  # 信頼度の閾値
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # Non-Maxima Suppressionを適用
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # 検出結果を描画
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in np.random.randint(0, 255, size=(3,))]
            cv2.rectangle(resized_frame, (x, y), (x + w, y + h), color, 2)
            text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.putText(resized_frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # フレームを表示
    cv2.imshow("Object Detection", resized_frame)

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# リソースを解放
cap.release()
cv2.destroyAllWindows()

import cv2
import mediapipe as mp

# MediaPipe Handsの初期化
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

# カメラキャプチャの設定
cap = cv2.VideoCapture(0)

# MediaPipe Handsのオブジェクトを作成
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("カメラ映像が取得できません。")
            break

        # BGRをRGBに変換
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # 手を検出
        results = hands.process(image)

        # BGRに戻す（描画のため）
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 画面の中心と四角形を描画
        h, w, _ = image.shape
        center_x, center_y = w // 2, h // 2
        rect_size = 80  # 四角形のサイズ
        top_left = (center_x - rect_size // 2, center_y - rect_size // 2)
        bottom_right = (center_x + rect_size // 2, center_y + rect_size // 2)

        cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)

        # 手のランドマークを取得してグッドジェスチャーを判定
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # ランドマーク座標をピクセルに変換
                landmarks = [
                    (int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark
                ]

                # 親指（ランドマーク4）と人差し指の付け根（ランドマーク2）を比較
                thumb_tip = landmarks[4]
                thumb_base = landmarks[2]

                # "グッド" ジェスチャー判定
                if (
                    thumb_tip[1] < thumb_base[1]  # 親指が上向き
                    and top_left[0] <= thumb_tip[0] <= bottom_right[0]  # 四角形のX範囲内
                    and top_left[1] <= thumb_tip[1] <= bottom_right[1]  # 四角形のY範囲内
                ):
                    cv2.putText(
                        image,
                        "Good!",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                    )
                    print("good")

        # 画像を表示
        cv2.imshow('Hand Gesture Recognition', image)

        # 'q'キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# リソースを解放
cap.release()
cv2.destroyAllWindows()

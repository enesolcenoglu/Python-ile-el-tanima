import cv2
import mediapipe as mp
import numpy as np
import random 

cap = cv2.VideoCapture(0)
mp_cizim = mp.solutions.drawing_utils
mp_butunsel = mp.solutions.holistic

# Beyaz bir pencere oluşturmak için bir matris oluşturun
fal_arkaplan = np.zeros((100, 1080), dtype=np.uint8)
fal_arkaplan.fill(255)

fal = ["Gokyuzune bak, umut dolu kal. icindeki gucu kesfet, hayatin sana sunacagi guzellikleri yakala. Ilerlemeye devam et, basari seninle olsun!"
       ,"Hayatta basari, icindeki tutkulari takip etmekle baslar. Bugun cesaretinle adim at, hayallerinin pesinden git."
       ,"Bugun hayatinin donum noktasina yaklasiyorsun. Yeni bir baslangica adim atmak icin cesaretin ve kararliligin onemli olacak."
       ,"Bugun, icsel rehberligin ve sezgilerin sana dogru yolu gosterecek. ihtiyacin olan ilhami bulmak icin icine don ve kalbinin sesini dinle."]

rastgele=random.choice(fal)

# Dikdörtgenin başlangıç koordinatları
x, y = 235, 110

# Dikdörtgenin boyutu
w, h = 200, 325

# Mavi renk kodu
color = (255, 0, 0)

with mp_butunsel.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as butun:
    while True:
        kontrol, resim = cap.read()
        resim_rgb = cv2.cvtColor(resim, cv2.COLOR_BGR2RGB)

        # Orijinal resim üzerine dikdörtgen çizimi
        cv2.rectangle(resim_rgb, (x, y), (x + w, y + h), color, thickness=2)

        # Maske resmini oluşturun ve dikdörtgen dışındaki alanları beyazlaştırın
        mask = np.zeros_like(resim_rgb)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Maskeyi uygulayarak dikdörtgen dışındaki alanları bulanıklaştırın
        blurred = cv2.GaussianBlur(resim_rgb, (51, 51), 0)
        resim_rgb = np.where(mask == np.array([255, 255, 255]), resim_rgb, blurred)
        
        sonuc = butun.process(resim_rgb)
        resim = cv2.cvtColor(resim_rgb, cv2.COLOR_RGB2BGR)
        mp_cizim.draw_landmarks(resim, sonuc.left_hand_landmarks, mp_butunsel.HAND_CONNECTIONS)
        if sonuc.left_hand_landmarks is not None:
           # Sol el tespit edildiğinde rastgele bir fal öğesi göster
           cv2.putText(fal_arkaplan, rastgele, (15, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
           if cv2.waitKey(20) == 27:
                break
           cv2.imshow("Mesaj",fal_arkaplan)

        if cv2.waitKey(20) == 27:
            break
        cv2.imshow("Pencere", resim)

cap.release()
cv2.destroyAllWindows()

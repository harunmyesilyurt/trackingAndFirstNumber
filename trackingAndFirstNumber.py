import cv2
import numpy as np

# Görüntüyü yükle
img_path = "img/duzsagikiyol.jpeg"
img = cv2.imread(img_path)

# Adım 1: Görüntüyü gri tonlamalıya çevir
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Adım 2: Gaussian Blur uygula
gaus = cv2.GaussianBlur(gray, (5, 5), 0, borderType=cv2.BORDER_DEFAULT)

# Adım 3: OTSU yöntemi ile eşikleme uygula
ret, otsu = cv2.threshold(gaus, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Adım 4: Canny kenar algılama uygula
edge = cv2.Canny(otsu, 100, 230)

# Adım 5: Kontur bulma
contours, _ = cv2.findContours(otsu, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Kontur analizi ve numaralandırma
direction_threshold = 45  # Derece cinsinden yön değiştirme eşiği


def find_angle(p1, p2, p3):
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


# Adım 6: Konturları analiz et ve numaralandır
path_id = 1
for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.02 * cv2.arcLength(contour, True), True)

    if len(approx) > 2:  # En az 3 nokta içermeli
        for i in range(len(approx)):
            p1 = approx[i % len(approx)][0]
            p2 = approx[(i + 1) % len(approx)][0]
            p3 = approx[(i + 2) % len(approx)][0]

            angle = find_angle(p1, p2, p3)

            if angle < direction_threshold or angle > 180 - direction_threshold:
                cv2.putText(img, f"Yol {path_id}", (p2[0], p2[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                path_id += 1
                break

# Adım 7: Konturları çiz ve numaralandır
for i, contour in enumerate(contours):
    cv2.drawContours(img, [contour], -1, (0, 255, 0), 2)
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.putText(img, f"Yol {i + 1}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

# Adım 8: Perspektif dönüşümü uygula
k1 = np.float32([[5, 250], [605, 250], [5, 400], [605, 400]])
k2 = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
M_perspective = cv2.getPerspectiveTransform(k1, k2)
perspektif = cv2.warpPerspective(img, M_perspective, (640, 480))

# Sonuçları göster
cv2.imshow("Orijinal Görüntü", img)
# cv2.imshow("Gaussian Blur", gaus)
cv2.imshow("OTSU Eşikleme", otsu)
cv2.imshow("Canny Kenar Algılama", edge)
# cv2.imshow("Perspektif Dönüşüm", perspektif)
cv2.waitKey(0)
cv2.destroyAllWindows()

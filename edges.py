import cv2 


img = cv2.imread("spec/noise/Bl122_58387343.jpg")

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 190, 200)

cv2.imwrite("edges.jpg", edges)
cv2.imwrite("audio.jpg", gray)

cv2.imshow("bl122", img)
cv2.imshow("Edges", edges)

cv2.waitKey(0)
cv2.destroyAllWindows()
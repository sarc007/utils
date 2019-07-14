import pytesseract
import cv2
from PIL import Image, ImageFilter,ImageOps
img = cv2.imread("img_5.png") # abcdefghijklmnopqrstuvwxyz

text = pytesseract.image_to_string(img, config="-c tessedit"
                                             "_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                                             " --psm 10"
                                             " -l osd"
                                             " ")
print(text)

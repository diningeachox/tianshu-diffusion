import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2

#Set image background
b,g,r,a = 0,0,0,0

fontpath = "./fonts/NotoSerifSC-Bold.otf" # Choose the font
font = ImageFont.truetype(fontpath, 28)

# Range of unicode representations for Chinese characters
start = 0x4e00
end = 0x9fa6

#Draw every single chinese character
i = 0
for x in range(start, end):
    img = np.ones((32,32,3),np.uint8) * 255
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    draw.text((2, -5),  chr(x), font = font, fill = (b, g, r, a))
    print(f'Generating char_{i}.png')
    img = np.array(img_pil)
    cv2.imwrite(f'./images/char_{i}.png', img)
    i += 1

# img = np.zeros((32,32,3),np.uint8)
# img_pil = Image.fromarray(img)
# draw = ImageDraw.Draw(img_pil)
# draw.text((2, -5),  chr(0x4e00), font = font, fill = (b, g, r, a))
# img = np.array(img_pil)
#
# #print("\uac01")
#
# cv2.imshow("res", img);cv2.waitKey();cv2.destroyAllWindows()

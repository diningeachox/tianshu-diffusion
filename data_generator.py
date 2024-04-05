import numpy as np
from PIL import ImageFont, ImageDraw, Image
import cv2

#Set image background
b,g,r,a = 0,0,0,0

fontpath = "./fonts/NotoSerifSC-Bold.otf" # Choose the font
font = ImageFont.truetype(fontpath, 60)

# Range of unicode representations for Chinese characters
start = 0x4e00
end = 0x9fa6

# start = 0x554a
# end = 0x57c4

# Draw every single chinese character and save the images
# i = 0
# for x in range(start, end):
#     img = np.ones((64,64,3),np.uint8) * 255
#     img_pil = Image.fromarray(img)
#     draw = ImageDraw.Draw(img_pil)
#     draw.text((2, -15),  chr(x), font = font, fill = (b, g, r, a))
#     print(f'Generating char_{i}.png')
#     img = np.array(img_pil)
#     cv2.imwrite(f'./images/char_{i}.png', img)
#     i += 1

#Draw images of characters from the data.txt
i = 0
with open("chars.txt", encoding='utf8') as file:
    while True:
        line = file.readline()
        if len(line) == 0:
            break
        char = line[:1]
        img = np.ones((64,64,3),np.uint8) * 255
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((2, -15),  char, font = font, fill = (b, g, r, a))
        print(f'Generating char_{i}.png')
        img = np.array(img_pil)
        cv2.imwrite(f'./images/char_{i}.png', img)
        i += 1

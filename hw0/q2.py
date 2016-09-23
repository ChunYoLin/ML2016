from PIL import Image
import sys
im = Image.open(sys.argv[1])
width, height = im.size
data = im.getdata()
out = im.transpose(Image.FLIP_TOP_BOTTOM)
out.save("ans2.png")

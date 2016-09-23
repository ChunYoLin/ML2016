from PIL import Image
import sys
im = Image.open(sys.argv[1])
out = im.transpose(Image.ROTATE_180)
out.save("ans2.png")

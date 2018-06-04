import cv2
import imutils
import operator
import segmentation
import cnn
import torch
from PIL import Image


if __name__ == "__main__":
	
	#Preprocessing and segmentation
	#image_src = cv2.imread("./Placas/placa2.jpg")
	#image_src = imutils.resize(image_src, height=500)
	#result = segmentation.digit_segmentation(image_src)
	#result_sort = sorted(result.items(), key=operator.itemgetter(0))
	#final_result = []
	#segmentation.save_digits(result_sort)

	#CNN	
	model = cnn.Net()
	model.load_state_dict(torch.load("char_recognizer.pt"))
	model.cuda()

	res = ""
	for i in range(6):
		pil_im = Image.open("./Test/test{}.jpg".format(i))
		res += cnn.predict_char(pil_im, model)
	print ('La placa es:  ', res)




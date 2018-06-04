import cv2
import imutils
import operator

def digit_segmentation(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	thresh = cv2.threshold(gray, 85, 240, cv2.THRESH_BINARY)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
	digits = {}
	for c in cnts:
		(x, y, w, h) = cv2.boundingRect(c)
		if (w > 90 and w < 300) and (h > 90 and h < 300):
			cv2.rectangle(image, (x - 10, y - 10), (x + w + 10, y + h + 10), (255, 0, 0), 5)
			roi = thresh[y : y + h, x : x + w]
			digits[x] = roi

	return digits

def save_digits(images):
	for idx, image in enumerate(images):
		image = cv2.resize(image, (28, 32), interpolation = cv2.INTER_CUBIC)
		image = roi_center(image)
		cv2.imwrite('./Test/test{}.jpg'.format(idx), image)


def roi_center(roi):
	ALTURA_ROI = 28
	ANCHURA_ROI = 32
	ANCHO_BORDE = 4

	relacion_aspecto_roi = float(roi.shape[1]) / float(roi.shape[0])
	nueva_anchura = int((ALTURA_ROI * relacion_aspecto_roi) + 0.5)
	b_top = ANCHO_BORDE
	b_bottom = ANCHO_BORDE
	b_left = int((ANCHURA_ROI - nueva_anchura) / 2)
	b_right = int((ANCHURA_ROI- nueva_anchura) / 2)

	roi_borde = cv2.copyMakeBorder(roi,b_top,b_bottom,b_left,b_right,cv2.BORDER_CONSTANT,value=[255,255,255])
	roi_trans = cv2.resize(roi_borde,(32,32))
	return roi_trans


'''if __name__ == "__main__":
	#image_src = cv2.imread("./Placas/placa4.jpg")
	#image_src = imutils.resize(image_src, height=500)
	#result = digit_segmentation(image_src)
	#result_sort = sorted(result.items(), key=operator.itemgetter(0))
	#final_result = []
	
	for r in result_sort:
		print r[0]
		final_result.append(r[1])


	
	for r in final_result:
		#r = cv2.resize(r, (32, 32), interpolation = cv2.INTER_CUBIC)
		cv2.imshow("Inicial", r)
		
		cv2.waitKey(0)

	#save_digits(final_result)
	#cv2.imshow("Inicial", image_src)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()'''
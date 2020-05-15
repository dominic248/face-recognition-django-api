# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import cv2
import os
import face_recognition
import pickle
from django.conf import settings

@csrf_exempt
def detect(request):
	returndata = {"success": False}
	detection_method="cnn" # face detection model to use: either `hog` or `cnn`
	encodings_file=os.path.join(settings.BASE_DIR,"encodings.pickle")
	if request.method == "POST":
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			image = _grab_image(url=url)
		print("[INFO] loading encodings...")
		data = pickle.loads(open(encodings_file, "rb").read())
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		print("[INFO] recognizing faces...")
		boxes = face_recognition.face_locations(rgb,model=detection_method)
		print(boxes)
		encodings = face_recognition.face_encodings(rgb, boxes)
		names = []
		for encoding in encodings:
			matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.54)
			print("matches", matches)
			name = "Unknown"
			if True in matches:
				matchedIdxs = [i for (i, b) in enumerate(matches) if b]
				counts = {}
				for i in matchedIdxs:
					name = data["names"][i]
					counts[name] = counts.get(name, 0) + 1
				name = max(counts, key=counts.get)
			names.append(name)
		print(names)
		returndata.update({"success": True,"name":names})
	return JsonResponse(returndata)


def _grab_image(path=None, stream=None, url=None):
    if path is not None:
        image = cv2.imread(path)
    else:
        if url is not None:
            resp = urllib.urlopen(url)
            data = resp.read()
        elif stream is not None:
            data = stream.read()
        image = np.asarray(bytearray(data), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    return image

@csrf_exempt
def train(request):
	returndata = {"success": False}
	knownEncodings = []
	knownNames = []
	detection_method="cnn" # face detection model to use: either `hog` or `cnn`
	encodings_file=os.path.join(settings.BASE_DIR,"encodings.pickle")
	resume=False
	name="Unknown"
	if request.method == "POST":
		if request.POST.get("resume", None):
			resume=bool(request.POST.get("resume", False))
		print(resume)
		if request.FILES.get("image", None) is not None:
			image = _grab_image(stream=request.FILES["image"])
		else:
			url = request.POST.get("url", None)
			if url is None:
				returndata["error"] = "No URL provided."
				return JsonResponse(returndata)
			image = _grab_image(url=url)
		if request.POST.get("name", None):
			name=request.POST.get("name", None)
		rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		boxes = face_recognition.face_locations(rgb,
			model=detection_method)
		encodings = face_recognition.face_encodings(rgb, boxes)
		for encoding in encodings:
			knownEncodings.append(encoding)
			knownNames.append(name)
		print("[INFO] serializing encodings...")
		if not resume:
			data = {"encodings": knownEncodings, "names": knownNames}
			f = open(encodings_file, "wb")
			f.write(pickle.dumps(data))
			f.close()
			returndata.update({"success": True})
		else:
			print("[INFO] Resuming...")
			f = open(encodings_file, "rb")
			file_encodings = pickle.load(f)
			f.close()
			encodings = file_encodings.get("encodings")
			names = file_encodings.get("names")
			print("new")
			for items in knownEncodings:
				encodings.append(items)
			for items in knownNames:
				names.append(items)
			data = {"encodings": encodings, "names": names}
			f = open(encodings_file, "wb")
			f.write(pickle.dumps(data))
			f.close()
			returndata.update({"success": True})
	return JsonResponse(returndata)


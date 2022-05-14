from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import model_from_json
import os



app = Flask(__name__)

name = 'ODIR_ResNet'

json_file = open('./'+name+'.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("./"+name+".h5")
print("Loaded model from disk")



picfolder = os.path.join('static','images')
app.config['UPLOAD_FOLDER']=picfolder

@app.route('/')
def home():
	return render_template("home.html", data="hey")

@app.route('/prediction/')
def prediction():
	return render_template("index.html", data="hii")


@app.route('/prediction/prediction', methods=["GET","POST"])
def result():


	left = request.files['left']
	left.save("left.jpg")

	img = cv2.imread("left.jpg")
	img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.array(img)
	left=[]
	left.append(img)


	right = request.files['right']
	right.save("right.jpg")

	img = cv2.imread("right.jpg")
	img = cv2.resize(img, (16, 16), interpolation=cv2.INTER_AREA)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	img = np.array(img)
	right=[]
	right.append(img)

	X1 = np.array(left)
	X2 = np.array(right)

	yhat = loaded_model.predict([X1, X2])

	a = yhat
	aa = np.argmax(a)

	if aa == 0:
		aa="Normal Eye"
	elif aa == 1:
		aa="diabetes"
	elif aa == 2:
		aa="glaucomna"
	elif aa == 3:
		aa="cataract"
	elif aa == 4:
		aa="Macular Degeneration"
	elif aa == 5:
		aa="Hypertension"
	elif aa == 6:
		aa="Myopia"
	elif aa == 7:
		aa="Other Diseases"

	if aa == "cataract":
		b="cataract"
		a="""
		A cataract is a clouding of the normally clear lens of the eye. For people who have cataracts, seeing through cloudy lenses is a bit like looking through a frosty or fogged-up window. Clouded vision caused by cataracts can make it more difficult to read, drive a car (especially at night) or see the expression on a friend's face.
		"""
		img=os.path.join(app.config['UPLOAD_FOLDER'],'12.jpg')
	elif aa =="glaucomna":
		b="glaucomna"
		a="""
		Glaucoma is the result of damage to the optic nerve. As this nerve gradually deteriorates, blind spots develop in your visual field. For reasons that doctors don't fully understand, this nerve damage is usually related to increased pressure in the eye.
		"""
	elif aa=="Macular Degeneration":
		b="Macular Degeneration"
		a="""
		macular degeneration is a common eye disorder among people over 50. It causes blurred or reduced central vision, due to thinning of the macula (MAK-u-luh). The macula is the part of the retina responsible for clear vision in your direct line of sight
		"""

	elif aa=="diabetes":
		b="diabetes"
		a="""
		Diabetic retinopathy (die-uh-BET-ik ret-ih-NOP-uh-thee) is a diabetes complication that affects eyes. It's caused by damage to the blood vessels of the light-sensitive tissue at the back of the eye (retina). At first, diabetic retinopathy might cause no symptoms or only mild vision problems.
		"""
	elif aa=="Hypertension":
		b="Hypertension"
		a="""
		In some cases, the retina becomes swollen. Over time, high blood pressure can cause damage to the retina's blood vessels, limit the retina's function, and put pressure on the optic nerve, causing vision problems. This condition is called hypertensive retinopathy (HR).
		"""
	elif aa=="Myopia":
		b="Myopia"
		a="""
		With normal vision, an image is sharply focused onto the retina. In nearsightedness (myopia), the point of focus is in front of the retina, making distant objects appear blurry. Nearsightedness (myopia) is a common vision condition in which you can see objects near to you clearly, but objects farther away are blurry
		"""
	elif aa=="Normal Eye":
		b="Normal Eye"
		a="Your eyes are healthy. Take care of it"
	elif aa=="Other Diseases":
		b="Other Diseases"
		a=" ;)"





	return render_template("prediction.html", a=a,img=img,b=b)


if __name__ == "__main__":
	app.run(debug=True)

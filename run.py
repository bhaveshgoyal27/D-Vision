import numpy as np
import cv2
import os
import streamlit as st
from obj import obj
from stairs import stairs
from emotion import emotion
from example import lie
from faces import facr

st.markdown('<style>body{background-color:powderblue;}</style>',unsafe_allow_html=True)

st.sidebar.header("CHEF'S HACK😷🔬")
ch = st.sidebar.selectbox(
    "Choice",
    [
        "Home",
        "Object Detector",
        "Stairs Detection",
        "Emotion Detection",
        "Lie Detection",
        "FACE RECOGNITION"
    ],
    key="main_select",
    )
if ch=="Home":
	a = "CHEF'S HACK😷🔬"
	st.title(a)

	a = '<p style="text-align: justify;font-size:20px;">The target population for the project will be all the specially abled people of our society.'
	a+=' The hardships faced by them is unimaginable.<br> As of March 4th 2019, there were 40 million people in India, including 1.6 million children, are visually '
	a+='impaired due to uncorrected refractive error. The number of physically challenged people in India count to a population of 9.5 million people.'
	a+='These numbers are not small as they tell us that nearly "3%" of our population is either visually impaired or physically challenged.'
	a+='Lives of such people is not easy. The difficulties they face on a daily basis is something no one can even think of it. '
	a+='<br>Today every visually challenged person uses a stick to know his surrounding and move accordingly. In this process many a times they lose their balance'
	a+=' and trip. They are always asked to wear black sunglasses to prevent any discomfort or dizziness. But this also causes an awareness amongst the people'
	a+=' near them, who look down upon them. They have a great sense of hearing which they have developed over the years, which help them to identify the people'
	a+=' who they already know and how they don’t. But there is always a chance of human error. On top of that they can easily be cheated by the others. '
	a+='The physically challenged people can barely move their body and have to rely on someone else to help them out in their day-to-day activities. This dependence'
	a+=' on others becomes really tiring after some point of time and the people can do nothing but feel helpless about themselves. '
	a+=' </p><br>'

	st.markdown(a,unsafe_allow_html=True)

	a = "<p style='text-align: justify;font-size:20px;'>Please choose an option you want to see first from the sidebar to proceed.</p>"
	st.markdown(a,unsafe_allow_html=True)

elif ch=="Object Detector":
	a = '<center><h2>OBJECT DETETCTION</h2></center><br>'
	a+= '<p style="text-align: justify;font-size:20px;">At present, every visually challenged person is equipped with a walking stick,'
	a+=' which the person uses to find the presence of an object in front of them. This process is a cumbersome process and gives a'
	a+=' feeling of dependence  to the person. Our Object detection can detect 80 objects which are used in day to day activities.</p> '
	st.markdown(a,unsafe_allow_html=True)

	space = st.empty()
	cap=cv2.VideoCapture(0)
	ss = st.sidebar.text_input(label='stop',key="stop")
	while True:
		ret,img = cap.read()
		if ret:
			img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
			img1 = obj(img)
			space.image(img1)
			if ss=="stop":
				cap.release()
				break

elif ch=="Stairs Detection":
	a = '<center><h2>Stairs Detection</h2></center><br><p style="text-align: justify;font-size:20px;">Currently going up and down the stairs is '
	a+= 'tiring process. We have come up with a solution which notifies the user the number of stairs in front of them.</p>'
	
	st.markdown(a,unsafe_allow_html=True)
	cap = cv2.VideoCapture('up.mp4')
	space1 = st.empty()
	while(cap.isOpened()):
		ret,img = cap.read()
		img1 = stairs(img)
		space1.image(img1)

elif ch=="Emotion Detection":
	space2 = st.empty()
	cap = cv2.VideoCapture(0)
	ss = st.sidebar.text_input(key="stop1",label="stop1")
	while True:
		ret,img = cap.read()
		if ret:
			img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
			img1 = emotion(img)
			space2.image(img1)
			if ss=="stop":
				cap.release()
				break

elif ch=="Lie Detection":
	space3 = st.empty()
	cap = cv2.VideoCapture(0)
	#cap = cv2.VideoCapture('lie.mp4')
	ss = st.sidebar.text_input(key="stop2",label="stop2")
	while True:
		ret,img = cap.read()
		if ret:
			img = cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
			img1 = lie(img)
			space3.image(img1)
			if ss=="stop":
				cap.release()
				break
elif ch=="FACE RECOGNITION":
	space4 = st.empty()
	cap = cv2.VideoCapture(0)
	ss = st.sidebar.text_input(key="stop3",label="stop3")
	while True:
		ret, img = cap.read()
		if ret:
			img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
			img1 = facr(img)
			space4.image(img1)
			if ss =="stop":
				cap.release()
				break
else:
	st.title("Choose one among the right option")

import sys
import caffe
import numpy as np
import cv2
import math

def EditFcnProto(templateFile, height, width):
	with open(templateFile, 'r') as ft:
		template = ft.read()
		outFile = 'DehazeNetFcn.prototxt'
		with open(outFile, 'w') as fd:
			fd.write(template.format(height_15=height+10, width_15=width+10))

def TransmissionEstimate(im_path, height, width):
	caffe.set_mode_gpu()
	#net = caffe.Net('DehazeNet-new122.prototxt', 'solver-new122_iter_57198.caffemodel', caffe.TEST)
	net_full_conv = caffe.Net('DehazeNetFcn.prototxt', '/home/zehao/shi-yan-2/models/model-new128/solver-new128_iter_10000.caffemodel', caffe.TEST)
	#net_full_conv.params['convC'][0].data.flat = net.params['ip1'][0].data.flat
	#net_full_conv.params['convC'][1].data[...] = net.params['ip1'][1].data
	im = caffe.io.load_image(im_path)
        im = im*40
	npad = ((7,8), (7,8), (0,0))
	im = np.pad(im, npad, 'symmetric')
	transformers = caffe.io.Transformer({'data': net_full_conv.blobs['data'].data.shape})
	transformers.set_transpose('data', (2,0,1))
        transformers.set_raw_scale('data',255)
	transformers.set_channel_swap('data', (2,1,0))
	out = net_full_conv.forward_all(data=np.array([transformers.preprocess('data', im)]))

	transmission = np.reshape(out['convD'], (height,width))
	return transmission

def DarkChannel(im,sz):
	b,g,r = cv2.split(im)
	dc = cv2.min(cv2.min(r,g),b)
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
	dark = cv2.erode(dc,kernel)
	return dark

def AtmLight(im,dark):
	[h,w] = im.shape[:2]
	imsz = h*w
	numpx = int(max(math.floor(imsz/1000),1))
	darkvec = dark.reshape(imsz,1)
	imvec = im.reshape(imsz,3)
	indices = darkvec.argsort()
	indices = indices[imsz-numpx::]
	atmsum = np.zeros([1,3])
	for ind in range(1,numpx):
		atmsum = atmsum + imvec[indices[ind]]
	A = atmsum / numpx
	return A

def Guidedfilter(im,p,r,eps):
	mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
	mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
	mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
	cov_Ip = mean_Ip - mean_I*mean_p
	mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
	var_I   = mean_II - mean_I*mean_I
	a = cov_Ip/(var_I + eps)
	b = mean_p - a*mean_I
	mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
	mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))
	q = mean_a*im + mean_b
	return q

def TransmissionRefine(im,et):
	gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
	gray = np.float64(gray)/255
	r = 60
	eps = 0.0001
	t = Guidedfilter(gray,et,r,eps)
	return t

def Recover(im,t,A,tx = 0.1):
	res = np.empty(im.shape,im.dtype)
	t = cv2.max(t,tx)
	for ind in range(0,3):
	    res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
	return res

if __name__ == '__main__':
	if not len(sys.argv) == 2:
		print 'Usage: python DeHazeNet.py haze_img_path'
		exit()
	else:
		im_path = sys.argv[1]
	src = cv2.imread(im_path)
	height = src.shape[0]
	width = src.shape[1]
	templateFile = 'DehazeFCN-new128.prototxt'
	EditFcnProto(templateFile, height, width)
	I = src/255.0
	dark = DarkChannel(I,15)
	A = AtmLight(I,dark)
	te = TransmissionEstimate(im_path, height, width)
        te =te*20
	t = TransmissionRefine(src,te)
        t=1-t
        #A=0.85
	J = Recover(I,t,A,0.1)
	cv2.imshow('TransmissionEstimate',te)
	cv2.imshow('TransmissionRefine',t)
        #t=1-t
	cv2.imshow('Origin',src)
	cv2.imshow('Dehaze',J)
	cv2.waitKey(0)
	save_path1 = im_path[:-4]+'_Dehaze3'+im_path[-4:len(im_path)]
        save_path2 = im_path[:-4]+'_DehazeT1'+im_path[-4:len(im_path)]
	cv2.imwrite(save_path1,J*255)
        cv2.imwrite(save_path2,t*255)
        print A
        print te[100,100],te[200,200]

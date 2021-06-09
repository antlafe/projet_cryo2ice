from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema
from scipy.interpolate import interp1d
from mpl_toolkits.basemap import Basemap
import os
from os.path import *
import math
from netCDF4 import Dataset
import glob
#from progress import *

def W99(mois,an,longitude,latitude,age):
	## Snow depth
	file='W99_sd.txt'
	a=np.zeros((12,6))
	ind=-1
	for line in open(file, 'r'):
		parts = line.split()
		ind=ind+1
		if len(parts)>0:
			for i in range(6):
				a[ind,i]=np.double(parts[i])






	x=(90-latitude)*np.cos(longitude/180*np.pi)
	y=(90-latitude)*np.sin(longitude/180*np.pi)
	snow=a[int(mois)-1,0]+a[int(mois)-1,1]*x+a[int(mois)-1,2]*y+a[int(mois)-1,3]*x*y+a[int(mois)-1,4]*x*x+a[int(mois)-1,5]*y*y

	vm=0
	vM=50
	snow[snow>50]=50
	snow[snow<0]=0

	## Snow Water Equivalent
	file='W99_swe.txt'
	a=np.zeros((12,6))
	ind=-1
	for line in open(file, 'r'):
		parts = line.split()
		ind=ind+1
		if len(parts)>0:
			for i in range(6):
				a[ind,i]=np.double(parts[i])






	x=(90-latitude)*np.cos(longitude/180*np.pi)
	y=(90-latitude)*np.sin(longitude/180*np.pi)
	swe=a[int(mois)-1,0]+a[int(mois)-1,1]*x+a[int(mois)-1,2]*y+a[int(mois)-1,3]*x*y+a[int(mois)-1,4]*x*x+a[int(mois)-1,5]*y*y

	sdens=swe*1000/snow





	return snow,sdens





import numpy

# Centers of chips in focal plane coordinates
N7=["N7",16.908,191.670]
N6=["N6",16.908,127.780]
N5=["N5",16.908,63.890]
N4=["N4",16.908,0.]
N3=["N3",16.908,-63.890]
N2=["N2",16.908,-127.780]
N1=["N1",16.908,-191.670]

N13=["N13",50.724,159.725]
N12=["N12",50.724,95.835]
N11=["N11",50.724,31.945]
N10=["N10",50.724,-31.945]
N9=["N9",50.724,-95.835]
N8=["N8",50.724,-159.725]

N19=["N19",84.540,159.725]
N18=["N18",84.540,95.835]
N17=["N17",84.540,31.945]
N16=["N16",84.540,-31.945]
N15=["N15",84.540,-95.835]
N14=["N14",84.540,-159.725]

N24=["N24",118.356,127.780]
N23=["N23",118.356,63.890]
N22=["N22",118.356,0.]
N21=["N21",118.356,-63.890]
N20=["N20",118.356,-127.780]

N28=["N28",152.172,95.835]
N27=["N27",152.172,31.945]
N26=["N26",152.172,-31.945]
N25=["N25",152.172,-95.835]

N31=["N31",185.988,63.890]
N30=["N30",185.988,0.]
N29=["N29",185.988,-63.890]

S7=["S7",-16.908,191.670]
S6=["S6",-16.908,127.780]
S5=["S5",-16.908,63.890]
S4=["S4",-16.908,0.]
S3=["S3",-16.908,-63.890]
S2=["S2",-16.908,-127.780]
S1=["S1",-16.908,-191.670]

S13=["S13",-50.724,159.725]
S12=["S12",-50.724,95.835]
S11=["S11",-50.724,31.945]
S10=["S10",-50.724,-31.945]
S9=["S9",-50.724,-95.835]
S8=["S8",-50.724,-159.725]

S19=["S19",-84.540,159.725]
S18=["S18",-84.540,95.835]
S17=["S17",-84.540,31.945]
S16=["S16",-84.540,-31.945]
S15=["S15",-84.540,-95.835]
S14=["S14",-84.540,-159.725]

S24=["S24",-118.356,127.780]
S23=["S23",-118.356,63.890]
S22=["S22",-118.356,0.]
S21=["S21",-118.356,-63.890]
S20=["S20",-118.356,-127.780]

S28=["S28",-152.172,95.835]
S27=["S27",-152.172,31.945]
S26=["S26",-152.172,-31.945]
S25=["S25",-152.172,-95.835]

S31=["S31",-185.988,63.890]
S30=["S30",-185.988,0.]
S29=["S29",-185.988,-63.890]

# order of chips when using numeric label
ccdid = [S29,S30,S31,S25,S26,S27,S28,S20,S21,S22,S23,S24,S14,S15,S16,S17,S18,S19,S8,S9,S10,S11,
         S12,S13,S1,S2,S3,S4,S5,S6,S7,N1,N2,N3,N4,N5,N6,N7,N8,N9,N10,N11,N12,N13,N14,N15,N16,N17,
         N18,N19,N20,N21,N22,N23,N24,N25,N26,N27,N28,N29,N30,N31]

# defines the size of a chip in mm.  One pixel=15 microns
xsize=2048*15e-6*1000
ysize=4096*15e-6*1000

# xc, yc are the (x,y) position of the lower left corner of each chip
xc = numpy.empty(len(ccdid)+1)
yc = numpy.empty(len(ccdid)+1)
for i,ext in enumerate(ccdid):
    xc[i+1] = ext[1]-xsize/2
    yc[i+1] = ext[2]-ysize/2

def toFocal(ccd,x,y):
    return x*15e-6*1000+xc[ccd],y*15e-6*1000+yc[ccd]

def toFocalArcmin(ccd,x,y):
    u,v = toFocal(ccd,x,y)
    # u,v are in mm.  Each pixel is 15 microns and 0.263 arcsec.
    # conversion is (1 pixel / 15e-3) (0.263 arcsec / pixel) (1 arcmin / 60 arcsec)
    factor = 0.263 / 15.e-3 / 60.
    return u * factor, v * factor

def toFocalArcsec(ccd,x,y):
    u,v = toFocal(ccd,x,y)
    # u,v are in mm.  Each pixel is 15 microns and 0.263 arcsec.
    # conversion is (1 pixel / 15e-3) (0.263 arcsec / pixel)
    factor = 0.263 / 15.e-3
    return u * factor, v * factor

def toFocalPixel(ccd,x,y):
    u,v = toFocal(ccd,x,y)
    # u,v are in mm.  Each pixel is 15 microns.
    # conversion is (1 pixel / 15e-3 mm)
    factor = 1. / 15.e-3
    return u * factor, v * factor

import colorsys
import random
import csv
import math
import base64
from PIL import Image
import numpy as np
import io

import cv2
from scipy import ndimage
from matplotlib.patches import Rectangle
from matplotlib import colors
from matplotlib import pyplot as plt
from skimage.color import rgb2lab


class Adapt:
    def __init__(self, dim, patchDim, occupancy):
        self.imgDim = dim
        self.halfImgDim = dim / 2 

        # TODO: obtain mCan and mProj values from Hololens
        self.mCam = np.array(  [[0.99693,	0.05401,	0.05667,	0.00708],
                               [-0.07171,	0.92042,	0.38429,	0.06751],
                               [0.03140,	0.38718,	-0.92147,	0.13505],
                               [0.00000,	0.00000,	0.00000,	1.00000]])

        self.mProj = np.array([[1.52314,	0.00000,	0.01697,	0.00000 ],
                                [0.00000,	2.70702,	-0.05741,	0.00000 ],
                                [0.00000,	0.00000,	-1.00000,	0.00000 ],
                                [0.00000,	0.00000,	-1.00000,	0.00000]])

        self.wPos = np.array([0, 0, 0])
        self.img = None
        self.patchDim = self.wDim2uvDim(np.array(patchDim))
        self.depth = 0
        self.colorScheme = True
        self.labelPosList = []
        self.occupancy = occupancy


    def place(self, b64img):
        imgData = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(imgData))
        self.img = img
        
        uvAnchor = self.anchorCentre()

        rDim = self.patchDim
        rOffset = 0
        cOffset = 0
        rSteps = int(self.imgDim[0] / rDim[0])
        cSteps = int(self.imgDim[1] / rDim[1])
        patchM = np.empty([rSteps, cSteps])
        patchE = np.empty([rSteps, cSteps])
        patchLogProb  = np.empty([rSteps, cSteps])
        
        for iR in range(rSteps):
            cOffset = 0
            for iC in range(cSteps):

                # crop image
                crop_rect = (cOffset, rOffset, cOffset + rDim[1], rOffset + rDim[0])                
                imgCrop = img.crop(crop_rect)

                # patch centre
                patchPos = np.array([cOffset + rDim[1]/2, rOffset + rDim[0]/2])
                posOffset = patchPos - uvAnchor
                offsetNorm = np.array([abs(posOffset[0]/rDim[0]), abs(posOffset[1]/rDim[1])])
                                                
                M = self.colourfulness(imgCrop)                
                #patchM[iR,iC] = M
                
                E = self.edgeness(imgCrop)
                #patchE[iR,iC] = E

                if len(self.labelPosList) == 0:
                    #anchor = (int(504/2), int(896/2))
                    F = 0
                else:
                    sum_h, sum_w = 0
                    for l in self.labelPosList:
                        sum_h += l[0]
                        sum_w += l[1]
                    anchor = (int(sum_h/len(self.labelPosList)), int(sum_w/len(self.labelPosList)))
                    F = self.fittsLaw(anchor, patchPos, rDim)

                if (self.check_occupancy(patchPos, rDim[1], rDim[0]) == 0):
                    patchLogProb[iR,iC] = ( self.offsetLogProb(offsetNorm) +
                                            self.colourfulnessLogProb(M) +
                                            self.edgenessLogProb(E)
                                            -0.5*F )
                
                    # TEMP fig only
                    patchE[iR,iC] = self.edgenessLogProb(E)

                cOffset += rDim[1]
                
            rOffset += rDim[0]

        #print(np.array2string(patchLogProb, formatter={'float_kind':lambda patchLogProb: "%.2f" % patchLogProb}))
        #print(np.array2string(patchF, formatter={'float_kind':lambda patchF: "%.2f" % patchF}))
        idxMax = np.unravel_index(np.argmax(patchLogProb, axis=None), patchLogProb.shape)
        uvMax = np.array([idxMax[1] * rDim[1] + rDim[1]/2, idxMax[0] * rDim[0] + rDim[0]/2])
        labelPos = self.uv2w(uvMax)
        labelPos = [labelPos[0], labelPos[1], 0.5] 
        self.labelPosList.append(labelPos)
        self.set_occupancy(patchPos, rDim[1], rDim[0])
        return (labelPos, uvMax, self.occupancy)


    # Checks if a given space is occupied (1 if occupied, 0 if else)
    def check_occupancy(self, center, panel_w, panel_h):
        min_x = int(center[1]-panel_w/2)
        min_y = int(center[0]-panel_h/2)
        max_x = int(center[1]+panel_w/2)
        max_y = int(center[0]+panel_h/2)

        if min_x < 0 or max_x > self.imgDim[1] or min_y < 0 or max_y > self.imgDim[0]:
            return 1
        
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if self.occupancy[(y, x)] == 1:
                    return 1
        return 0


    # Sets a given space as occupied (1 if occupied, 0 if else)
    def set_occupancy(self, center, panel_w, panel_y):
        min_x = int(center[1]-panel_w/2)
        min_y = int(center[0]-panel_y/2)
        max_x = int(center[1]+panel_w/2)
        max_y = int(center[0]+panel_y/2)

        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                self.occupancy[(y, x)] = 1

        
    def color(self, uvPos):
        crop_rect = (   uvPos[0] - self.patchDim[1]/2,
                        uvPos[1] - self.patchDim[0]/2, 
                        uvPos[0] + self.patchDim[1]/2,
                        uvPos[1] + self.patchDim[0]/2 )                
        patchBG = self.img.crop(crop_rect)
        
        # DEV ONLY Show patch
        #plt.imshow(patchBG, interpolation='nearest')
        #plt.show()

        #medianColor = ImageStat.Stat(patchBG).median
        dominantColor = self.dominantColor(patchBG)

        # Retrieve color distribution given dominant color and select best
        pColorLogProbs = self.panelColorDistribution(dominantColor)        
        pLightnessLogProbs = self.panelLightnessDistribution(dominantColor)        
        pColorScheme = [-1.0986, -1.0986, -999.9, -999.9, -1.0986, -999.9, -999.9, -999.9]

        pLogProbs = ( np.array(pColorLogProbs) + np.array(pLightnessLogProbs) )
        if (self.colorScheme):
            pLogProbs = pLogProbs + np.array(pColorScheme)

        #print(pColorLogProbs)
        #print(pLightnessLogProbs)
        #print(pLogProbs)

        iHueBin = np.argmax(pLogProbs)

        if (self.colorScheme):
            palette = [ [202, 184, 173],
                        [245, 240, 96],
                        [255, 0, 255],
                        [255, 0, 255],
                        [79, 91, 168],
                        [255, 0, 255],
                        [255, 0, 255],
                        [255, 0, 255] ]
            rgb = palette[iHueBin]
        else:
            rgb = np.array([255, 0, 255])
            if (iHueBin < 6):
                hVal = (iHueBin * 60.0) / 360.0
                rgb = np.multiply(colors.hsv_to_rgb(np.array([hVal, 1, 1])),255.0)
            elif (iHueBin == 6):
                rgb = np.array([255, 255, 255])
            elif (iHueBin == 7):
                rgb = np.array([0, 0, 0])

        labelColor = rgb

        # Toggle text colour
        textColor = np.array([255, 255, 255])
        if (self.perceivedBrightness(labelColor) > 140):
            textColor = np.array([0, 0, 0])

        return (labelColor, textColor)

    def dominantColor(self, imgPatch):
        domColor = np.array([255, 0, 0])

        # Convert patch pixels to HSV
        RGB = imgPatch.getdata()
        RGB_s = np.divide(RGB, 255.0)
        HSV = colors.rgb_to_hsv( RGB_s )

        # Find histogram hue peak
        nBins = 100
        binEdges = np.linspace(start = 0, stop = 1, num = nBins + 1)        
        [counts, edges] = np.histogram(HSV[:,0], bins=binEdges)
        iMaxCount = np.argmax(counts)
        iHueBin = iMaxCount + 1
        inds = np.digitize(HSV[:,0], bins=binEdges)

        # Convert peak to hVal and find medians of other dimensions
        hVal = iMaxCount / nBins + (1 / nBins / 2)
        sMean = np.mean(HSV[inds == iHueBin,1])
        vMean = np.mean(HSV[inds == iHueBin,2])

        # Assemble dominant color
        domColor  = colors.hsv_to_rgb(np.array([hVal, sMean, vMean]))
        domColor = np.multiply(domColor, 255)

        return domColor

    def colorHarmony(self, rgb_color, angle_size, num):
        ret = []
        hsv_color = colorsys.rgb_to_hsv(rgb_color[0], rgb_color[1], rgb_color[2])
        hsv_color = (hsv_color[0]*180, hsv_color[0]*255, hsv_color[0]*255)
        print(hsv_color)

        start = hsv_color[0] - angle_size
        start_angle = random.uniform(start, hsv_color[0])
        end_angle = start_angle + angle_size
        print(start_angle, end_angle)
        
        for i in range(num):
            angle = random.uniform(start_angle, end_angle)
            if angle < 0:
                angle = 360 + angle
            
            rgb = colorsys.hsv_to_rgb(angle/180, hsv_color[1]/255, hsv_color[2]/255)
            rgb = (rgb[0]*255, rgb[1]*255, rgb[2]*255)
            ret.append(rgb)

        return ret

    def panelColorDistribution(self, patchColor):
        rgb = np.divide(np.copy(patchColor).reshape((1,1,3)), 255.0)
        lab = rgb2lab(rgb).reshape((1,3))[0]
        hsv = colors.rgb_to_hsv(np.divide(patchColor, 255.0))
        
        abTreshold = 5
        whiteThreshold = 80
        blackThreshold = 20

        iBin = -1

        hueBinEdges =  np.array([0, 30, 90, 150, 210, 270, 330, 360])        
        iHueBin = np.argmax(hueBinEdges > hsv[0]*360)-1
        if (iHueBin == 6):
            iHueBin = 0

        if  (np.hypot(lab[1],lab[2]) < abTreshold):
            iBin = 2
        elif (lab[0] > whiteThreshold):
            iBin = 0
        elif (lab[0] < blackThreshold):
            iBin = 1
        else:
            iBin = 3 + iHueBin    

        # Columns are Red, Yellow, Green, Cyan, Blue, Magenta
        binLogProbs = [[-2.1203, -1.9196, -2.9312, -1.6784, -1.2264, -2.3716, -3.6243, -2.5257],
                        [-1.7198, -2.0075, -1.8068, -1.9534, -1.8068, -2.5953, -2.5953, -2.7006],
                        [-1.3218, -2.2225, -2.0794, -2.0794, -1.9543, -2.0149, -3.1781, -2.8416],
                        [-1.8207, -2.0149, -1.9459, -2.2116, -1.5404, -2.3026, -3.0445, -2.4027],
                        [-1.8608, -2.1327, -2.0526, -2.1752, -1.5612, -2.1537, -2.6717, -2.4204],
                        [-0.6931, -99.9000, -99.9000, -99.9000, -0.6931, -99.9000, -99.9000, -99.9000],
                        [-3.1355, -2.4423, -1.7492, -2.0369, -1.3437, -1.5261, -3.1355, -3.1355],
                        [-2.1203, -1.4271, -1.8326, -2.5257, -1.8326, -2.1203, -99.9000, -2.1203],
                        [-99.9000, -99.9000, -99.9000, -99.9000, -99.9000, -99.9000, 0.0000, -99.9000]]
        
        binRow = binLogProbs[iBin]

        return binRow

    def panelLightnessDistribution(self, patchColor):
        binLogProbs = [[-1.7198, -2.0075, -1.8068, -1.9534, -1.8068, -2.5953, -2.5953, -2.7006],
                        [-1.6405, -2.1001, -1.8769, -2.4449, -1.5645, -2.2336, -2.8802, -2.6391],
                        [-1.7825, -1.9873, -2.1339, -2.2749, -1.5667, -2.0338, -2.7757, -2.6359],
                        [-1.8570, -2.2274, -1.9838, -1.9311, -1.6827, -2.2274, -2.9557, -2.2274],
                        [-2.1203, -1.9196, -2.9312, -1.6784, -1.2264, -2.3716, -3.6243, -2.5257]]

        rgb = np.divide(np.copy(patchColor).reshape((1,1,3)), 255.0)
        lab = rgb2lab(rgb).reshape((1,3))[0]
        
        lBinEdges =  np.array([0, 20, 40, 60, 80, 100])        
        iBin = np.argmax(lBinEdges > lab[0])-1

        binRow = binLogProbs[iBin]

        return binRow

    def fittsLaw(self, anchor, pos, label):
        width = label[1]
        a = 0.4926
        b = 0.6332
        dist = math.sqrt( (anchor[0] - pos[0])**2 + (anchor[1] - pos[1])**2 )
        ID = math.log(dist/width+1,2)
        T = a + b*ID
        return T

    def anchorPos(self):
        return self.wPos
        
    def anchorCentre(self):
        return self.w2uv(self.wPos)
        
    def setMetaData(self, img_meta):
        metaSplit = img_meta.split(';')
        self.wPos = self.strArr2numArr(metaSplit[0])
        self.mCam = self.strArr2mat(metaSplit[1])        
        self.mProj = self.strArr2mat(metaSplit[2])
    
    def msgString(self, placePos, labelColor, textColor):
        sComma = ","
        placePosStr = ["%.4f" % x for x in placePos]
        placePosStr = sComma.join(placePosStr)
        labelColorStr = ["%.1f" % x for x in labelColor]
        labelColorStr = sComma.join(labelColorStr)
        textColorStr = ["%d" % x for x in textColor]
        textColorStr = sComma.join(textColorStr)
        #placePosStr = np.array2string(patchLogProb, formatter={'float_kind':lambda placePos: "%.3f," % placePos})
        msg = "%s;%s;%s" % (placePosStr, labelColorStr, textColorStr)
        return msg


    def w2uv(self, wPos):
        mCamInv = np.linalg.inv(self.mCam)
        
        wPosTemp = np.append(wPos, 1)
        cPos = np.matmul(mCamInv, wPosTemp)        
        iPos = np.matmul(self.mProj, cPos)
        self.depth = iPos[2] 
        
        # Convert to img coords
        u = self.halfImgDim[1] + self.halfImgDim[1] * iPos[0]
        v = self.halfImgDim[0] - self.halfImgDim[0] * iPos[1]
        pos = np.array([u, v])
        return pos
        
    def uv2w(self, uv):
        xImg = (uv[0] - self.halfImgDim[1]) / self.halfImgDim[1]
        yImg = -(uv[1] - self.halfImgDim[0]) / self.halfImgDim[0]
        
        iPos = np.array([xImg, yImg, self.depth])
        mProjInv = np.linalg.inv(self.mProj[np.ix_([0,1,2],[0,1,2])])
        cPos = np.matmul(mProjInv, iPos)
                
        cPosTemp = np.append(cPos, 1)
        wPos = np.matmul(self.mCam, cPosTemp)
        
        return wPos[0:3]

    # Compute label dimensions in uv coord
    def wDim2uvDim(self, labelDim):
        wPosTemp = np.append(self.wPos, 1)
        mCamInv = np.linalg.inv(self.mCam)
        cPos = np.matmul(mCamInv, wPosTemp)        
   
        halfEdgeWidth = np.copy(cPos)
        halfEdgeWidth[0] += labelDim[0] / 2.0

        halfEdgeHeight = np.copy(cPos)
        halfEdgeHeight[1] += labelDim[1] / 2.0

        iC = np.matmul(self.mProj, cPos)
        iW = np.matmul(self.mProj, halfEdgeWidth)        
        iH = np.matmul(self.mProj, halfEdgeHeight)
                
        # Convert to img coords
        uW = 2.0 * self.halfImgDim[1] * abs(iC[0] - iW[0])
        vH = 2.0 * self.halfImgDim[0] * abs(iC[1] - iH[1])
        dim = np.array([uW, vH])

        return dim
    
    def strArr2numArr(self, strArr):
        cols = strArr.split(',')
        l = len(cols)
        arr = np.zeros(l)
        iC = 0
        for c in cols:
            arr[iC] = float(c)
            iC += 1
        return arr
        
    def strArr2mat(self, strArr):
        arr = self.strArr2numArr(strArr)
        m =  arr.reshape(4,4)
        return m
    
    def colourfulness(self, img):
        R = np.array(img.getdata(0))
        G = np.array(img.getdata(1))
        B = np.array(img.getdata(2))
        rg = R - G
        yb = 0.5 * (R + G) - B        
        sig_rgyb = math.sqrt(np.var(rg) + np.var(yb))
        mu_rgyb  = math.sqrt(math.pow(np.mean(rg),2) + math.pow(np.mean(yb),2))
        M = sig_rgyb + 0.3 * mu_rgyb
        return M

    def colourfulnessLogProb(self, M):
        binEdges = np.array([0,15,33,45,59,82,109,200])
        binLogProbs = np.array([-0.4899,-1.2813,-2.7429,-3.5005,-4.5038,-5.5154,-99.9])
        iBin = np.argmax(binEdges>M)-1
        return binLogProbs[iBin]
        
    def edgeness(self, img):
        imgBW = img.convert('L')
        imgCV = np.array(imgBW)
        sx = cv2.Sobel(imgCV,cv2.CV_64F,1,0, borderType=cv2.BORDER_REPLICATE )
        sy = cv2.Sobel(imgCV,cv2.CV_64F,0,1, borderType=cv2.BORDER_REPLICATE )
        sobel=np.hypot(sx,sy)
        nEl = sobel.shape[0] * sobel.shape[1]
        nEdgePxls = np.count_nonzero(sobel.flatten() > 100)
        F = nEdgePxls / nEl * 100
        return F
        
    def edgenessLogProb(self, F):
        binEdges = np.array([0,10,20,30,40,50,60,70,80,90,100])
        binLogProbs = np.array([-0.3841,-2.2289,-2.7121,-3.1175,-3.4052,-4.0685,-3.7237,-4.1291,-4.8223,-6.2086])
        iBin = np.argmax(binEdges>F)-1
        return binLogProbs[iBin]
                    
    def offsetLogProb(self, normalized_offset):
        xBinEdges = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
        yBinEdges = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        ixBin = np.argmax(xBinEdges>normalized_offset[0])-1
        iyBin = np.argmax(yBinEdges>normalized_offset[1])-1
        
        binLogProbs = [[-5.1110, -1.7968, -1.5321, -3.0741, -5.1110, -6.9027],
                        [-5.5164, -3.1186, -2.8597, -4.2637, -99.9000, -99.9000],
                        [-4.8233, -2.8252, -2.6542, -3.9070, -6.2096, -99.9000],
                        [-4.0695, -2.5207, -2.7756, -3.6069, -6.2096, -99.9000],
                        [-4.1302, -3.5705, -4.0695, -5.1110, -6.2096, -6.9027],
                        [-6.2096, -5.2933, -4.8233, -6.2096, -99.9000, -99.9000],
                        [-6.9027, -6.9027, -6.9027, -99.9000, -99.9000, -99.9000]]

        logProb = -99.9
        if (ixBin >= 0 and iyBin >= 0):
            logProb = binLogProbs[ixBin][iyBin]

        return logProb

    def perceivedBrightness(self, color):
        # From: https://www.w3.org/TR/AERT/
        #((Red value X 299) + (Green value X 587) + (Blue value X 114)) / 1000
        b = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        return b

def parse_csv(panel_file):
    with open(panel_file, 'r') as f:
        panel_info = [{k: v for k, v in row.items()}
            for row in csv.DictReader(f, skipinitialspace=True)]
        return panel_info

def test():
    f = open("C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\context_img_buff_1623145003.log", "r")
    byte_arr = bytes(f.read(), 'utf-8')

    img_dim = [504, 896]
    patch_dim = [63, 112]
    label_dim = [0.2, 0.05]

    a = Adapt(np.array(img_dim), np.array(patch_dim), np.array(label_dim))
    (labelPos, uvPlace) = a.place(byte_arr)
    (labelColor, textColor) = a.color(uvPlace)
    adaptPayload = a.msgString(labelPos, labelColor, textColor)
    print(adaptPayload)
    #plt.imshow(adapt.img, interpolation='nearest')

    print(a.colorHarmony(labelColor, 93.6, 5))


def main():
    img_dim = [504, 896]
    occupancy = {}
    
    for y in range(img_dim[0]):
        for x in range(img_dim[1]):
            occupancy[(y, x)] = 0

    directory = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images"
    img_file = "context_img_1623145003.png"
    panel_file = "C:\\Users\\2020\\Documents\\Dissertation\\data.csv"
    f = open("C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\context_img_buff_1623145003.log", "r")
    byte_arr = bytes(f.read(), 'utf-8')
    
    panel_info = parse_csv(panel_file)
    out_filename = directory + '\\out.txt'
    print('Saving info to %s' % out_filename)

    with open(out_filename, "w") as f:
        for i, p in enumerate(panel_info):
            dim = [float(p['height']), float(p['width'])]
            a = Adapt(np.array(img_dim), np.array(dim), occupancy)
            (labelPos, uvPlace, occupancy) = a.place(byte_arr)
            (labelColor, textColor) = a.color(uvPlace)
            adaptPayload = a.msgString(labelPos, labelColor, textColor)
            
            dim_str = str(dim[0]) + ',' + str(dim[1])
            pos_str = str(labelPos[0]) + ',' + str(labelPos[1]) + ',' + str(labelPos[2])
            
            if (i == 0):
                color = a.colorHarmony(labelColor, 93.6, len(panel_info))
                color_str = str(labelColor[0]) + ',' + str(labelColor[1]) + ',' + str(labelColor[2])
            else:
                color_str = str(color[i][0]) + ',' + str(color[i][1]) + ',' + str(color[i][2])

            text_color_str = str(textColor[0]) + ',' + str(textColor[1]) + ',' + str(textColor[2])

            line =  dim_str + ';' + pos_str + ';' + color_str + ';' + text_color_str + '\n'
            print(line)
            f.write(line)


if __name__ == "__main__":
    main()
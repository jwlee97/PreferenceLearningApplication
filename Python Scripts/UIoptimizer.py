import base64
import io
import math
import numpy as np
import random

import linalg_helpers as linalg
import arm_position_helpers as armpos

import cv2
from skimage.color import rgb2lab
from matplotlib import colors
from PIL import Image


class UIOptimizer:
    def __init__(self, b64img, imgDim, panelDim, num_panels, occlusion, colorfulness, edgeness,
                 fitts_law, ce, muscle_act, rula, arm_proper_length=33, forearm_hand_length=46, spacing=10):
        self.imgDim = imgDim
        self.halfImgDim = imgDim / 2
        self.num_panels = num_panels

        self.occlusion = occlusion # True if occlusion is enabled
        self.colorfulness_weight = colorfulness
        self.edgeness_weight = edgeness
        self.fittslaw_weight = fitts_law
        self.ce_weight = ce
        self.muscle_act_weight = muscle_act
        self.rula_weight = rula

        self.forearm_hand_length = forearm_hand_length
        self.arm_proper_length = arm_proper_length
        self.arm_total_length = arm_proper_length + forearm_hand_length

        # TODO: obtain mCan and mProj values from Hololens
        self.mCam = np.array(  [[0.99693,	0.05401,	0.05667,	0.00708],
                               [-0.07171,	0.92042,	0.38429,	0.06751],
                               [0.03140,	0.38718,	-0.92147,	0.13505],
                               [0.00000,	0.00000,	0.00000,	1.00000]])

        self.mProj = np.array(  [[1.52314,	0.00000,	0.01697,	0.00000 ],
                                [0.00000,	2.70702,	-0.05741,	0.00000 ],
                                [0.00000,	0.00000,	-1.00000,	0.00000 ],
                                [0.00000,	0.00000,	-1.00000,	0.00000]])

        self.wPos = np.array([0, 0, 0])
        self.depth = 0
        self.colorScheme = True

        imgData = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(imgData))
        self.img = img

        self.occupancyMap = {}
        self.labelPosList = []
        self.uvPlaceList = []
        self.panelDim = []

        self.voxels = armpos.compute_interaction_space(spacing,
                                                       [(-15, self.arm_total_length), (-self.arm_total_length, self.arm_total_length),
                                                       (-self.arm_proper_length / 2 - forearm_hand_length, self.arm_total_length)],
                                                        self.arm_total_length)

        for i in range(self.num_panels):
            self.panelDim.append(self.wDim2uvDim(np.array(panelDim[i]))) # panel dimensions in uv space
    
        if self.occlusion == False:
            for y in range(imgDim[0]):
                for x in range(imgDim[1]):
                    self.occupancyMap[(y, x)] = 0


    def weighted_optimization(self):
        for i in range(self.num_panels):
            panel_dim = self.panelDim[i]
            panelWeightedSum = {}

            for v in self.voxels:
                uv = self.w2uv([v[1]/100, v[0]/100, v[2]/100])
                crop_rect = (uv[1], uv[0], uv[1]+panel_dim[1], uv[0]+panel_dim[0])         
                imgCrop = self.img.crop(crop_rect)
                arm_pos = self.get_pose(v)
                                       
                M = self.colourfulness(imgCrop)                
                E = self.edgeness(imgCrop)
                CE = 0
                MA = 0
                R = 0

                if (arm_pos != []):
                    CE = self.consumed_endurance(arm_pos)
                    MA = self.muscle_activation_reserve_function(arm_pos)
                    R = self.rula(arm_pos)

                if len(self.labelPosList) == 0:
                    F = 0
                else:
                    sum_h = 0
                    sum_w = 0
                    for l in self.labelPosList:
                        sum_h += l[0]
                        sum_w += l[1]
                    anchor = (int(sum_h/len(self.labelPosList)), int(sum_w/len(self.labelPosList)))
                    F = self.fittsLaw(anchor, uv, panel_dim)

                panelWeightedSum[(v[0], v[1], v[2])] = M*self.colorfulness_weight + E*self.edgeness_weight + F*self.fittslaw_weight + \
                                                        CE*self.ce_weight + MA*self.muscle_act_weight + R*self.rula_weight

            sorted_pts = {k: v for k, v in sorted(panelWeightedSum.items(), key=lambda item: item[1])}
            sorted_keys = list(sorted_pts.keys())
        
            if (self.occlusion == False):
                for k in sorted_keys:
                    uvPos = self.w2uv([k[1]/100, k[0]/100, k[2]/100])
                    if self.check_occupancyMap(uvPos, panel_dim) == 0:
                        self.set_occupancyMap(uvPos, panel_dim)
                        wPos = [k[1]/100, k[0]/100, k[2]/100]
                        self.labelPosList.append(wPos)
                        self.uvPlaceList.append(uvPos)
                        break
            else:
                k = sorted_keys[0]
                wPos = [k[1]/100, k[0]/100, k[2]/100]
                uvPos = self.w2uv(wPos)
                self.labelPosList.append(wPos)
                self.uvPlaceList.append(uvPos)
            
            print(wPos, uvPos)

        return (self.labelPosList, self.uvPlaceList)


    # Checks if a given space is occupied (1 if occupied, 0 if else)
    def check_occupancyMap(self, pos, panel_dim):
        min_x = int(pos[1]-panel_dim[1]/2)
        min_y = int(pos[0]-panel_dim[0]/2)
        max_x = int(pos[1]+panel_dim[1]/2)
        max_y = int(pos[0]+panel_dim[0]/2)
   
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                if (y, x) in self.occupancyMap.keys():
                    if self.occupancyMap[(y, x)] == 1:
                            return 1
        return 0


    # Sets a given space as occupied (1 if occupied, 0 if else)
    def set_occupancyMap(self, pos, panel_dim):
        min_x = int(pos[1]-panel_dim[1]/2)
        min_y = int(pos[0]-panel_dim[0]/2)
        max_x = int(pos[1]+panel_dim[1]/2)
        max_y = int(pos[0]+panel_dim[0]/2)
        
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                 self.occupancyMap[(y, x)] = 1


    def color(self, uvPosList):
        labelColors = []
        textColors = []
        for i in range(self.num_panels):
            (label, text) = self._color(uvPosList[i], i)
            labelColors.append(label)
            textColors.append(text)

        return (labelColors, textColors)


    def _color(self, uvPos, i):
        crop_rect = (   uvPos[0] - self.panelDim[i][1]/2,
                        uvPos[1] - self.panelDim[i][0]/2, 
                        uvPos[0] + self.panelDim[i][1]/2,
                        uvPos[1] + self.panelDim[i][0]/2 )                
        panelBG = self.img.crop(crop_rect)
        
        dominantColor = self.dominantColor(panelBG)

        # Retrieve color distribution given dominant color and select best
        pColorLogProbs = self.panelColorDistribution(dominantColor)        
        pLightnessLogProbs = self.panelLightnessDistribution(dominantColor)        
        pColorScheme = [-1.0986, -1.0986, -999.9, -999.9, -1.0986, -999.9, -999.9, -999.9]

        pLogProbs = ( np.array(pColorLogProbs) + np.array(pLightnessLogProbs) )
        if (self.colorScheme):
            pLogProbs = pLogProbs + np.array(pColorScheme)

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


    def colorHarmony(self, RGB, angle_size):
        ret = []

        RGB_s = np.divide(RGB, 255.0)
        HSV = colors.rgb_to_hsv(RGB_s)

        start = HSV[0]*360 - angle_size
        start_angle = random.uniform(start, HSV[0]*360)
        end_angle = start_angle + angle_size
        
        for i in range(self.num_panels):
            angle = random.uniform(start_angle, end_angle)
            if angle < 0:
                angle = 360 + angle
            
            RGB_ret = colors.hsv_to_rgb([angle/360, HSV[1], HSV[2]])
            RGB_ret = np.multiply(RGB_ret, 255.0)
            ret.append(RGB_ret)

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
        

    def get_pose(self, voxel):
        arm_poses = self.compute_anchor_arm_poses(voxel)
        if (arm_poses == []):
            return arm_poses

        arm_poses = arm_poses[0]
        pose = [0, voxel[0], voxel[1], voxel[2], arm_poses['elbow_x'], arm_poses['elbow_y'], 
             arm_poses['elbow_z'], arm_poses['elv_angle'], arm_poses['shoulder_elv'], arm_poses['elbow_flexion']]
        
        return pose

    
    def compute_anchor_arm_poses(self, end_effector, rotation_step=-math.pi / 8, limit=-math.pi * 3 / 4):
        arm_poses = []
        u, v, c, r = armpos.compute_elbow_plane(end_effector, self.arm_proper_length, self.forearm_hand_length)

        # If result is None, pose is not possible with current constraints
        if u is None:
            return arm_poses

        # rotate from 0 to 135 degrees
        elbow_positions = []
        theta = 0

        while theta > limit:
            elbow_positions.append(r * (math.cos(theta) * u + math.sin(theta) * v) + c)
            theta += rotation_step

        for elbow_pos in elbow_positions:
            elv_angle = math.atan2(elbow_pos[0], elbow_pos[2])
            if math.degrees(elv_angle) > 130:
                continue

            # both values are normalized
            if not math.isclose(math.cos(elv_angle), 0, abs_tol=1e-5):
                s2 = elbow_pos[2] / (math.cos(elv_angle) * self.arm_proper_length)
            else:
                s2 = elbow_pos[0] / (math.sin(elv_angle) * self.arm_proper_length)

            shoulder_elv = math.atan2(s2, -elbow_pos[1] / self.arm_proper_length)
            elbow_flexion = linalg.law_of_cosines_angle(self.arm_proper_length, self.forearm_hand_length,
                                                                linalg.magnitude(end_effector), radians=False)
            elbow_flexion_osim = 180 - elbow_flexion

            humerus_transform = np.linalg.inv(armpos.compute_base_shoulder_rot(elv_angle, shoulder_elv))
            forearm_vector = end_effector - elbow_pos
            point = humerus_transform @ np.array([forearm_vector[0], forearm_vector[1], forearm_vector[2], 1])
            shoulder_rot = -math.degrees(math.atan2(point[2], point[0]))

            arm_poses.append({'elbow_x': elbow_pos[0], 'elbow_y': elbow_pos[1], 'elbow_z': elbow_pos[2],
                            'elv_angle': math.degrees(elv_angle), 'shoulder_elv': math.degrees(shoulder_elv),
                            'shoulder_rot': shoulder_rot, 'elbow_flexion': elbow_flexion_osim})

        return arm_poses


    def consumed_endurance(self, pose):
        # Frievalds arm data for 50th percentile male:
        # upper arm: length - 33cm; mass - 2.1; distance cg - 13.2
        # forearm: length - 26.9cm; mass - 1.2; distance cg - 11.7
        # hand: length - 19.1cm; mass - 0.4; distance cg - 7.0

        # retrieve pose data and convert to meters
        end_effector = np.array(pose[1:4]) / 100
        elbow = np.array(pose[4:7]) / 100

        # ehv stands for elbow hand vector
        ehv_unit = linalg.normalize(end_effector - elbow)
        elbow_unit = linalg.normalize(elbow)
        
        # Due to the fact that we lock the hand coordinate (always at 0 degrees), the CoM of the elbow - hand vector
        # will always be at 17.25cm from the elbow for 50th percent male
        # 11.7 + 0.25 * 22.2 = 17.25
        # 17.25 / 46 = 0.375
        # check appendix B of Consumed Endurance paper for more info
        d = elbow + ehv_unit * self.forearm_hand_length * 0.01 * 0.375
        a = elbow_unit * self.arm_proper_length * 0.01 * 0.4
        ad = d - a
        com = a + 0.43 * ad

        # mass should be adjusted if arm dimensions change
        # 3.7kg for 50th percentile male, currently a simple heuristic based on arm size.
        adjusted_mass = (self.forearm_hand_length + self.arm_proper_length) / 79 * 3.7
        torque_shoulder = np.cross(com, adjusted_mass * np.array([0, 9.8, 0]))
        torque_shoulder_mag = linalg.magnitude(torque_shoulder)

        strength = torque_shoulder_mag / 101.6 * 100
        return strength


    def rula(self, pose):
        # arm pose is already computed for osim
        end_effector = pose[1:4]
        elv_angle = pose[7]
        shoulder_elv = pose[8]
        elbow_flexion = pose[9]
        rula_score = 0

        # upper arm flexion / extension
        if shoulder_elv < 20:
            rula_score += 1
        elif shoulder_elv < 45:
            rula_score += 2
        elif shoulder_elv < 90:
            rula_score += 3
        else:
            rula_score += 4

        # add 1 if upper arm is abducted
        # we consider arm abducted if elv_angle is < 45 and > -45, and shoulder_elv > 30
        if -60 > elv_angle < 60 and shoulder_elv > 30:
            rula_score += 1

        # lower arm flexion
        if 60 < elbow_flexion < 100:
            rula_score += 1
        else:
            rula_score += 2

        # if lower arm is working across midline or out to the side add 1
        # according to MoBL model, shoulder is 17cm from thorax on z axis (osim coord system), we use that value:
        if end_effector[2] + 17 < 0 or end_effector[2] > 0:
            rula_score += 1

        # wrist is always 1, due to fixed neutral position
        rula_score += 1

        return rula_score


    def muscle_activation_reserve_function(self, pose):
        # we want to give priority to poses with the lowest reserve values. Hence, we use the max reserve value of all
        # voxels, where their reserve value is the minimum between all the poses.
        # voxels that have reserve values among a threshold receive the worst comfort rating (1).
       
        reserve_threshold = 250
        muscle_activation_reserve = pose[1] + pose[2] / reserve_threshold
        return muscle_activation_reserve
    


def test_file():
    directory = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\"
    f = open(directory + "context_img_buff_1623145003.log", "r")
    byte_arr = bytes(f.read(), 'utf-8')
    out_file = directory + 'out.txt'
    print('Saving info to %s' % out_file)

    img_dim = [504, 896]
    panel_dim = [(0.1, 0.15), (0.05, 0.1), (0.2, 0.1), (0.1, 0.2)]
    occlusion = True
    num_panels = 4
    color_harmony_template = 93.6

    colorfulness = 0.33
    edgeness = 0.33
    fitts_law = 0.33
    ce = 0.0
    muscle_act = 0.0
    rula = 0.0

    with open(out_file, "w") as f:
        opt = UIOptimizer(byte_arr, np.array(img_dim), np.array(panel_dim), num_panels, occlusion, 
                          colorfulness, edgeness, fitts_law, ce, muscle_act, rula)
        
        (labelPos, uvPlaces) = opt.place()
        (labelColors, textColors) = opt.color(uvPlaces)
        colors = opt.colorHarmony(labelColors[0], color_harmony_template)
   
        for i in range(num_panels):
            dim_str = str(panel_dim[i][0]) + ',' + str(panel_dim[i][1])
            pos_str = str(labelPos[i][0]) + ',' + str(labelPos[i][1]) + ',' + str(labelPos[i][2])
            color_str = str(colors[i][0]) + ',' + str(colors[i][1]) + ',' + str(colors[i][2])
            text_color_str = str(textColors[i][0]) + ',' + str(textColors[i][1]) + ',' + str(textColors[i][2])

            line =  dim_str + ';' + pos_str + ';' + color_str + ';' + text_color_str + '\n'
            print(line)
            f.write(line)


def main():
    directory = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\"
    f = open(directory + "context_img_buff_1623145003.log", "r")
    byte_arr = bytes(f.read(), 'utf-8')
    out_file = directory + '\\out\\' + 'out.png'
    img_path = directory + "context_img_1623145003.png"
    img = cv2.imread(img_path)

    img_dim = [504, 896]
    panel_dim = [(0.1, 0.15), (0.05, 0.1), (0.2, 0.1), (0.1, 0.2)]
    occlusion = False
    color_harmony = True
    num_panels = 4
    color_harmony_template = 93.6

    colorfulness = 0.0
    edgeness = 0.0
    fitts_law = 0.0
    ce = 0.33
    muscle_act = 0.33
    rula = 0.33

    opt = UIOptimizer(byte_arr, np.array(img_dim), np.array(panel_dim), num_panels, occlusion, 
                      colorfulness, edgeness, fitts_law, ce, muscle_act, rula)

    (labelPos, uvPlace) = opt.weighted_optimization()
    (labelColor, textColor) = opt.color(uvPlace)

    if color_harmony == True:
        colors =  opt.colorHarmony(labelColor[0], color_harmony_template)
    else:
        colors = labelColor

    for i in range(num_panels):
        min_y = int(uvPlace[i][0] - opt.panelDim[i][0]/2)
        max_y = int(uvPlace[i][0] + opt.panelDim[i][0]/2)
        min_x = int(uvPlace[i][1] - opt.panelDim[i][1]/2)
        max_x = int(uvPlace[i][1] + opt.panelDim[i][1]/2)
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), colors[i], -1)
    
    cv2.imwrite(out_file, img)


if __name__ == "__main__":
    main()
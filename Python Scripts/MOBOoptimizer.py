import base64
import io
import math
import numpy as np

import cv2
from matplotlib import colors
from skimage.color import rgb2lab
from PIL import Image

import linalg_helpers as linalg
import arm_position_helpers as armpos

from gpflowopt.domain import ContinuousParameter
from gpflowopt.bo import BayesianOptimizer
from gpflowopt.design import FactorialDesign
from gpflowopt.acquisition import ExpectedImprovement
from gpflowopt.optim import SciPyOptimizer, StagedOptimizer, MCOptimizer
import gpflow
import gpflowopt


class ObjectiveFunction:
    def __init__(self, b64img, imgDim, panel_dim, arm_proper_length=33, forearm_hand_length=46, spacing=10):
        imgData = base64.b64decode(b64img)
        img = Image.open(io.BytesIO(imgData))
        self.img = img
        self.imgDim = imgDim
        self.halfImgDim = self.imgDim / 2
        self.panelDim = panel_dim
        self.num_panels = 1

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

        self.voxels = armpos.compute_interaction_space(spacing,
                            [(-15, self.arm_total_length), (-self.arm_total_length, self.arm_total_length),
                             (-self.arm_proper_length / 2 - forearm_hand_length, self.arm_total_length)],
                              self.arm_total_length)

        x_l = self.voxels[0][0]
        y_l = self.voxels[0][1]
        z_l = self.voxels[0][2]
        x_u = self.voxels[0][0]
        y_u = self.voxels[0][1]
        z_u = self.voxels[0][2]

        for v in self.voxels:
            uv = self.w2uv([v[0]/100, v[1]/100, v[2]/100])
            min_x = int(uv[0]-self.panelDim[1]/2)
            min_y = int(uv[1]-self.panelDim[0]/2)
            max_x = int(uv[0]+self.panelDim[1]/2)
            max_y = int(uv[1]+self.panelDim[0]/2)

            if min_x >= 0 and min_y >= 0 and max_x < self.imgDim[1] and max_y < self.imgDim[0]:
                if v[0] < x_l: x_l = v[0]
                if v[1] < y_l: y_l = v[1]
                if v[2] < z_l: z_l = v[2]
                if v[0] > x_u: x_u = v[0]
                if v[1] > y_u: y_u = v[1]
                if v[2] > z_u: z_u = v[2]

        self.xl = np.array([x_l, y_l, z_l])
        self.xu = np.array([x_u, y_u, z_u])
        print(self.xl, self.xu)

        # Setup input domain
        self.domain = ContinuousParameter('x1', self.xl[0], self.xu[0]) + \
                      ContinuousParameter('x2', self.xl[1], self.xu[1]) + \
                      ContinuousParameter('x3', self.xl[2], self.xu[2])


    def problem(self, voxel):
        M, E, CE, MA, R = (np.array([]) for i in range(5))
        for vox in voxel:
            uv = self.w2uv([vox[0]/100, vox[1]/100, vox[2]/100])
            min_x = int(uv[0]-self.panelDim[1]/2)
            min_y = int(uv[1]-self.panelDim[0]/2)
            max_x = int(uv[0]+self.panelDim[1]/2)
            max_y = int(uv[1]+self.panelDim[0]/2)

            x = np.atleast_2d(uv)
            u = x[:, 0]
            v = x[:, 1]

            crop_rect = (min_x, min_y, max_x, max_y)
            imgCrop = self.img.crop(crop_rect)
            arm_pos = self.get_pose(vox)

            M = np.append(M, self.colourfulness(imgCrop))            
            E = np.append(E, self.edgeness(imgCrop))         
               
            if (arm_pos != []):
                CE = np.append(CE, self.consumed_endurance(arm_pos))
                MA = np.append(MA, self.muscle_activation_reserve_function(arm_pos))
                R = np.append(R, self.rula(arm_pos))
            else:
                CE = np.append(CE, 0.0)
                MA = np.append(MA, 0.0)
                R = np.append(R, 0.0)

        return np.hstack((np.transpose([M]), np.transpose([E]), np.transpose([CE]), 
                          np.transpose([MA]), np.transpose([R])))

    
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


    def fittsLaw(self, anchor, pos, label):
        width = label[1]
        a = 0.4926
        b = 0.6332
        dist = math.sqrt( (anchor[0] - pos[0])**2 + (anchor[1] - pos[1])**2 )
        ID = math.log(dist/width+1,2)
        T = a + b*ID
        return T

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
        

    def color(self, uvPos):
        labelColors = []
        textColors = []
        (label, text) = self._color(uvPos)
        labelColors.append(label)
        textColors.append(text)

        return (labelColors, textColors)
        

    def _color(self, uvPos):
        crop_rect = (   uvPos[0] - self.panelDim[1]/2,
                        uvPos[1] - self.panelDim[0]/2, 
                        uvPos[0] + self.panelDim[1]/2,
                        uvPos[1] + self.panelDim[0]/2 )                
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

    def perceivedBrightness(self, color):
        # From: https://www.w3.org/TR/AERT/
        #((Red value X 299) + (Green value X 587) + (Blue value X 114)) / 1000
        b = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
        return b

def main():
    directory = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\"
    img_path = directory + "context_img_1623145003.png"
    out_file = directory + '\\out\\' + 'mobo_out.png'
    f = open(directory + "context_img_buff_1623145003.log", "r")
    img = cv2.imread(img_path)
    byte_arr = bytes(f.read(), 'utf-8')
    print(f'Processing: {img_path}')

    dim = [504, 896]
    panel_dim = [[100, 100], [50, 100], [100, 50]] # panel dimensions in uv (height, width)

    occupancyMap = {}
    for y in range(dim[0]):
        for x in range(dim[1]):
            occupancyMap[(y, x)] = 0

    for panel in panel_dim:
        prob = ObjectiveFunction(byte_arr, np.array(dim), panel)

        #X = FactorialDesign(11, prob.domain).generate()
        X = gpflowopt.design.LatinHyperCube(11, prob.domain).generate()
        Y = prob.problem(X)

        objective_models = [gpflow.gpr.GPR(X.copy(), Y[:,[i]].copy(), gpflow.kernels.Matern52(2, ARD=True)) for i in range(Y.shape[1])]
        
        for model in objective_models:
            model.likelihood.variance = 0.01

        alpha = gpflowopt.acquisition.HVProbabilityOfImprovement(objective_models)

        acquisition_opt = StagedOptimizer([MCOptimizer(prob.domain, 1000),
                                        SciPyOptimizer(prob.domain)])

        # Run the Bayesian optimization
        optimizer = BayesianOptimizer(prob.domain, alpha, optimizer=acquisition_opt, verbose=True)
        r = optimizer.optimize([prob.problem], n_iter=2)
        #print(r)
        
        pt = [r.x[0][0]/100, r.x[0][1]/100, r.x[0][2]/100]
        uv_pt = prob.w2uv(pt)
        (labelColor, textColor) = prob.color([int(uv_pt[0]),int(uv_pt[1])])

        min_x = int(uv_pt[0]-panel[1]/2)
        max_x = int(uv_pt[0]+panel[1]/2)
        min_y = int(uv_pt[1]-panel[0]/2)
        max_y = int(uv_pt[1]+panel[0]/2)
        print(uv_pt, min_x, max_x, min_y, max_y, labelColor[0])
        cv2.rectangle(img, (min_x, min_y), (max_x, max_y), labelColor[0], -1)

    cv2.imwrite(out_file, img)


if __name__ == "__main__":
    main()

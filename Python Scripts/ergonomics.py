import numpy as np
import math
import linalg_helpers as linalg
import arm_position_helpers as armpos


class Ergonomics:
    def __init__(self, arm_proper_length=33, forearm_hand_length=46, spacing=10):
        self.forearm_hand_length = forearm_hand_length
        self.arm_proper_length = arm_proper_length
        self.arm_total_length = arm_proper_length + forearm_hand_length
        self.spacing = spacing


    def get_pose(self, voxel):
        arm_poses = self.compute_anchor_arm_poses(voxel)[0]
        pose = [0, voxel[0], voxel[1], voxel[2], arm_poses['elbow_x'], arm_poses['elbow_y'], 
             arm_poses['elbow_z'], arm_poses['elv_angle'], arm_poses['shoulder_elv'], arm_poses['elbow_flexion']]
        
        return pose

    
    def compute_anchor_arm_poses(self, end_effector, rotation_step=-math.pi / 8, limit=-math.pi * 3 / 4):
        arm_poses = []
        u, v, c, r = armpos.compute_elbow_plane(end_effector, self.arm_proper_length, self.forearm_hand_length)

        # If result is None, pose is not possible with current constraints
        if u is None:
            print("Pose is not possible.")
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


    def compute_consumed_endurance(self, pose):
        print('Computing CE...')
      
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


    def compute_rula(self, pose):
        print('Computing RULA...')
      
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


    def compute_muscle_activation_reserve_function(self, pose):
        print('Computing muscle activation reserve...')

        # we want to give priority to poses with the lowest reserve values. Hence, we use the max reserve value of all
        # voxels, where their reserve value is the minimum between all the poses.
        # voxels that have reserve values among a threshold receive the worst comfort rating (1).
       
        reserve_threshold = 250
        muscle_activation_reserve = pose[1] + pose[2] / reserve_threshold
        return muscle_activation_reserve
    


def main():
    panel_location = [15, 15, 15]
    erg = Ergonomics()
    arm_pos = erg.get_pose(panel_location)
    
    print(arm_pos)
    print(erg.compute_consumed_endurance(arm_pos))
    print(erg.compute_muscle_activation_reserve_function(arm_pos))
    print(erg.compute_rula(arm_pos))

if __name__ == "__main__":
    main()
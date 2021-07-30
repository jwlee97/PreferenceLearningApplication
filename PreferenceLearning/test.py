from kernels import Matern
from posterior import Laplace
from acquisitions import UCB
import numpy as np
import cv2

from UIoptimizer import UIOptimizer
from optimization import ProbitBayesianOptimization

def main():
    directory = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\"
    f = open(directory + "context_img_buff_1623145003.log", "r")
    byte_arr = bytes(f.read(), 'utf-8')
    img_path = directory + "context_img_1623145003.png"

    img_dim = [504, 896]
    panel_dim = [(0.1, 0.15), (0.1, 0.1), (0.2, 0.1), (0.15, 0.1)]
    #panel_dim = [(0.1, 0.15)]
    occlusion = False
    num_panels = 4

    colorfulness = 0.0
    edgeness = 0.0
    fitts_law = 0.0
    ce = 0.33
    muscle_act = 0.33
    rula = 0.33

    ui = UIOptimizer(byte_arr, np.array(img_dim), np.array(panel_dim), num_panels, occlusion, 
                      colorfulness, edgeness, fitts_law, ce, muscle_act, rula)

    ### Preference learning w/ Bayesian Optimization ###
    # 3D example. Initialization.
    X = np.random.sample(size=(2, 3)) * 10
    M = np.array([0, 1]).reshape(-1, 2)

    GP_params = {'kernel': Matern(length_scale=1, nu=2.5),
                'post_approx': Laplace(s_eval=1e-5, max_iter=1000,
                                        eta=0.01, tol=1e-3),
                'acquisition': UCB(kappa=2.576),
                'alpha': 1e-5,
                'random_state': None}

    gpr_opt = ProbitBayesianOptimization(ui, img_path, X, M, GP_params)
    bounds = {'x0': (ui.xl[0], ui.xu[0]), 'x1': (ui.xl[1], ui.xu[1]), 'x2': (ui.xl[2], ui.xu[2])}
    print('Bounds: ', bounds)

    opt_vals, ret_img = gpr_opt.interactive_optimization(bounds=bounds, n_init=100, n_solve=10)
    #optimal_values, suggestion, X_post, M_post, f_post = console_opt
    print('--- Optimal values: ---')
    for val in opt_vals: 
        print(val)
    
    ret_img = cv2.cvtColor(ret_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('Preference Learning output', ret_img)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
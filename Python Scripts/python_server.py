import zmq
import json
import adapt
import numpy as np


def get_panel_info(request):
    info = []
    directory = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\"
    f = open(directory + "context_img_buff_1623145003.log", "r")
    byte_arr = bytes(f.read(), 'utf-8')

    color_harmony_template = 93.6
    img_dim = [504, 896]
    panel_dim = []
    num_panels = request["numPanels"]
    occlusion = request["occlusion"]
    constraints = request["constraints"]
    colorfulness = request["colorfulness"]
    edgeness = request["edgeness"]
    fitts_law = request["fittsLaw"]

    for c in constraints:
        panel_dim.append((c["height"], c["width"]))

    a = adapt.Adapt(byte_arr, np.array(img_dim), np.array(panel_dim), num_panels, occlusion, colorfulness, edgeness, fitts_law)
        
    (labelPos, uvPlaces) = a.weighted_optimization()
    (labelColors, textColors) = a.color(uvPlaces)
   
    for i in range(num_panels):
        dim_str = str(panel_dim[i][0]) + ',' + str(panel_dim[i][1])
        pos_str = str(labelPos[i][0]) + ',' + str(labelPos[i][1]) + ',' + str(labelPos[i][2])
            
        if (i == 0):
            color = a.colorHarmony(labelColors[i], color_harmony_template)
            color_str = str(labelColors[i][0]) + ',' + str(labelColors[i][1]) + ',' + str(labelColors[i][2])
        else:
            color_str = str(color[i][0]) + ',' + str(color[i][1]) + ',' + str(color[i][2])

        text_color_str = str(textColors[i][0]) + ',' + str(textColors[i][1]) + ',' + str(textColors[i][2])
        line =  dim_str + ';' + pos_str + ';' + color_str + ';' + text_color_str
        info.append(line)

    return info


context = zmq.Context()
print("Connecting to server...")
socket = context.socket(zmq.REP)
socket.bind('tcp://*:5555')


while True:
    request = socket.recv_multipart()

    if request[0].decode('utf-8') == 'C':
        req = json.loads(request[1])
        print(req)
        position = get_panel_info(req)
        print(position)
        socket.send(json.dumps(position).encode('utf-8'))
    else:
        socket.send(b'Error')
using System;
using Newtonsoft.Json;

public static class Serialization {        
    [Serializable]
    public class Panel {
        public int id;
        public float[] position;

        public override string ToString()
        {
            return "x: " + position[0] + " " + "y: " + position[1] + " " +
                   "z: " + position[2] + " ";
        }
    }

    
    [Serializable]
    public class ComputePositionRequest {
        public ComputePositionRequest(int numPanels, UITool.PanelConstraints[] constraints, bool occlusion, float colorfulness, float edgeness, float fittsLaw)
        {
            this.numPanels = numPanels;
            this.constraints = constraints;
            this.occlusion = occlusion;
            this.colorfulness = colorfulness;
            this.edgeness = edgeness;
            this.fittsLaw = fittsLaw;
        }
        public int numPanels;
        public UITool.PanelConstraints[] constraints;
        public bool occlusion;
        public float colorfulness;
        public float edgeness;
        public float fittsLaw;
    }
}

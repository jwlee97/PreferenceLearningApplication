using UnityEngine;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;

public class UITool : MonoBehaviour {
    public PinchSlider slider;
    private string panelFile;
    private string fileContents;

    private string[] appNames = {"Messages", "Weather", "Email", "Internet"};
    private GameObject[] panels;
    private float[,] panelSizes;
    private int numPanels;

    public bool UIGenerated;

    public void Start() {
        panelFile = "C:\\Users\\2020\\UNITY\\HololensComms\\Assets\\Images\\out.txt";
        numPanels = 4;
        panels = new GameObject[numPanels];
        panelSizes = new float[numPanels, 2];

        if (slider == null)
            slider = GameObject.FindObjectOfType(typeof(PinchSlider)) as PinchSlider;

        Debug.Log("UI tool initialized.");
        UIGenerated = false;
    }
   
    public void CreateUI() {
        Debug.Log("Creating UI.");
        UIGenerated = true;

        var sr = new StreamReader(panelFile);
        fileContents = sr.ReadToEnd();
        sr.Close();

        var lines = fileContents.Split("\n"[0]);
        int i = 0;
        foreach (var line in lines) {
            panels[i] = InitializePanel(line, i);
            i++;
        }
    }

    private GameObject InitializePanel(string data, int i) {
        string[] spl = data.Split(';');
        float height = float.Parse(spl[0].Split(',')[0]);
        float width = float.Parse(spl[0].Split(',')[1]);
        panelSizes[i, 0] = height;
        panelSizes[i, 1] = width;

        float[] panelPos = {float.Parse(spl[1].Split(',')[0]), float.Parse(spl[1].Split(',')[1]), float.Parse(spl[1].Split(',')[2])};
        Color panelColor = new Color(float.Parse(spl[2].Split(',')[0])/255, float.Parse(spl[2].Split(',')[1])/255, float.Parse(spl[2].Split(',')[2])/255);
        Color textColor = new Color(float.Parse(spl[3].Split(',')[0])/255, float.Parse(spl[3].Split(',')[1])/255, float.Parse(spl[3].Split(',')[2])/255);

        GameObject cube = GameObject.CreatePrimitive(PrimitiveType.Cube);
        cube.transform.position = new Vector3(panelPos[0], panelPos[1], panelPos[2]);
        cube.transform.localScale = new Vector3(width, height, 0.0001f);
        var cubeRenderer = cube.GetComponent<Renderer>();
        cubeRenderer.material.SetColor("_Color", panelColor);

        GameObject label = new GameObject();
        label.transform.parent = cube.transform;
        label.transform.localPosition = cube.transform.localPosition;
        label.transform.localScale = new Vector3(cube.transform.localScale.y, cube.transform.localScale.x, 1.0f);

        RectTransform rectTransform = label.AddComponent<RectTransform>();
        rectTransform.anchoredPosition = new Vector2(0, 0);

        TextMesh textMesh = label.AddComponent<TextMesh>();
        textMesh.text = appNames[i];
        textMesh.color = textColor;
        textMesh.anchor = TextAnchor.UpperCenter;

        return cube;
    }

    public void ChangeLOD() {
        var value = slider.SliderValue;

        if (UIGenerated == false) {
            Debug.Log("UI not generated.");
        } else {
            for (int i = 0; i < numPanels; i++) {
                GameObject go = panels[i];
                TextMesh textMesh = go.transform.GetChild(0).GetComponent<TextMesh>();
                if (value < 0.33) {
                    go.transform.localScale = new Vector3(panelSizes[i, 1], panelSizes[i, 0], 0.001f);
                    textMesh.text = appNames[i];
                    //textMesh.fontSize = 1;
                } else if (value >= 0.33 && value < 0.67) {
                    go.transform.localScale = new Vector3(panelSizes[i, 1], panelSizes[i, 0], 0.001f);
                    textMesh.text = appNames[i].Substring(0, 1);
                    //textMesh.fontSize = 1;
                } else {
                    go.transform.localScale = new Vector3(0.05f, 0.05f, 0.05f);
                    textMesh.text = appNames[i].Substring(0, 1);
                    //textMesh.fontSize = 1;
                }
            }
        }
    }
}
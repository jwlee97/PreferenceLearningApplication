using System;
using UnityEngine;
using Microsoft.MixedReality.Toolkit.Input;
using Microsoft.MixedReality.Toolkit.UI;


public class SceneGenerator : MonoBehaviour {
    public PinchSlider slider;
    public UITool tool;
    public GameObject labScene;
    public GameObject classScene;
    public GameObject officeScene;

    public void Start() {
        if (slider == null)
            slider = GameObject.FindObjectOfType(typeof(PinchSlider)) as PinchSlider;
        if (tool == null)
            tool = GameObject.FindObjectOfType(typeof(UITool)) as UITool;
        if (labScene == null)
            labScene = GameObject.FindGameObjectWithTag("labScene");
        if (classScene == null)
            classScene = GameObject.FindGameObjectWithTag("classScene");
        if (officeScene == null)
            officeScene = GameObject.FindGameObjectWithTag("officeScene");

        DeactivateScenes();
    }

    public void GenerateScene() {
        DeactivateScenes();
        var value = slider.SliderValue;

        if (tool.UIGenerated == false) {
            Debug.Log("UI not generated.");
        } else {
            if (value < 0.33) {
                officeScene.SetActive(true);
            } else if (value >= 0.33 && value < 0.67) {
                classScene.SetActive(true);
            } else {
                labScene.SetActive(true);
            }
        }
    }

    private void DeactivateScenes() {
        labScene.SetActive(false);
        classScene.SetActive(false);
        officeScene.SetActive(false);
    }
}
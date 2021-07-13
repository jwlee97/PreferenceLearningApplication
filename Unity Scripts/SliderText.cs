using UnityEngine;
using UnityEngine.UI;
 
public class SliderText : MonoBehaviour {
    public Slider slider;
    public Text textComponent;
 
    public void SetSliderValue(float sliderValue) {
        textComponent.text = slider.value.ToString();
    }
}
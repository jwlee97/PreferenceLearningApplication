using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

public class PhotoCapture : MonoBehaviour {
    private UnityEngine.Windows.WebCam.PhotoCapture photoCaptureObject_ = null;
    private UnityEngine.Windows.WebCam.CameraParameters cameraParameters_;

    private bool initialized_ = false;
    public bool photoTaken = false;

    [System.Serializable]
    public class ImageObject
    {
        public Matrix4x4 c2wM;
        public Matrix4x4 projM;
        public string imageDataBase64;
    }

    public HololensSocket socket;

    void Start() {
        if (socket == null)
            socket = GameObject.FindObjectOfType(typeof(HololensSocket)) as HololensSocket;
        InitializeCameraObject();
    }


    private void InitializeCameraObject() {

#if !UNITY_EDITOR
        Debug.Log("Camera initialized.");
        
        //Resolution cameraResolution = PhotoCapture.SupportedResolutions.OrderByDescending((res) => res.width * res.height).First();
        Resolution cameraResolution = new Resolution();
        cameraResolution.width = 896;
        cameraResolution.height = 504;

        // Create a PhotoCapture object
        UnityEngine.Windows.WebCam.PhotoCapture.CreateAsync(true, delegate (UnityEngine.Windows.WebCam.PhotoCapture captureObject)
        {
            photoCaptureObject_ = captureObject;            

            cameraParameters_ = new UnityEngine.Windows.WebCam.CameraParameters();
            cameraParameters_.hologramOpacity = 0.0f;            
            cameraParameters_.cameraResolutionWidth = cameraResolution.width;
            cameraParameters_.cameraResolutionHeight = cameraResolution.height;
            cameraParameters_.pixelFormat = UnityEngine.Windows.WebCam.CapturePixelFormat.JPEG;
        });

        initialized_ = true;
#endif

#if UNITY_EDITOR
        Debug.Log("PhotoCapture not supported in Unity Editor.");
        return;
#endif
    }


    // Use this for initialization
    public void TakePhoto() {
        if (!initialized_)
        {
            Debug.Log("PhotoCapture not initialized...");
            return;
        }
        // Activate the camera
        photoCaptureObject_.StartPhotoModeAsync(cameraParameters_, delegate (UnityEngine.Windows.WebCam.PhotoCapture.PhotoCaptureResult result)
        {
            // Take a picture
            Debug.Log("Take Photo!");                
            photoCaptureObject_.TakePhotoAsync(OnPhotoCaptured);
        });
    }


    private void OnPhotoCaptured(UnityEngine.Windows.WebCam.PhotoCapture.PhotoCaptureResult result, UnityEngine.Windows.WebCam.PhotoCaptureFrame photoCaptureFrame) {
        List<byte> imageBufferList = new List<byte>();
        photoTaken = true;

        // Copy the raw IMFMediaBuffer data into our empty byte list.
        photoCaptureFrame.CopyRawImageDataIntoBuffer(imageBufferList);

        Matrix4x4 c2wM = new Matrix4x4();
        Matrix4x4 projM = new Matrix4x4();
        photoCaptureFrame.TryGetCameraToWorldMatrix(out c2wM);
        photoCaptureFrame.TryGetProjectionMatrix(out projM);

        string imageDataBase64 = Convert.ToBase64String(imageBufferList.ToArray());
        Debug.Log("len image: " + System.Text.Encoding.UTF8.GetByteCount(imageDataBase64));

        ImageObject imageObject = new ImageObject();
        imageObject.c2wM = c2wM;
        imageObject.projM = projM;
        imageObject.imageDataBase64 = imageDataBase64;

#if !UNITY_EDITOR
        socket.SendImageData(imageBufferList);
        SaveFile(imageObject);
#endif
    }

     public void SaveFile(ImageObject io)
     {
        System.IO.Directory.CreateDirectory(System.IO.Path.Combine(Application.persistentDataPath, "metadata.txt"));
        string data = io.c2wM.ToString() + "," + io.projM.ToString();
        Debug.Log("Meta data: " + data);
        File.WriteAllText(data, "metadata.txt");
        Debug.Log("Data saved.");
     }
 

    private void OnStoppedPhotoMode(UnityEngine.Windows.WebCam.PhotoCapture.PhotoCaptureResult result) {
        photoCaptureObject_.Dispose();
        photoCaptureObject_ = null;
    }
}
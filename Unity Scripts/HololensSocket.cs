using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

#if !UNITY_EDITOR
    using System.Threading.Tasks;
    using Windows.Networking;
    using Windows.Networking.Sockets;
    using Windows.Storage.Streams;
#endif

//Able to act as a reciever 
public class HololensSocket : MonoBehaviour {
    public String _input = "Waiting";

#if !UNITY_EDITOR
    StreamSocket socket;
    StreamSocketListener listener;
    StreamSocketListenerConnectionReceivedEventArgs args;
    String port;
    String message;
#endif

    // Use this for initialization
    void Start() {
#if !UNITY_EDITOR
        listener = new StreamSocketListener();
        port = "9090";
        listener.ConnectionReceived += Listener_ConnectionReceived;
        listener.Control.KeepAlive = false;

        Listener_Start();
#endif
    }

#if !UNITY_EDITOR
    private async void Listener_Start() {
        Debug.Log("Listener started");
        try {
            await listener.BindServiceNameAsync(port);
        } catch (Exception e) {
            Debug.Log("Error: " + e.Message);
            print("Error: " + e.Message);
        }

        Debug.Log("Listening");
    }

    private async void Listener_ConnectionReceived(StreamSocketListener sender, StreamSocketListenerConnectionReceivedEventArgs a) {
        Debug.Log("Listening");

        try {
            args = a;
            print("Connection established");

            //while (true) {
            //    using (var dw = new DataWriter(a.Socket.OutputStream)) {
            //        dw.WriteString("Hello There");
            //        await dw.StoreAsync();
            //        dw.DetachStream();
            //    }  

            //    using (var dr = new DataReader(a.Socket.InputStream)) {
            //        dr.InputStreamOptions = InputStreamOptions.Partial;
            //        await dr.LoadAsync(12);
            //        var input = dr.ReadString(12);
            //        Debug.Log("received: " + input);
            //        _input = input;
            //    }
            //}
        } catch (Exception e ) {
            Debug.Log("Disconnected! " + e);
        }
    }

    public void SendImageData(List<byte> imageData) {
        Debug.Log("list n-bytes: " + imageData.Count);

        string imageDataBase64 = Convert.ToBase64String(imageData.ToArray());
        int len = System.Text.Encoding.UTF8.GetByteCount(imageDataBase64);

        Debug.Log("encoded n-bytes: " + len);

        _input = imageDataBase64;
        sendData(imageDataBase64);
    }

    public async Task sendData(string message) {
        DataWriter writer;

        // Create the data writer object backed by the in-memory stream. 
        using (writer = new DataWriter(this.args.Socket.OutputStream)) {
            // Set the Unicode character encoding for the output stream
            writer.UnicodeEncoding = Windows.Storage.Streams.UnicodeEncoding.Utf8;
            // Specify the byte order of a stream.
            writer.ByteOrder = Windows.Storage.Streams.ByteOrder.LittleEndian;

            // Gets the size of UTF-8 string.
            writer.MeasureString(message);
            // Write a string value to the output stream.
            writer.WriteString(message);

            // Send the contents of the writer to the backing stream.
            try {
                await writer.StoreAsync();
            }  catch (Exception exception) {
                switch (SocketError.GetStatus(exception.HResult))
                {
                    case SocketErrorStatus.HostNotFound:
                        // Handle HostNotFound Error
                        Debug.Log("HostNotFound Error in sendData");
                        throw;
                    default:
                        // If this is an unknown status it means that the error is fatal and retry will likely fail.
                        Debug.Log("Unknown SocketError in sendData");
                        throw;
                }
            }

            await writer.FlushAsync();
            // In order to prolong the lifetime of the stream, detach it from the DataWriter
            writer.DetachStream();
            writer.Dispose();
        }
    }
#endif
}
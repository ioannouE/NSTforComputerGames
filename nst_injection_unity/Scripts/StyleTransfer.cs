using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Barracuda;
using UnityEditor;

public class StyleTransfer : MonoBehaviour
{

    [Tooltip("Performs the preprocessing and postprocessing steps")]
    public ComputeShader styleTransferShader;

    [Tooltip("Stylize the camera feed")]
    public bool applyStylization = false;

    [Tooltip("The height og the image being fed to the model")]
    public int targetHeight = 640;

    [Tooltip("The model asset file that will be used when performing inference")]
    public NNModel modelAsset;

    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;

    // The compiled model used for performing inference
    private Model m_RuntimeModel;

    // The inference used to execute the neural network
    private IWorker engine;



    // Start is called before the first frame update
    void Start()
    {
        // compile the model asset
        m_RuntimeModel = ModelLoader.Load(modelAsset);

        // initialize inference engine
        engine = WorkerFactory.CreateWorker(workerType, m_RuntimeModel);
    }

    // Update is called once per frame
    void Update()
    {
        // Controls: Enable/Disable style transfer
        if (Input.GetMouseButtonDown(0))
        {
            applyStylization = !applyStylization;
        }
    }

    void OnAfterCameraSubmit(Camera camera)
    {
        if (camera.GetComponent<StyleTransfer>() == null)
        {
            //scene camera
            //OnRenderImage(RenderTexture source, RenderTexture destination);
        }
        else
        {
            //final blit to screen
            //Graphics.Blit(renderTarget, null as RenderTexture);
        }
    }

    private void OnRenderImage(RenderTexture source, RenderTexture destination)
    {
        Debug.Log("On Render Image");
        if (applyStylization)
        {
            StylizeImage(source);
        }
        Graphics.Blit(source, destination);
    }
    /**
     * Stylize the provided image
     * <param name="src"></param>
     */
    private void StylizeImage(RenderTexture src)
    {
        // create a new RenderTexture variable 
        RenderTexture rTex;

        //check if the target display is larger than the targetHeight and
        //make sure the targetHeight is at least 8(src.height > targetHeight && targetHeight >=8)
        if ((src.height > targetHeight && targetHeight >= 8))
        {
            //calculate the scale for reducing the size of the input image
            float scale = src.height / targetHeight;
            // new image width
            int targetWidth = (int)(src.width / scale);

            // adjust the target image dimensions to be multiples of 8
            targetHeight -= (targetHeight % 8);
            targetWidth -= (targetWidth % 8);
            //assign a temporary RenderTexture with the new dimensions
            rTex = RenderTexture.GetTemporary(targetWidth, targetHeight, 24, src.format);
        }
        else
        {
            // Assign a temporary RenderTexture with the src dimensions
            rTex = RenderTexture.GetTemporary(src.width, src.height, 24, src.format);    

            // Copy the src RenderTexture to the new rTex RenderTexture
            Graphics.Blit(src, rTex);

            // Apply preprocessing steps
            ProcessImage(rTex, "ProcessInput");

            // Create a Tensor of shape [1, rTex.height, rTex.width, 3]
            Tensor input = new Tensor(rTex, channels: 3);

            // Execute neural network with the provided input
            engine.Execute(input);

            // Get the raw model output
            Tensor prediction = engine.PeekOutput();
            // Release GPU resources allocated for the Tensor
            input.Dispose();

            // Make sure rTex is not the active RenderTexture
            RenderTexture.active = null;
            // Copy the model output to rTex
            prediction.ToRenderTexture(rTex);
            // Release GPU resources allocated for the Tensor
            prediction.Dispose();

            Debug.Log("Save Output");
            // Create a new Texture2D object to read the pixels from the RenderTexture
            Texture2D texture2D = new Texture2D(rTex.width, rTex.height, TextureFormat.RGBA32, false);
            RenderTexture previousActive = RenderTexture.active;
            RenderTexture.active = (RenderTexture)rTex;
            texture2D.ReadPixels(new Rect(0, 0, rTex.width, rTex.height), 0, 0);
            texture2D.Apply();
            RenderTexture.active = previousActive;

            // Save the texture as an asset in the project's Assets folder
            byte[] bytes = texture2D.EncodeToPNG();
            string path = "Assets/Stylized/stylization.png";
            System.IO.File.WriteAllBytes(path, bytes);
            AssetDatabase.ImportAsset(path);
            TextureImporter importer = (TextureImporter)AssetImporter.GetAtPath(path);
            importer.sRGBTexture = true;
            importer.alphaSource = TextureImporterAlphaSource.None;
            importer.alphaIsTransparency = false;
            importer.SaveAndReimport();

            // Apply post processing steps
            ProcessImage(rTex, "ProcessOutput");
            // Copy rTex into src
            Graphics.Blit(rTex, src);

            // Release the temporary RenderTexture
            RenderTexture.ReleaseTemporary(rTex);
        }
       

    }

    /// <summary>]/// Process the provided image using the specified function on the GPU
    /// </summary>
    /// <param name="image"></param>
    /// <param name="functionName"></param>
    /// <returns>The processed image</returns>
    private void ProcessImage(RenderTexture image, string functionName)
    {
        // number of threads on the GPU
        int numthreads = 8;
        // Get the index of the specified function in the ComputeShader
        int kernelHandle = styleTransferShader.FindKernel(functionName);
        //temporary HDR RenderTexture
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);
        // Enable random write access
        result.enableRandomWrite = true;
        // create the HDR RenderTexture
        result.Create();

        // set the value for the Result variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "Result", result);
        // set the value for the InputImage variable in the ComputeShader
        styleTransferShader.SetTexture(kernelHandle, "InputImage", image);

        // execute the ComputeShader
        styleTransferShader.Dispatch(kernelHandle, result.width / numthreads, result.height / numthreads, 1);
        // copy the result into the source RenderTexture
        Graphics.Blit(result, image);
        // release the temporary RenderTexture
        RenderTexture.ReleaseTemporary(result);

    }


    // OnDisable is called when the MonoBehavior becomes disabled or inactive
    private void OnDisable()
    {
        // Release the resources allocated for the inference engine
        engine.Dispose();
    }
}

using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.HighDefinition;
using Unity.Barracuda;
using UnityEditor;
using UnityEngine.Experimental.Rendering;

/*#if UNITY_EDITOR
    using UnityEditor;
    using UnityEditor.Rendering.HighDefinition;

    [CustomPassDrawer(typeof(CopyPass))]
    public class CopyPassDrawer : CustomPassDrawer
    {
        protected override PassUIFlag commonPassUIFlags => PassUIFlag.Name;
    }
#endif*/

public class CopyPassStylization : CustomPass
{

    public NNModel modelAsset; // Reference to the Barracuda neural network asset
    public ComputeShader styleTransferShader; // Reference to the compute shader responsible for applying the style transfer
    [Tooltip("The height og the image being fed to the model")]
    public int targetHeight;
    private Model styleTransferModel;
    private IWorker styleTransferWorker;
    // The inference used to execute the neural network
    private IWorker engine;
    [Tooltip("The backend used when performing inference")]
    public WorkerFactory.Type workerType = WorkerFactory.Type.Auto;


    public enum BufferType
    {
        Color,
        Normal,
        Roughness,
        Depth,
        MotionVectors,
    }

    public RenderTexture outputRenderTexture;

    [SerializeField, HideInInspector]
    Shader customCopyShader;
    Material customCopyMaterial;

    public BufferType bufferType;

    Shader fullscreenShader;
    public Material fullscreenMaterial;
    RenderTargetIdentifier diffuseTextureIdentifier = new RenderTargetIdentifier("_GBufferTexture0");

    protected override bool executeInSceneView => true;

    int normalPass;
    int roughnessPass;
    int depthPass;
    private RenderTexture diffuseTexture;
    public ComputeShader gbufferShader;
    Material gbufferMaterial;
     
    //private Texture diffuseTxt;

    protected override void Setup(ScriptableRenderContext renderContext, CommandBuffer cmd)
    {
        if (modelAsset != null)
            styleTransferModel = ModelLoader.Load(modelAsset);

        if (styleTransferModel != null)
            styleTransferWorker = WorkerFactory.CreateWorker(styleTransferModel);


        // initialize inference engine
        engine = WorkerFactory.CreateWorker(workerType, styleTransferModel);

        if (customCopyShader == null)
            customCopyShader = Shader.Find("Hidden/FullScreen/CustomCopy");
        customCopyMaterial = CoreUtils.CreateEngineMaterial(customCopyShader);

        // if (gbufferShader == null)
        gbufferShader = Resources.Load<ComputeShader>("GbufferShader");

        normalPass = customCopyMaterial.FindPass("Normal");
        roughnessPass = customCopyMaterial.FindPass("Roughness");
        depthPass = customCopyMaterial.FindPass("Depth");

        if (fullscreenShader == null)
            fullscreenShader = Shader.Find("FullScreen/Fullscreen_NST");
        // fullscreenMaterial = CoreUtils.CreateEngineMaterial(fullscreenShader);
    }

    protected override void Execute(CustomPassContext ctx)
    {
        if (outputRenderTexture == null || customCopyMaterial == null)
            return;

        SyncRenderTextureAspect(outputRenderTexture, ctx.hdCamera.camera);

        var scale = RTHandles.rtHandleProperties.rtHandleScale;
        customCopyMaterial.SetVector("_Scale", scale);

        diffuseTexture = ctx.hdCamera.GetCurrentFrameRT((int)HDCameraFrameHistoryType.ColorBufferMipChain);
        // diffuseTexture = ctx.cameraColorBuffer.rt;
        // int kernelHandle = gbufferShader.FindKernel("CSMain");
        // gbufferShader.SetTextureFromGlobal(kernelHandle, "_GBufferTexture0", "_GBufferTexture0");
        //ctx.cmd.DispatchCompute(gbufferShader, kernelHandle, (int)scale.x,(int) scale.y, 1);

        
        Texture diffuseTxt = Shader.GetGlobalTexture("_GBufferTexture0");
        // //diffuseTexture = RenderTexture.GetTemporary(1920, 1080, 16, RenderTextureFormat.ARGB32);

        // Graphics.Blit(diffuseTxt, diffuseTexture);
        SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);



        // Texture2D texture2D = new Texture2D(diffuseTexture.width, diffuseTexture.height, TextureFormat.RGBAHalf, false);
        // RenderTexture previousActive = RenderTexture.active;
        // RenderTexture.active = (RenderTexture)diffuseTexture;
        // texture2D.ReadPixels(new Rect(0, 0, diffuseTexture.width, diffuseTexture.height), 0, 0);
        // texture2D.Apply();
        // RenderTexture.active = previousActive;

        // // Save the texture as an asset in the project's Assets folder
        // byte[] bytes = texture2D.EncodeToPNG();
        // string path = "Assets/Stylized/6_DiffuseTexture.png";
        // System.IO.File.WriteAllBytes(path, bytes);
        // AssetDatabase.ImportAsset(path);
        // TextureImporter importer = (TextureImporter)AssetImporter.GetAtPath(path);
        // importer.sRGBTexture = true;
        // importer.alphaSource = TextureImporterAlphaSource.None;
        // importer.alphaIsTransparency = false;
        // importer.SaveAndReimport();

        // SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);
        //RenderTexture diffuseTexture = new RenderTexture(1920, 1080, 16, RenderTextureFormat.ARGB32);
        // ctx.cmd.Blit(diffuseTexture, outputRenderTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        // ctx.cmd.Blit(ctx.cameraColorBuffer, diffuseTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);

        //SyncRenderTextureAspect(diffuseTexture, ctx.hdCamera.camera);
        // Graphics.Blit(outputRenderTexture, diffuseTexture);
        StylizeImage(diffuseTexture);
        //Graphics.Blit(diffuseTexture, outputRenderTexture);
        ctx.cmd.Blit(diffuseTexture, outputRenderTexture, new Vector2(scale.x, scale.y), Vector2.zero, 0, 0);
        // int kernelHandle = gbufferShader.FindKernel("CSMain");
        // Debug.Log(kernelHandle);
        Texture2D texture2D = new Texture2D(diffuseTexture.width, diffuseTexture.height, TextureFormat.RGBA32, false);
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = (RenderTexture)diffuseTexture;
        texture2D.ReadPixels(new Rect(0, 0, diffuseTexture.width, diffuseTexture.height), 0, 0);
        texture2D.Apply();
        RenderTexture.active = previousActive;
        //gbufferShader.SetTextureFromGlobal(0, "_GBufferTexture0", "_GBufferTexture0");
        gbufferShader.SetTexture(0, "_GBufferTexture0", ctx.cameraColorBuffer.rt);
        gbufferShader.SetTexture(0, "_StylizedTexture", texture2D);
        ctx.cmd.DispatchCompute(gbufferShader, 0, 64, 64, 1);
        // ctx.cmd.Blit(outputRenderTexture, ctx.cameraColorBuffer);
        /*Texture2D texture2D = new Texture2D(diffuseTexture.width, diffuseTexture.height, TextureFormat.RGBA32, false);
        RenderTexture previousActive = RenderTexture.active;
        RenderTexture.active = (RenderTexture)diffuseTexture;
        texture2D.ReadPixels(new Rect(0, 0, diffuseTexture.width, diffuseTexture.height), 0, 0);
        texture2D.Apply();*/
        //SyncRenderTextureAspect(outputRenderTexture, ctx.hdCamera.camera);
        //fullscreenMaterial.SetTexture("_StylizedTexture", outputRenderTexture);
        //Shader.SetGlobalTexture("_GBufferTexture0", outputRenderTexture, RenderTextureSubElement.Color);

        // CoreUtils.SetRenderTarget(ctx.cmd, ctx.cameraColorBuffer, ClearFlag.All);
        // CoreUtils.DrawFullScreen(ctx.cmd, fullscreenMaterial, ctx.propertyBlock, shaderPassId: 0);
        
        //RenderTexture.ReleaseTemporary(diffuseTexture);
        
        // Texture2D texture2D = new Texture2D(diffuseTexture.width, diffuseTexture.height, TextureFormat.RGBAHalf, false);
        // RenderTexture previousActive = RenderTexture.active;
        // RenderTexture.active = (RenderTexture)diffuseTexture;
        // texture2D.ReadPixels(new Rect(0, 0, diffuseTexture.width, diffuseTexture.height), 0, 0);
        // texture2D.Apply();
        // RenderTexture.active = previousActive;

        // // Save the texture as an asset in the project's Assets folder
        // byte[] bytes = texture2D.EncodeToPNG();
        // string path = "Assets/Stylized/6_DiffuseTexture.png";
        // System.IO.File.WriteAllBytes(path, bytes);
        // AssetDatabase.ImportAsset(path);
        // TextureImporter importer = (TextureImporter)AssetImporter.GetAtPath(path);
        // importer.sRGBTexture = true;
        // importer.alphaSource = TextureImporterAlphaSource.None;
        // importer.alphaIsTransparency = false;
        // importer.SaveAndReimport();

    }


    void SyncRenderTextureAspect(RenderTexture rt, Camera camera)
    {
        float aspect = rt.width / (float)rt.height;

        if (!Mathf.Approximately(aspect, camera.aspect))
        {
            rt.Release();
            rt.width = camera.pixelWidth;
            rt.height = camera.pixelHeight;
            rt.Create();
        }
    }

    protected override void Cleanup()
    {
        base.Cleanup();
        CoreUtils.Destroy(customCopyMaterial);
        // diffuseTexture.Release();
        // diffuseTexture.DiscardContents();

        if (styleTransferWorker != null)
            styleTransferWorker.Dispose();
    }


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
            // Debug.Log(prediction[1]);
            // Debug.Log(prediction.shape + "  " + prediction.channels);

            // Make sure rTex is not the active RenderTexture
            RenderTexture.active = null;
            // Copy the model output to rTex
            prediction.ToRenderTexture(rTex);
            // Release GPU resources allocated for the Tensor
            prediction.Dispose();


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
        RenderTexture result = RenderTexture.GetTemporary(image.width, image.height, 24, RenderTextureFormat.ARGBHalf);// RenderTextureFormat.ARGBHalf);
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
}
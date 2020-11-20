package gruv.apps.opencvproject;

import android.app.Activity;
import android.content.Context;
import android.os.Bundle;
import android.util.Log;
import android.view.WindowManager;

import org.opencv.android.*;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;

public class MainActivity extends Activity
        implements CvCameraViewListener {

    private CameraBridgeViewBase openCvCameraView;
    private CascadeClassifier mCascadeClassifierMain;
    private CascadeClassifier mCascadeClassifierAdditional;
    private Mat grayscaleImage;
    private int absoluteFaceSize;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        openCvCameraView = new JavaCameraView(this, -1);
        setContentView(openCvCameraView);
        openCvCameraView.setCvCameraViewListener(this);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        grayscaleImage = new Mat(height, width, CvType.CV_8UC4);

        // The faces will be a 20% of the height of the screen
        absoluteFaceSize = (int) (height * 0.2);
    }

    @Override
    public void onCameraViewStopped() {
        //
    }

    @Override
    public Mat onCameraFrame(Mat aInputFrame) {
        // Create a grayscale image
        Imgproc.cvtColor(aInputFrame, grayscaleImage, Imgproc.COLOR_RGBA2RGB);

        //---------------------------------------------------------------------------------------------

        MatOfRect faces = new MatOfRect();

        // Use the classifier to detect faces
        if (mCascadeClassifierMain != null) {
            mCascadeClassifierMain.detectMultiScale(grayscaleImage, faces, 1.1, 2, 2,
                    new Size(absoluteFaceSize, absoluteFaceSize), new Size());
        }

        // If there are any faces found, draw a rectangle around it
        Rect[] facesArray = faces.toArray();

        for (Rect face : facesArray) {
            Imgproc.rectangle(aInputFrame, face.tl(), face.br(), new Scalar(0, 255, 0, 255), 3);
        }
        //---------------------------------------------------------------------------------------------
        MatOfRect eyes = new MatOfRect();

        // Use the classifier to detect eyes
        if(mCascadeClassifierAdditional != null) {
            mCascadeClassifierAdditional.detectMultiScale(grayscaleImage, eyes, 1.1, 2, 2,
                    new Size(absoluteFaceSize * 0.3f, absoluteFaceSize * 0.3f), new Size());
        }

        // If there are any eyes found, draw a rectangle around it
        Rect[] eyesArray = eyes.toArray();

        for (Rect eye : eyesArray) {
            Imgproc.rectangle(aInputFrame, eye.tl(), eye.br(), new Scalar(200, 255, 0, 255), 3);
        }
        //---------------------------------------------------------------------------------------------


        return aInputFrame;
    }

    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_4_0, this, mLoaderCallback);
    }

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            if (status == LoaderCallbackInterface.SUCCESS) {
                initializeOpenCVDependencies();
            } else {
                super.onManagerConnected(status);
            }
        }

        private void initializeOpenCVDependencies() {
            mCascadeClassifierMain = getCascade(R.raw.lbpcascade_frontalface, "lbpcascade_frontalface.xml");//лицо анфас
            //mCascadeClassifier = getCascade(R.raw.lbpcascade_frontalcatface, "lbpcascade_frontalcatface.xml");
            //mCascadeClassifier = getCascade(R.raw.lbpcascade_profileface, "lbpcascade_profileface.xml");//лицо профиль
            //mCascadeClassifier = getCascade(R.raw.lbpcascade_silverware, "lbpcascade_silverware.xml");
            //mCascadeClassifier = getCascade(R.raw.haarcascade_eye, "haarcascade_eye.xml");
            mCascadeClassifierAdditional = getCascade(R.raw.haarcascade_eye_tree_eyeglasses, "haarcascade_eye_tree_eyeglasses.xml");
            //mCascadeClassifier = getCascade(R.raw.haarcascade_lefteye_2splits, "haarcascade_lefteye_2splits.xml");

            // And we are ready to go
            openCvCameraView.enableView();
        }

        private CascadeClassifier getCascade(int id, String fileName) {
            File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
            File mCascadeFile = new File(cascadeDir, fileName);

            try (FileOutputStream os = new FileOutputStream(mCascadeFile);  InputStream is = getResources().openRawResource(id)) {
                byte[] buffer = new byte[4096];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    os.write(buffer, 0, bytesRead);
                }

                // Load the cascade classifier
                return new CascadeClassifier(mCascadeFile.getAbsolutePath());
            } catch (Exception e) {
                Log.e("OpenCVActivity", "Error loading cascade", e);
                return null;
            }
        }
    };
}
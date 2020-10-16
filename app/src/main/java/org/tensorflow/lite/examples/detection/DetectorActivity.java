/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.annotation.SuppressLint;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.location.Location;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.SystemClock;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.Toast;

import androidx.appcompat.widget.SwitchCompat;
import androidx.recyclerview.widget.LinearLayoutManager;
import androidx.recyclerview.widget.RecyclerView;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Date;
import java.util.LinkedList;
import java.util.List;
import java.util.Locale;

import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.model.SignEntity;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;
import org.tensorflow.lite.examples.detection.tracking.SignAdapter;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the prepackaged SSD model.
  private static final int TF_OD_API_INPUT_SIZE = 300;
  private static final boolean TF_OD_API_IS_QUANTIZED = false;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/detect_labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private BorderedText borderedText;

  private SignAdapter adapter;
  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e(e, "Exception initializing classifier!");
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });

    tracker.setFrameConfiguration(previewWidth, previewHeight, sensorOrientation);
    setupRecycler();
  }

  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
                runOnUiThread(() -> updateSignList(result, croppedBitmap));
              }
            }

            tracker.trackResults(mappedRecognitions, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }

  private void updateSignList(Classifier.Recognition result, Bitmap bitmap) {

    SignEntity sign = getSignImage(result, bitmap);

    ArrayList<SignEntity> list = new ArrayList<>(adapter.getSigns());

    if (list.isEmpty()) {
      addSignToAdapter(sign);
      return;
    }
    if (list.contains(sign)) {
      if (isRemoveValid(sign, list.get(list.indexOf(sign)))) {
        adapter.getSigns().remove(sign);
        addSignToAdapter(sign);
      }
    } else {
      addSignToAdapter(sign);
    }

  }

  private void addSignToAdapter(SignEntity sign) {
    adapter.setSign(sign);
    if (sign.getSoundNotification() != null){
      setSpeak(sign.getSoundNotification());
    }
  }

  private boolean isRemoveValid(SignEntity sign1, SignEntity sign2) {
    return isTimeDifferenceValid(sign1.getDate(), sign2.getDate())
            || isLocationDifferenceValid(sign1.getLocation(), sign2.getLocation());
  }

  private boolean isTimeDifferenceValid(Date date1, Date date2) {
    long milliseconds = date1.getTime() - date2.getTime();
    Log.i("sign", "isTimeDifferenceValid " + ((milliseconds / (1000)) > 30));
    return (int) (milliseconds / (1000)) > 30;
  }

  private boolean isLocationDifferenceValid(Location location1, Location location2) {
    if (location1 == null || location2 == null)
      return false;
    return location1.distanceTo(location2) > 50;
  }

  private void setupRecycler() {
    adapter = new SignAdapter(this);

    RecyclerView signRecycler = findViewById(R.id.signRecycler);
    signRecycler.setAdapter(adapter);
    signRecycler.setLayoutManager(new LinearLayoutManager(this));
  }
  @Override
  protected int getLayoutId() {
    return R.layout.tfe_od_camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  TextToSpeech speak;
  private void setSpeak(String txt){
    speak = new TextToSpeech(this, i -> {
      if (i == TextToSpeech.SUCCESS){
        speak.setLanguage(Locale.ROOT);
        Toast.makeText(getApplicationContext(), txt,Toast.LENGTH_SHORT).show();
        speak.speak(txt, TextToSpeech.QUEUE_ADD, null);
      }
    });
  }
  private SignEntity getSignImage(Classifier.Recognition result, Bitmap bitmap) {
    SignEntity sign = null;
    if ("Bat Dau Duong Uu Tien".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.camdungvadoxe,"Bắt đầu đường ưu tiên");
    } else if ("Cam Di Nguoc chieu".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Cấm đi ngược chiều");
    }else if ("Cam Vuot".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Cấm vượt");
    }else if ("Cam dung va do xe".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Cấm dừng và đỗ xe");
    }else if ("Canh Bao Co Tuyet".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Cảnh báo có tuyết");
    }else if ("Cong truong".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Công trường");
    }else if ("Duong Cam".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Đường cấm");
    }else if ("Duong tron".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Đường trơn");
    }else if ("Giao Nhau Voi Duong Khong Uu Tien".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Giao nhau với đường không ưu tiên");
    }else if ("Giao Nhau Voi Duong Uu Tien".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Giao nhau với đường ưu tiên");
    }else if ("Huong Di Vong Chuong Ngai Vat Sang Trai".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Hướng đi vòng chướng ngại vật sang trái");
    }else if ("Huong Di Vong Chuong Ngai Vat Sang phai".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Hướng đi vòng chướng ngại vật sang phải");
    }else if ("Ke, Vuc Sau phia truoc".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Kè, vực sâu phía trước");
    }else if ("Nguoi Di Bo Sang Ngang".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Người đi bộ sang ngang");
    }else if ("Nguoi di bo cat ngang".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Người đi bộ cắt ngang");
    }else if ("Nguy Hiem Khac".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Nguy hiểm khác");
    }else if ("Nhieu Cho Ngoat Lien Tiep Ben Trai".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Nhiều chỗ ngoặt liên tiếp bên trái");
    }else if ("Stop".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Stop");
    }else if ("Toc Do Gioi han 100km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"tốc độ giới hạn 100km/h");
    }else if ("Toc Do Gioi han 20km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Tốc độ giới hạn 20km/h");
    }else if ("Toc Do Gioi han 30km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Tốc độ giới hạn 30km/h");
    }else if ("Toc Do Gioi han 50km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Tốc độ giới hạn 50km/h");
    }else if ("Toc Do Gioi han 60km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Tốc độ giới hạn 60km/h");
    }else if ("Toc Do Gioi han 70km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Tốc độ giới hạn 70km/h");
    }else if ("Toc Do Gioi han 80km-h".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Tốc độ giới hạn 0km/h");
    }else if ("cam xe tai".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Cấm xe tải");
    }else if ("tre em".equals(result.getTitle())) {
      sign = new SignEntity(result.getTitle(), R.drawable.sp50km,"Trẻ em");
    }


    if (sign != null) {
      sign.setConfidenceDetection(result.getConfidence());

      sign.setScreenLocation(result.getLocation());
    }

    return sign;
  }
}

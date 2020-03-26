/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
package co.huggingface.android_transformers.bertqa.ml;

import android.content.Context;
import android.content.res.AssetFileDescriptor;
import android.content.res.AssetManager;
import androidx.annotation.WorkerThread;
import android.util.Log;
import com.google.common.base.Joiner;
import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.commons.math3.stat.descriptive.DescriptiveStatistics;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.gpu.GpuDelegate;

/** Interface to load TfLite model and provide predictions. */
public class QaClient {
  private static final String TAG = "BertDemo";
//  private static final String MODEL_PATH = "model.tflite";
  private static final String DIC_PATH = "vocab.txt";

  private static final int MAX_ANS_LEN = 32;
  private static final int MAX_QUERY_LEN = 64;
  private static final int MAX_SEQ_LEN = 384;
  private static final boolean DO_LOWER_CASE = true;
  private static final int PREDICT_ANS_NUM = 5;
  private static final int NUM_LITE_THREADS = 4;

  // Need to shift 1 for outputs ([CLS]).
  private static final int OUTPUT_OFFSET = 1;

  private final Context context;
  private final Map<String, Integer> dic = new HashMap<>();
  private final FeatureConverter featureConverter;
  private Interpreter tflite;

  private static final Joiner SPACE_JOINER = Joiner.on(" ");

  public QaClient(Context context) {
    this.context = context;
    this.featureConverter = new FeatureConverter(dic, DO_LOWER_CASE, MAX_QUERY_LEN, MAX_SEQ_LEN);
  }

//  @WorkerThread
//  public synchronized void loadModel() {
//    try {
//      ByteBuffer buffer = loadModelFile(this.context.getAssets());
////      Interpreter.Options opt = (new Interpreter.Options()).addDelegate(delegate);
//
//      Interpreter.Options opt = new Interpreter.Options();
//      GpuDelegate delegate = new GpuDelegate();
//      opt.addDelegate(delegate);
//
////      opt.setNumThreads(NUM_LITE_THREADS);
//      tflite = new Interpreter(buffer, opt);
//      Log.v(TAG, "TFLite model loaded.");
//    } catch (IOException ex) {
//      Log.e(TAG, ex.getMessage());
//    }
//  }

  boolean enableGpuDelegate = true;

  @WorkerThread
  public synchronized void loadModel(String modelName) {
    try {
      ByteBuffer buffer = loadModelFile(this.context.getAssets(), modelName);
      Interpreter.Options opt = new Interpreter.Options();


      if(enableGpuDelegate) {
        GpuDelegate delegate = new GpuDelegate();
        opt.addDelegate(delegate);
      }

      opt.setNumThreads(NUM_LITE_THREADS);

      tflite = new Interpreter(buffer, opt);
      Log.v(TAG, "TFLite model loaded: " + (enableGpuDelegate ? "GPU": "CPU"));
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  @WorkerThread
  public synchronized void loadDictionary() {
    try {
      loadDictionaryFile(this.context.getAssets());
      Log.v(TAG, "Dictionary loaded.");
    } catch (IOException ex) {
      Log.e(TAG, ex.getMessage());
    }
  }

  @WorkerThread
  public synchronized void unload() {
    tflite.close();
    dic.clear();
  }

//  /** Load tflite model from assets. */
//  public MappedByteBuffer loadModelFile(AssetManager assetManager) throws IOException {
//    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(MODEL_PATH);
//         FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
//      FileChannel fileChannel = inputStream.getChannel();
//      long startOffset = fileDescriptor.getStartOffset();
//      long declaredLength = fileDescriptor.getDeclaredLength();
//      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
//    }
//  }

  /** Load tflite model from assets. */
  public MappedByteBuffer loadModelFile(AssetManager assetManager, String modelName) throws IOException {
  //    File file = new File(context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS), modelName);
  //    Log.v(TAG, "File path: " + file.getAbsolutePath());
    try (AssetFileDescriptor fileDescriptor = assetManager.openFd(modelName);
         FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor())) {
      FileChannel fileChannel = inputStream.getChannel();
      long startOffset = fileDescriptor.getStartOffset();
      long declaredLength = fileDescriptor.getDeclaredLength();
      return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }
  }

  /** Load dictionary from assets. */
  public void loadDictionaryFile(AssetManager assetManager) throws IOException {
    try (InputStream ins = assetManager.open(DIC_PATH);
        BufferedReader reader = new BufferedReader(new InputStreamReader(ins))) {
      int index = 0;
      while (reader.ready()) {
        String key = reader.readLine();
        dic.put(key, index++);
      }
    }
  }

  //// Models were exported with the following bash script:
  /*
  #!/bin/bash

  for hiddenLayers in $(seq 2 2 12); do
    for attentionLayers in 2 4 8 12; do
      hiddenSize=$((attentionLayers*64))
      for max_seq_length in 384 ; do
        modelName="uncased_L-${hiddenLayers}_H-${hiddenSize}_A-${attentionLayers}"
        modelDir="./data/bert_24/${modelName}"
        echo ${modelDir}

        python3 run_ckpt_model.py --mode lite \
        --bert_config_file ${modelDir}/bert_config.json \
        --ckpt_file ${modelDir}/bert_model.ckpt \
        --max_seq_length ${max_seq_length} \
        --output_dir data/bert_24/tflite_models \
        --output_name ${modelName}
      done
    done
  done
   */

  public List<String> getModelNames() {
    List<String> modelNames = new ArrayList<String>();
//    modelNames.add("uncased_L-12_H-256_A-4.tflite");

    for(int hiddenLayers = 2; hiddenLayers <= 12; hiddenLayers+=2) {
      for(int attentionHeads : new int[] {2, 4, 8, 12}) {
        int hiddenSize = attentionHeads * 64;
        String modelName = "uncased_L-" + hiddenLayers + "_H-"
                + hiddenSize + "_A-" + attentionHeads + ".tflite";
        modelNames.add(modelName);
      }
    }

    return modelNames;
  }

  /**
   * Input: Original content and query for the QA task. Later converted to Feature by
   * FeatureConverter. Output: A String[] array of answers and a float[] array of corresponding
   * logits.
   */
  @WorkerThread
  public synchronized List<QaAnswer> predict(String query, String content) {
//    Log.v(TAG, "TFLite model: " + MODEL_PATH + " running...");
    Log.v(TAG, "Convert Feature...");
    Feature feature = featureConverter.convert(query, content);

    Log.v(TAG, "Set inputs...");
    int[][] inputIds = new int[1][MAX_SEQ_LEN];
//    int[][] inputMask = new int[1][MAX_SEQ_LEN];
//    int[][] segmentIds = new int[1][MAX_SEQ_LEN];
//    float[][] startLogits = new float[1][MAX_SEQ_LEN];
    float[][] probTensor = new float[1][2];
    float[][] endLogits = new float[1][MAX_SEQ_LEN];

    for (int j = 0; j < MAX_SEQ_LEN; j++) {
      inputIds[0][j] = feature.inputIds[j];
//      inputMask[0][j] = feature.inputMask[j];
//      segmentIds[0][j] = feature.segmentIds[j];
    }

//    Object[] inputs = {inputIds, inputMask, segmentIds};
    Map<Integer, Object> output = new HashMap<>();
    output.put(0, probTensor);
//    output.put(1, endLogits);

    Log.v(TAG, "Run inference...");
    List<String> modelNames = getModelNames();

    for (String modelName : modelNames) {
      loadModel(modelName);
      Log.v(TAG, "##### " + modelName);
      DescriptiveStatistics stats = new DescriptiveStatistics();
      
      for (int i = 0; i < 10; i++) {
        long beforeTime = System.currentTimeMillis();
        tflite.runForMultipleInputsOutputs(new Object[]{inputIds}, output);
        long afterTime = System.currentTimeMillis();
        double totalMilliseconds = (afterTime - beforeTime);

        Log.v(TAG, modelName + ", iteration " + i + ", latency: " + totalMilliseconds + " s");

        if(i == 0) {
          Log.v(TAG, "Ignoring the first inference");
          continue;
        }

        stats.addValue(totalMilliseconds);
      }

      Log.v(TAG,"Mean latency: " + stats.getMean() + " ms, std: " + stats.getStandardDeviation() + ", " + modelName + " " + (enableGpuDelegate ? "GPU" : "CPU"));
    }

    Log.v(TAG, "Convert answers...");
    List<QaAnswer> answers = new ArrayList<>(); //getBestAnswers(probTensor[0], endLogits[0], feature);
    answers.add(new QaAnswer("[DEFAULT]", 0, 0, 0));
    Log.v(TAG, "Finish.");
    return answers;
  }

  /** Find the Best N answers & logits from the logits array and input feature. */
  private synchronized List<QaAnswer> getBestAnswers(
      float[] startLogits, float[] endLogits, Feature feature) {
    // Model uses the closed interval [start, end] for indices.
    int[] startIndexes = getBestIndex(startLogits, feature.tokenToOrigMap);
    int[] endIndexes = getBestIndex(endLogits, feature.tokenToOrigMap);

    List<QaAnswer.Pos> origResults = new ArrayList<>();
    for (int start : startIndexes) {
      for (int end : endIndexes) {
        if (end < start) {
          continue;
        }
        int length = end - start + 1;
        if (length > MAX_ANS_LEN) {
          continue;
        }
        origResults.add(new QaAnswer.Pos(start, end, startLogits[start] + endLogits[end]));
      }
    }

    Collections.sort(origResults);

    List<QaAnswer> answers = new ArrayList<>();
    for (int i = 0; i < origResults.size(); i++) {
      if (i >= PREDICT_ANS_NUM) {
        break;
      }

      String convertedText;
      if (origResults.get(i).start > 0) {
        convertedText = convertBack(feature, origResults.get(i).start, origResults.get(i).end);
      } else {
        convertedText = "";
      }
      QaAnswer ans = new QaAnswer(convertedText, origResults.get(i));
      answers.add(ans);
    }
    return answers;
  }

  /** Get the n-best logits from a list of all the logits. */
  @WorkerThread
  private synchronized int[] getBestIndex(float[] logits, Map<Integer, Integer> tokenToOrigMap) {
    List<QaAnswer.Pos> tmpList = new ArrayList<>();
    for (int i = 0; i < MAX_SEQ_LEN; i++) {
      if (tokenToOrigMap.containsKey(i + OUTPUT_OFFSET)) {
        tmpList.add(new QaAnswer.Pos(i, i, logits[i]));
      }
    }

    Collections.sort(tmpList);

    int[] indexes = new int[PREDICT_ANS_NUM];
    for (int i = 0; i < PREDICT_ANS_NUM; i++) {
      indexes[i] = tmpList.get(i).start;
    }

    return indexes;
  }

  /** Convert the answer back to original text form. */
  @WorkerThread
  private static String convertBack(Feature feature, int start, int end) {
     // Shifted index is: index of logits + offset.
    int shiftedStart = start + OUTPUT_OFFSET;
    int shiftedEnd = end + OUTPUT_OFFSET;
    int startIndex = feature.tokenToOrigMap.get(shiftedStart);
    int endIndex = feature.tokenToOrigMap.get(shiftedEnd);
    // end + 1 for the closed interval.
    String ans = SPACE_JOINER.join(feature.origTokens.subList(startIndex, endIndex + 1));
    return ans;
  }
}

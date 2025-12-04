import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:image/image.dart' as img;
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:flutter/services.dart' show rootBundle;

class ClassificationModel {
  Interpreter? _interpreter;
  List<String> labels = [];
  String classification = '';
  int _height = 160; // adjust to your model input
  int _width = 160;

  late List<List<double>> result;
  bool isReady = false; // flag to check if model and labels are loaded

  /// Load the TFLite model and initialize the result array
  Future<void> loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/best_cls_float32.tflite');
      if (kDebugMode) print("Classifier model loaded(yolo!)");

      // Initialize result based on model output shape
      var outputShape = _interpreter!.getOutputTensor(0).shape;
      result = List.generate(outputShape[0], (_) => List.filled(outputShape[1], 0.0));

      _checkReady();
    } catch (e) {
      if (kDebugMode) print("Classifier load error: $e");
      isReady = false;
    }
  }

  /// Load labels from assets
  Future<void> loadLabels() async {
    try {
      final String labelsData = await rootBundle.loadString('assets/labels.txt');
      labels = labelsData.split('\n').map((e) => e.trim()).toList();
      if (kDebugMode) print("Labels loaded: ${labels.length}");

      _checkReady();
    } catch (e) {
      labels = [];
      if (kDebugMode) print("Labels load error: $e");
      isReady = false;
    }
  }

  /// Check if both model and labels are loaded
  void _checkReady() {
    if (_interpreter != null && labels.isNotEmpty) {
      isReady = true;
      if (kDebugMode) print("Classifier is ready for use!");
    }
  }

  /// Preprocess image to model input size
  img.Image preprocessImage(Uint8List imageBytes) {
    final decoded = img.decodeImage(imageBytes);
    if (decoded == null) {
      throw Exception("Cannot decode image");
    }
    return img.copyResize(decoded, width: _width, height: _height);
  }

  /// Convert preprocessed image to Float32 tensor (normalized 0-1)
  Float32List imageToTensor(img.Image image) {
    final convertedBytes = Float32List(1 * _height * _width * 3);
    int pixelIndex = 0;

    for (int y = 0; y < _height; y++) {
      for (int x = 0; x < _width; x++) {
        final pixel = image.getPixel(x, y);
        convertedBytes[pixelIndex++] = img.getRed(pixel) / 255.0;
        convertedBytes[pixelIndex++] = img.getGreen(pixel) / 255.0;
        convertedBytes[pixelIndex++] = img.getBlue(pixel) / 255.0;
      }
    }

    return convertedBytes;
  }

  /// Classify image given its file path
  Future<void> classifyImage(String path) async {
    if (!isReady) {
      if (kDebugMode) print("Classifier not ready yet!");
      return;
    }

    final file = File(path);
    if (!await file.exists()) {
      if (kDebugMode) print("Image file does not exist: $path");
      return;
    }

    try {
      Uint8List imageBytes = await file.readAsBytes();
      final processedImage = preprocessImage(imageBytes);
      final inputTensor = imageToTensor(processedImage).reshape([1, _height, _width, 3]);

      _interpreter!.run(inputTensor, result);

      // Find max prediction
      final output = result[0];
      int maxIndex = 0;
      double maxValue = output[0];

      for (int i = 1; i < output.length; i++) {
        if (output[i] > maxValue) {
          maxValue = output[i];
          maxIndex = i;
        }
      }

      classification = (maxIndex < labels.length) ? labels[maxIndex] : '';
      if (kDebugMode) print("Predicted: $classification");
    } catch (e) {
      if (kDebugMode) print("TFLite run error: $e");
      classification = '';
    }
  }

  /// Clear previous classification
  void clearClassification() {
    classification = '';
  }

  /// Dispose interpreter safely
  void dispose() {
    _interpreter?.close();
    _interpreter = null;
    isReady = false;
  }
}

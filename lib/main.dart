import 'dart:io';
import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:image_picker/image_picker.dart';
import 'classification.dart';
import 'extraction.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  final cameras = await availableCameras();
  runApp(MyApp(cameras: cameras));
}

class MyApp extends StatelessWidget {
  final List<CameraDescription> cameras;
  const MyApp({super.key, required this.cameras});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      home: ImageClassificationScreen(cameras: cameras),
    );
  }
}

class ImageClassificationScreen extends StatefulWidget {
  final List<CameraDescription> cameras;
  const ImageClassificationScreen({super.key, required this.cameras});

  @override
  _ImageClassificationScreenState createState() =>
      _ImageClassificationScreenState();
}

class _ImageClassificationScreenState extends State<ImageClassificationScreen> {
  final ClassificationModel _classifier = ClassificationModel();
  final ExtractionModel _extractor = ExtractionModel();

  late CameraController _cameraController;
  late Future<void> _initializeControllerFuture;

  String? _imagePath;
  String _classification = '';
  String _reading = '';
  List<Map<String, dynamic>> _detectedBoxes = [];
  bool _isCameraInitialized = false;
  bool _modelsReady = false;

  final ImagePicker _picker = ImagePicker();

  @override
  void initState() {
    super.initState();
    _loadModels();
    _initializeCamera();
  }

  Future<void> _loadModels() async {
    await _classifier.loadModel();
    await _classifier.loadLabels();
    await _extractor.loadModel();
    setState(() {
      _modelsReady = true;
    });
  }


  _initializeCamera() {
    _cameraController = CameraController(
      widget.cameras.first,
      ResolutionPreset.medium, // Use a higher resolution for better quality
    );
    _initializeControllerFuture = _cameraController.initialize().then((_) {
      setState(() {
        _isCameraInitialized = true;
        _imagePath = null;
        _classification = '';
        _reading = '';
      });
    });
  }


  Future<void> _captureImage() async {
    if (_cameraController == null || !_cameraController!.value.isInitialized) return;

    try {
      final file = await _cameraController!.takePicture();
      setState(() {
        _imagePath = file.path;
        _classification = '';
        _reading = '';
        _detectedBoxes.clear();
      });
    } catch (e) {
      if (mounted) print("Capture error: $e");
    }
  }

  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(source: ImageSource.gallery);
    if (image != null) {
      setState(() {
        _imagePath = image.path;
        _classification = ""; // Reset classification
        _reading = "";
        _detectedBoxes.clear();
      });
    }
  }

  Future<void> _classifyAndDetect() async {
    if (!_modelsReady || _imagePath == null) return;

    await _classifier.classifyImage(_imagePath!);

    setState(() {
      _classification = _classifier.classification;
    });

    if (_classifier.classification.trim() == 'Meter') {
      await _extractor.detect(_imagePath!);
      setState(() {
        _reading = _extractor.reading;
        _detectedBoxes = _extractor.detectedBoxes;
      });
    }



  }

  @override
  void dispose() {
    _cameraController?.dispose();
    _classifier.dispose();
    _extractor.close();
    super.dispose();
  }

  @override
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('DPDC Meter Reading')),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const SizedBox(height: 20),
              // ===== BUTTONS =====
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  ElevatedButton(
                    onPressed: _pickImage,
                    child: const Text('Gallery'),
                  ),
                  const SizedBox(width: 10),
                  ElevatedButton(
                    onPressed: _initializeCamera,
                    child: const Text('Camera'),
                  ),
                  const SizedBox(width: 10),
                  ElevatedButton(
                    onPressed: _captureImage,
                    child: const Text('Capture'),
                  ),
                ],
              ),
              const SizedBox(height: 20),

              // ===== IMAGE DISPLAY =====
              if (_imagePath != null)
                LayoutBuilder(
                  builder: (context, constraints) {
                    final displayWidth = constraints.maxWidth;
                    final displayHeight = displayWidth *
                        _extractor.imageActualHeight /
                        _extractor.imageActualWidth;

                    return Container(
                      width: displayWidth,
                      height: displayHeight,
                      decoration:
                      BoxDecoration(border: Border.all(color: Colors.grey)),
                      child: Stack(
                        children: [
                          Image.file(
                            File(_imagePath!),
                            width: displayWidth,
                            height: displayHeight,
                            fit: BoxFit.fill,
                          ),
                          if (_detectedBoxes.isNotEmpty)
                            CustomPaint(
                              size: Size(displayWidth, displayHeight),
                              painter: _extractor.boxPainter(
                                  displayWidth, displayHeight),
                            ),
                        ],
                      ),
                    );
                  },
                )
              else if (_isCameraInitialized && _cameraController != null)
                LayoutBuilder(
                  builder: (context, constraints) {
                    final displayWidth = constraints.maxWidth;
                    final displayHeight = displayWidth * 320 / 480; // camera ratio
                    return Container(
                      width: displayWidth,
                      height: displayHeight,
                      decoration:
                      BoxDecoration(border: Border.all(color: Colors.grey)),
                      child: CameraPreview(_cameraController!),
                    );
                  },
                )
              else
                Container(
                  width: 480,
                  height: 320,
                  decoration: BoxDecoration(border: Border.all(color: Colors.grey)),
                  child: const Center(child: Text('No image selected')),
                ),

              const SizedBox(height: 20),

              // ===== CLASSIFY BUTTON =====
              ElevatedButton(
                onPressed: (_modelsReady && _imagePath != null)
                    ? _classifyAndDetect
                    : null,
                child: const Text('Classify Image'),
              ),
              const SizedBox(height: 30),

              // ===== RESULTS =====
              Text(
                'Classified as: $_classification',
                style: const TextStyle(
                    fontSize: 18, color: Colors.red, fontWeight: FontWeight.bold),
              ),
              Text(
                'Reading: $_reading',
                style: const TextStyle(
                    fontSize: 18, color: Colors.red, fontWeight: FontWeight.bold),
              ),
            ],
          ),
        ),
      ),
    );
  }

}

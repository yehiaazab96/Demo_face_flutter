import 'dart:convert';
import 'dart:ffi';
import 'dart:io' as io;
import 'dart:math';
import 'dart:typed_data';
import 'dart:ui';
import 'package:face_detect/firebase_options.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:flutter/material.dart';
import 'package:flutter/rendering.dart';
import 'package:flutter/services.dart';
import 'package:image/image.dart' as imglib;
import 'dart:async';
import 'package:flutter_face_api/face_api.dart' as Regula;
// import 'package:flutter_face_api_example/firebase_options.dart';
import 'package:image_picker/image_picker.dart';
// import 'package:sqflite/sqflite.dart';
// import 'package:sqflite/sqlite_api.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:path_provider/path_provider.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  runApp(new MaterialApp(home: new MyApp()));
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  bool isLoading = false;
  var image1 = new Regula.MatchFacesImage();
  List<StorageImage> filteredImages = [];
  var image2 = new Regula.MatchFacesImage();
  var img1 = Image.asset('assets/images/portrait.png');
  var img2 = Image.asset('assets/images/portrait.png');
  List? img2PredictedData;
  String _similarity = "nil";
  double threshold = 1.4;

  String? _name = "";
  Interpreter? _interpreter;
  List _predictedData = [];
  bool foundMatch = false;

  String _liveness = "nil";
  List<StorageImage> images = [];

  @override
  void initState() {
    super.initState();
    initPlatformState();
    initialize();
    openFireStore();
    // openDatabasePath();
  }

  Future initialize() async {
    late Delegate delegate;
    try {
      if (io.Platform.isAndroid) {
        // delegate = GpuDelegateV2(
        //   options: GpuDelegateOptionsV2(
        //     isPrecisionLossAllowed: false,
        //     inferencePreference: TfLiteGpuInferenceUsage.fastSingleAnswer,
        //     inferencePriority1: TfLiteGpuInferencePriority.minLatency,
        //     inferencePriority2: TfLiteGpuInferencePriority.auto,
        //     inferencePriority3: TfLiteGpuInferencePriority.auto,
        //   ),
        // );
      } else if (io.Platform.isIOS) {
        // delegate = GpuDelegate(
        //   options: GpuDelegateOptions(
        //       allowPrecisionLoss: true,
        //       waitType: TFLGpuDelegateWaitType.active),
        // );
        // delegate = GpuDelegate(
        //   options: GpuDelegateOptions(
        //       allowPrecisionLoss: true,
        //       waitType: TFLGpuDelegateWaitType.active),
        // );
      }
      // var interpreterOptions = InterpreterOptions()..addDelegate(delegate);

      _interpreter = await Interpreter.fromAsset(
        'mobilefacenet.tflite',
        // options: interpreterOptions
      );
    } catch (e) {
      print('Failed to load model.');
      print(e);
    }
  }

  openFireStore() async {
    final storageRef = FirebaseStorage.instance.ref();
    Reference imagesRef = storageRef.child("images");
    final listResult = await imagesRef.listAll();
    print(listResult.items.length);

    setState(() {
      isLoading = true;
    });
    for (var item in listResult.items) {
      final islandRef = storageRef.child(item.fullPath);

      try {
        const oneMegabyte = 1024 * 1024;
        final Uint8List? data = await islandRef.getData(oneMegabyte);
        // print(data);
        images.add(StorageImage(
            name: item.name,
            path: 'gs://facedemoapp.appspot.com/' + item.fullPath,
            byteSync: data));

        // Data for "images/island.jpg" is returned, use this as needed.
      } on FirebaseException catch (e) {
        // Handle any errors.
      }
    }
    setState(() {
      isLoading = false;
    });
  }

  // openDatabasePath() async {
  //   var databasesPath = await getDatabasesPath();
  //   String path = databasesPath + '/demo.db';
  //   open(path);
  // }

  Future<void> initPlatformState() async {}

  showAlertDialog(BuildContext context, bool first) => showDialog(
      context: context,
      builder: (BuildContext context) =>
          AlertDialog(title: Text("Select option"), actions: [
            // ignore: deprecated_member_use
            FlatButton(
                child: Text("Use gallery"),
                onPressed: () {
                  ImagePicker().pickImage(source: ImageSource.gallery).then(
                      (value) => setImage(
                          first,
                          io.File(value!.path).readAsBytesSync(),
                          Regula.ImageType.PRINTED));
                  Navigator.pop(context);
                }),
            // ignore: deprecated_member_use
            FlatButton(
                child: Text("Use camera"),
                onPressed: () {
                  Regula.FaceSDK.presentFaceCaptureActivity().then((result) {
                    print(result);
                    setImage(
                        first,
                        base64Decode(Regula.FaceCaptureResponse.fromJson(
                                json.decode(result))!
                            .image!
                            .bitmap!
                            .replaceAll("\n", "")),
                        Regula.ImageType.LIVE);
                  });
                  Navigator.pop(context);
                })
          ]));

  Float32List imageToByteListFloat32(imglib.Image image) {
    var convertedBytes = Float32List(1 * 112 * 112 * 3);
    var buffer = Float32List.view(convertedBytes.buffer);
    int pixelIndex = 0;

    for (var i = 0; i < 112; i++) {
      for (var j = 0; j < 112; j++) {
        var pixel = image.getPixel(j, i);
        buffer[pixelIndex++] = (imglib.getRed(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getGreen(pixel) - 128) / 128;
        buffer[pixelIndex++] = (imglib.getBlue(pixel) - 128) / 128;
      }
    }
    return convertedBytes.buffer.asFloat32List();
  }

  double _euclideanDistance(List? e1, List? e2) {
    if (e1 == null || e2 == null) throw Exception("Null argument");

    double sum = 0.0;
    for (int i = 0; i < e1.length; i++) {
      sum += pow((e1[i] - e2[i]), 2);
    }
    return sqrt(sum);
  }

  setImage(bool first, List<int> imageFile, int type) {
    if (imageFile == null) return;
    setState(() => _similarity = "nil");
    if (first) {
      image1.bitmap = base64Encode(imageFile);
      image1.imageType = type;
      imglib.Image? libIm = imglib.decodeImage(imageFile);
      imglib.Image img = imglib.copyResizeCropSquare(libIm!, 112);
      List imageAsList = imageToByteListFloat32(img);
      imageAsList = imageAsList.reshape([1, 112, 112, 3]);
      List output = List.generate(1, (index) => List.filled(192, 0));
      _interpreter?.run(imageAsList, output);
      output = output.reshape([192]);
      img2PredictedData = List.from(output);
      print(img2PredictedData);

      setState(() {
        img1 = Image.memory(imageFile as Uint8List);
        _liveness = "nil";
      });
    } else {
      image2.bitmap = base64Encode(imageFile);
      image2.imageType = type;

      setState(() => img2 = Image.memory(imageFile as Uint8List));
    }
  }

  clearResults() {
    setState(() {
      img1 = Image.asset('assets/images/portrait.png');
      img2 = Image.asset('assets/images/portrait.png');
      _similarity = "nil";
      _liveness = "nil";
    });
    image1 = new Regula.MatchFacesImage();
    image2 = new Regula.MatchFacesImage();
  }

  matchFaces() {
    if (image1 == null ||
        image1.bitmap == null ||
        image1.bitmap == "" ||
        image2 == null ||
        image2.bitmap == null ||
        image2.bitmap == "") return;
    setState(() => _similarity = "Processing...");
    var request = new Regula.MatchFacesRequest();
    request.images = [image1, image2];
    Regula.FaceSDK.matchFaces(jsonEncode(request)).then((value) {
      var response = Regula.MatchFacesResponse.fromJson(json.decode(value))!;
      Regula.FaceSDK.matchFacesSimilarityThresholdSplit(
              jsonEncode(response.results), 0.75)
          .then((str) {
        var split = Regula.MatchFacesSimilarityThresholdSplit.fromJson(
            json.decode(str));
        setState(() => _similarity = split!.matchedFaces.length > 0
            ? ((split.matchedFaces[0]!.similarity! * 100).toStringAsFixed(2) +
                "%")
            : "error");
      });
    });
  }

  SearchFormatchFaces() async {
    foundMatch = false;
    filteredImages = [];
    if (image1 == null || image1.bitmap == null || image1.bitmap == "") return;

    setState(() {
      _similarity = "Processing...";
      isLoading = true;
    });
    var request = new Regula.MatchFacesRequest();
    int num = 0;
    Future.delayed(Duration(milliseconds: 0), () {
      images.forEach((element) {
        imglib.Image? libIm = imglib.decodeImage(element.byteSync!);
        imglib.Image img = imglib.copyResizeCropSquare(libIm!, 112);
        List imageAsList = imageToByteListFloat32(img);
        // print(imageAsList);
        imageAsList = imageAsList.reshape([1, 112, 112, 3]);
        List output = List.generate(1, (index) => List.filled(192, 0));
        _interpreter?.run(imageAsList, output);
        output = output.reshape([192]);
        List<dynamic> predictedData = List.from(output);

        double minDist = 999;
        double currDist = 0.0;
        // print(predictedData);
        // print(img2PredictedData);
        currDist = _euclideanDistance(predictedData, img2PredictedData);
        print(currDist);
        print(element.name);
        if (currDist <= threshold && currDist < minDist) {
          minDist = currDist;
          element.currDist = currDist;
          filteredImages.add(element);
          print('added new Image');
        }
      });
      filteredImages.sort((a, b) => a.currDist!.compareTo(b.currDist!));

      print(filteredImages.length);

      loopOverFiltered(filteredImages,
          10 > filteredImages.length ? filteredImages.length : 10, request);

      Future.delayed(Duration(seconds: 60), () {
        setState(() {
          isLoading = false;
        });
      });
    });
  }

  loopOverFiltered(List<StorageImage> filteredImages, int index,
      Regula.MatchFacesRequest request) {
    for (var i = index - 10; i < index; i++) {
      print(i);

      image2.bitmap = base64Encode(filteredImages[i].byteSync!);
      image2.imageType = Regula.ImageType.PRINTED;
      if (!foundMatch) {
        request.images = [image1, image2];
        Regula.FaceSDK.matchFaces(jsonEncode(request)).then((value) {
          var response =
              Regula.MatchFacesResponse.fromJson(json.decode(value))!;
          Regula.FaceSDK.matchFacesSimilarityThresholdSplit(
                  jsonEncode(response.results), 0.75)
              .then((str) {
            var split = Regula.MatchFacesSimilarityThresholdSplit.fromJson(
                json.decode(str))!;

            if (!foundMatch &&
                i == index - 1 &&
                i != filteredImages.length - 1) {
              loopOverFiltered(
                  filteredImages,
                  index + 10 > filteredImages.length
                      ? filteredImages.length
                      : index + 10,
                  request);
            }
            if (split.matchedFaces.isNotEmpty &&
                split.matchedFaces[0]!.similarity! * 100 > 90) {
              setState(() {
                _similarity = split.matchedFaces.length > 0
                    ? ((split.matchedFaces[0]!.similarity! * 100)
                            .toStringAsFixed(2) +
                        "%")
                    : "error";

                foundMatch = true;
                isLoading = false;
                _name = filteredImages[i].name;
                createSnackBar(
                    'Found a Match and the similarity is ' +
                        (split.matchedFaces[0]!.similarity! * 100)
                            .toStringAsFixed(2) +
                        '%',
                    context);
              });
            }
          });
        });
      }
    }
  }

  // liveness() => Regula.FaceSDK.startLiveness().then((value) {
  //       var result = Regula.LivenessResponse.fromJson(json.decode(value));
  //       setImage(true, base64Decode(result.bitmap.replaceAll("\n", "")),
  //           Regula.ImageType.LIVE);
  //       setState(() => _liveness = result.liveness == 0 ? "passed" : "unknown");
  //     });

  Widget createButton(String text, VoidCallback onPress) => Container(
        // ignore: deprecated_member_use
        child: FlatButton(
            color: Color.fromARGB(50, 10, 10, 10),
            onPressed: onPress,
            child: Text(text)),
        width: 250,
      );

  Widget createImage(image, VoidCallback onPress) => Material(
          child: InkWell(
        onTap: onPress,
        child: Container(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(20.0),
            child: Image(height: 150, width: 150, image: image),
          ),
        ),
      ));

  // void saveImage() {
  //   insert(image2);
  // }

  // Database db;

  // Future open(String path) async {
  //   db = await openDatabase(path, version: 1,
  //       onCreate: (Database db, int version) async {
  //     await db.execute('''
  //   create table Users (
  //   _id integer primary key autoincrement,
  //   imageType integer ,
  //   detectAll integer ,
  //   bitmap text ,
  //   identifier text )
  //   ''');
  //   });
  // }

  // Future insert(Regula.MatchFacesImage image) async {
  //   print(image2);
  //   print(image.toMap());
  //   var id = await db.insert('Users', image.toMap(),
  //       nullColumnHack: Random().nextInt(100).toString());
  //   if (id != null) {
  //     setState(() {
  //       // img2 = null;
  //       image2 = new Regula.MatchFacesImage();
  //     });

  //     return id;
  //   }
  // }

  // Future clearDatabase() async {
  //   await db.delete('Users');
  // }

  // Future<List<Regula.MatchFacesImage>> getImages() async {
  //   List<Map> maps = await db.query('Users');
  //   List<Regula.MatchFacesImage> images = [];
  //   if (maps.length > 0) {
  //     return maps.map((e) => Regula.MatchFacesImage().fromMap(e)).toList();
  //   }
  //   return null;
  // }

  // Future close() async => db.close();

  @override
  Widget build(BuildContext context) {
    // isLoading = false;
    return Scaffold(
      body: Stack(
        children: [
          Container(
              margin: EdgeInsets.fromLTRB(0, 0, 0, 100),
              width: double.infinity,
              child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: <Widget>[
                    // createImage(
                    //     img2.image, () => showAlertDialog(context, false)),
                    // createButton("Save to Database", () => saveImage()),

                    // Container(margin: EdgeInsets.fromLTRB(0, 0, 0, 15)),
                    // createButton("Match", () => matchFaces()),

                    createImage(
                        img1.image, () => showAlertDialog(context, true)),
                    createButton(
                        "Search for Match", () => SearchFormatchFaces()),
                    Container(margin: EdgeInsets.fromLTRB(0, 0, 0, 15)),

                    // createButton("Liveness", () => liveness()),
                    createButton("Clear", () => clearResults()),
                    // createButton("Clear Database", () => clearDatabase()),

                    Container(
                        margin: EdgeInsets.fromLTRB(0, 15, 0, 0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text("Similarity: " + _similarity,
                                style: TextStyle(fontSize: 18)),
                            Container(margin: EdgeInsets.fromLTRB(20, 0, 0, 0)),
                            // Text("Liveness: " + _liveness,
                            //     style: TextStyle(fontSize: 18))
                          ],
                        )),

                    Container(
                        margin: EdgeInsets.fromLTRB(0, 15, 0, 0),
                        child: Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Text("name: " + _name!,
                                style: TextStyle(fontSize: 18)),
                            Container(margin: EdgeInsets.fromLTRB(20, 0, 0, 0)),
                            // Text("Liveness: " + _liveness,
                            //     style: TextStyle(fontSize: 18))
                          ],
                        ))
                  ])),
          if (isLoading)
            Container(
                height: MediaQuery.of(context).size.height,
                width: MediaQuery.of(context).size.width,
                color: Colors.blue.withOpacity(0.6),
                child: Center(child: CircularProgressIndicator())),
        ],
      ),
    );
  }
}

void createSnackBar(String message, BuildContext context) {
  final snackBar =
      new SnackBar(content: new Text(message), backgroundColor: Colors.red);

  ScaffoldMessenger.of(context).showSnackBar(snackBar);
}

extension DatabaseMatchFacesImage on Regula.MatchFacesImage {
  Map toMap() {
    Map<String, Object?> result = {};

    if (imageType != null) result.addAll({"imageType": imageType});
    if (detectAll != null) result.addAll({"detectAll": boolToInt(detectAll!)});
    if (bitmap != null) result.addAll({"bitmap": bitmap});
    if (identifier != null) result.addAll({"identifier": identifier});

    return result;
  }

  Regula.MatchFacesImage? fromMap(jsonObject) {
    if (jsonObject == null) return null;
    var result = new Regula.MatchFacesImage();

    result.imageType = jsonObject["imageType"];
    result.detectAll = intToBool(jsonObject["detectAll"]);
    result.bitmap = jsonObject["bitmap"];
    result.identifier = jsonObject["identifier"];

    return result;
  }

  static int boolToInt(bool Bool) {
    if (Bool) {
      return 1;
    } else {
      return 0;
    }
  }

  static bool intToBool(int? Int) {
    if (Int == 1) {
      return true;
    } else {
      return false;
    }
  }
}

class StorageImage {
  String? name;
  double? currDist;
  String? path;
  List<int>? byteSync;
  StorageImage({this.name, this.path, this.byteSync, this.currDist});
}

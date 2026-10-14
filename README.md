# EdgeLens

EdgeLens is a general-purpose vision teaching tool for junior-high classrooms: connect cameras, process images, run models, collect labelled samples, and investigate camera and model performance.

繁體中文：[README.zh-TW.md](README.zh-TW.md)

## What you can do

- **Connect image sources:** built-in and USB webcams, iPhone cameras exposed by the operating system, image/video files, HTTP MJPEG, RTSP, ESP32-CAM, and K210 boards running MaixPy v1. Test patterns and a simulated ESP32-CAM let students practise without hardware.
- **Explore image processing:** view cameras side by side; crop, resize, filter, threshold, mask colours, and analyse contours or motion. Pick colours, save pipelines, and add Python plugins.
- **Try object detection right away:** "Built-in models" downloads YOLO26 models trained on Objects365 (365 everyday objects, class names in Traditional Chinese or English) plus sample photos, so no model or camera hunting is needed.
- **Run classification and detection:** ONNX, TFLite, Darknet (`.weights` plus `.cfg`), TensorFlow/Caffe formats through OpenCV DNN, Haar cascades, and K210 `.kmodel` inference on the board. Decoders cover common YOLO outputs, SSD and MobileNet classification. Compatibility depends on the exported format; see [model support and tested configurations](docs/MODEL_SUPPORT.md).
- **Build datasets:** class shortcuts, burst/countdown/stillness capture, quality checks, similarity reminders, composition references, multi-camera capture, relabelling, deletion/restoration, splitting and ZIP export.
- **Record experiments:** FPS, latency, long-duration tests, parameter sweeps, CSV logs and video. Train models on an external platform, then use EdgeLens for testing.

## Quick start

1. Install **Python 3.10–3.13** from [python.org](https://www.python.org/downloads/). On Windows, tick **Add python.exe to PATH**. Linux also needs `venv` and `pip` for the selected Python version.
2. Download the repository ZIP from its code page (for example, Code → Download ZIP on GitHub). **Extract the entire archive** into a local folder before running anything.
3. Double-click the launcher for your system:

   | System | Launcher |
   |---|---|
   | Windows | [run_windows.bat](run_windows.bat) |
   | macOS | [run_macos.command](run_macos.command) |
   | Linux | [run_linux.sh](run_linux.sh) |

   On the first macOS launch, right-click `run_macos.command` → Open. If downloaded-file quarantine still blocks it, verify that you trust the ZIP, then run `xattr -dr com.apple.quarantine "/full/path/to/EdgeLens-folder"` in Terminal and try again. If executable permission is missing, run `chmod +x run_macos.command` from the extracted folder. On Linux, enable execution in the file manager; if double-clicking opens an editor, run `bash run_linux.sh` from that folder.
4. The first run downloads approximately **300–400 MB** and may take several minutes. Keep the terminal open and connected. The launcher creates a dedicated Python environment in the local cache. Optional packages such as TFLite may be unavailable on some platforms; basic camera and processing functions remain usable.
5. Choose a workspace (the folder for projects and photos). The default is `Documents/EdgeLens`, or `EdgeLens` in your home folder when Documents is synced by iCloud or OneDrive. If you pick another folder, **keep it outside cloud-synced folders**, or photos may be moved to the cloud. In the default Traditional Chinese interface, select 「新增相機」 → 「測試圖案（不需相機）」 → 「尋找」, select a result, and click 「開啟」. A moving test pattern confirms that it works. Switch the interface through 「語言」 → 「English」 if preferred.

Continue with the [student guide](docs/STUDENT_GUIDE.md). The guides below are currently written in Traditional Chinese.

## Offline installation

Prepare packages on an online computer with the **same operating system, CPU architecture and Python minor version** as the classroom computers. From the extracted EdgeLens folder, run:

```bash
python3 bootstrap.py --make-wheelhouse ./wheelhouse
```

On Windows, use `py -3.12` instead of `python3`, adjusting the version to the one installed. Copy the full EdgeLens folder, `wheelhouse/`, and the appropriate Python installer to each offline computer. Install Python first, then run from the EdgeLens folder:

```bash
python3 bootstrap.py --wheelhouse ./wheelhouse
```

Again, use `py -3.12` on Windows as appropriate. Launchers also forward these options: `run_windows.bat --wheelhouse .\wheelhouse`, `bash run_macos.command --wheelhouse ./wheelhouse`, or `bash run_linux.sh --wheelhouse ./wheelhouse`. After successful setup, double-click the launcher for normal use.

Check the download output for skipped optional packages and test one offline computer before deploying the class. A wheelhouse does not include Python, models, device drivers or firmware. See the [teacher guide](docs/TEACHER_GUIDE.md).

## Documentation

| Guide | Contents |
|---|---|
| [Student guide](docs/STUDENT_GUIDE.md) | First launch, capture, dataset cleanup and export |
| [Teacher guide](docs/TEACHER_GUIDE.md) | Classroom setup, project management, lessons and assessment |
| [Plugin tutorial](docs/PLUGINS.md) | Custom processing steps and three Python examples |
| [Model support](docs/MODEL_SUPPORT.md) | Formats, decoders, tested models and limitations |
| [Webcam / iPhone](docs/hardware/WEBCAM_IPHONE.md) | Local cameras and phones |
| [Network cameras](docs/hardware/NETWORK_CAMERAS.md) | MJPEG and RTSP |
| [ESP32-CAM](docs/hardware/ESP32CAM.md) | Wi-Fi cameras and simulation |
| [K210](docs/hardware/K210.md) | Serial connections, board software and hardware precautions |
| [Development](docs/DEVELOPMENT.md) | Architecture, extension interfaces and tests |

## License and acknowledgements

EdgeLens uses the [MIT license](LICENSE). Third-party packages, models and firmware retain their own licenses.

Thanks to the Python, Qt/PySide6, OpenCV, NumPy, pySerial, cv2-enumerate-cameras, ONNX Runtime, LiteRT/TensorFlow Lite, Sipeed MaixPy, kflash and Espressif communities, and to the teachers and students who contribute classroom feedback.

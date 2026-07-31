# Windows 完整安裝指南(含 Azure Kinect)

> 目的:在一台全新的 Windows 電腦上把 FIELD viewer + Azure Kinect backend 裝到能跑。
> 照順序做即可;只想跑 MediaPipe(不接 Kinect)做完 §1–§4 就結束。
> 最後更新:2026-07-31(branch `feat/kinect-pose-source`)。

## 0. 系統需求

| 項目 | 需求 |
|---|---|
| OS | Windows 10/11 x64(Kinect backend 不支援 macOS/Linux 版 viewer;Mac 只能跑 MediaPipe) |
| GPU | 任何 DX12 GPU 都能跑 DirectML body tracking;建議 NVIDIA 獨顯(實測 RTX 4080 = 30fps,Intel 內顯 ≈ 6fps 不堪用)。**不需要安裝 CUDA Toolkit** |
| USB | Azure Kinect 需要 **USB 3.0** 實體埠(接 hub 常出問題)+ 電源(DK 的 Y 型線或外接電源) |
| 其他 | Git |

## 1. Python 3.10

裝 **Python 3.10.x**(python.org 安裝器,勾 "Add python.exe to PATH")。
mediapipe / pykinect_azure 都在 3.10 驗證過;3.12+ 未測。

確認:

```powershell
py -3.10 --version
```

> 本文件所有 `python` 指令,若機器上有多個 Python,一律用 `py -3.10` 代替,
> 避免抓到別的直譯器(這在原開發機上就踩過:`python` 指到別的 venv)。

## 2. 取得程式碼

```powershell
git clone https://github.com/DazaiStudio/field-realtime-dance.git
cd field-realtime-dance
git checkout feat/kinect-pose-source   # Kinect backend 目前在這條分支(合併 master 後可省略)
```

## 3. Python 套件

```powershell
py -3.10 -m pip install -r requirements.txt
```

選配(要什麼裝什麼):

| 功能 | 指令 | 備註 |
|---|---|---|
| Azure Kinect backend | `py -3.10 -m pip install pykinect_azure` | 還需要 §5 的兩個 SDK |
| Stable ID(MediaPipe 用) | `py -3.10 -m pip install ultralytics lap` | Kinect backend 不需要(原生多人);YOLO 權重首次使用自動下載 |
| RTMPose3D backend | 見 `requirements.txt` 註解 | 排練 UI 目前隱藏此選項,一般不用裝 |

注意:`mediapipe` 需 ≥ 0.10.35(protobuf 6 相容;0.10.20 會在建 PoseLandmarker 時 crash)。
`pose_landmarker_*.task` 模型檔首次啟動會自動下載。

## 4. 跑起來(MediaPipe 驗證)

```powershell
py -3.10 backend\osc_viewer.py
```

瀏覽器開 http://127.0.0.1:9100 → 選相機 → Enter → 有畫面 + 骨架就算通過。
(OSC 預設送 127.0.0.1:9000;`backend\osc_monitor.py` 可監看。)

## 5. Azure Kinect SDK(兩個都要裝)

官方直連(Microsoft 簽章,裝到預設路徑,**不要改安裝位置** — 程式依預設路徑偵測):

| SDK | 大小 | 下載 |
|---|---|---|
| Sensor SDK **v1.4.2** | 32 MB | https://download.microsoft.com/download/d/c/1/dc1f8a76-1ef2-4a1a-ac89-a7e22b3da491/Azure%20Kinect%20SDK%201.4.2.exe |
| Body Tracking SDK **1.1.2** | 1.6 GB | https://download.microsoft.com/download/b/4/6/b469e83e-7884-4bd9-a284-1959cd2c0b76/Azure%20Kinect%20Body%20Tracking%20SDK%201.1.2.msi |

連結失效時:Sensor SDK 列表在 GitHub `microsoft/Azure-Kinect-Sensor-SDK` 的 `docs/usage.md`;Body Tracking 在 Microsoft 下載中心 id=104221。原開發機留有一份安裝檔備份(`C:\Users\tommy\Downloads\kinect-sdk\`),用隨身碟帶走可以省 1.6 GB 下載。

靜默安裝(系統管理員 PowerShell):

```powershell
Start-Process ".\Azure Kinect SDK 1.4.2.exe" -ArgumentList "/quiet","/norestart" -Wait
Start-Process msiexec -ArgumentList "/i","Azure Kinect Body Tracking SDK 1.1.2.msi","/qn","/norestart" -Wait
```

裝完應存在:
- `C:\Program Files\Azure Kinect SDK v1.4.2\`(含 `tools\k4aviewer.exe`)
- `C:\Program Files\Azure Kinect Body Tracking SDK\`(含 `tools\k4abt_simple_3d_viewer.exe`)

## 6. 環境變數(視機器調整)

| 變數 | 預設 | 說明 |
|---|---|---|
| `FIELD_KINECT_GPU` | `1` | **DirectML GPU adapter 編號,最容易踩的一個。** 雙顯卡筆電 adapter 0 通常是內顯(只有 ~6fps),所以預設 1;**單顯卡桌機要設 `0`**,否則 tracker 建立失敗、viewer 自動 fallback 回 MediaPipe |
| `FIELD_KINECT_MODEL` | `full` | `full` \| `lite`;4080 上兩者皆 30fps,弱 GPU 可換 lite |
| `FIELD_POSE_BACKEND` | `mediapipe` | 開機預設 backend;可設 `azure_kinect` |
| `FIELD_DEFAULT_CALIBRATION_PRESET` | `Rebecca_Clibrate_001_0630` | 開機自動套用的校準 preset(**是 MediaPipe 錄的**,Kinect 需重新校準;設空字串停用) |
| `FIELD_OSC_ENABLED` | `1` | 測試時不想發 OSC 可設 0 |

設定方式(當前使用者永久生效):

```powershell
[Environment]::SetEnvironmentVariable("FIELD_KINECT_GPU", "0", "User")   # 單顯卡機器範例
```

## 7. Kinect 驗證(接上裝置,USB3)

按順序,每步過了再下一步:

```powershell
# a. 硬體:裝置管理員應出現 Azure Kinect 4K Camera / Depth Camera / Microphone Array
#    或直接開官方檢視器(能看到彩色+深度畫面即硬體 OK):
& "C:\Program Files\Azure Kinect SDK v1.4.2\tools\k4aviewer.exe"

# b. Python 環境 + body tracking(repo 根目錄):
py -3.10 check_kinect_env.py
#    期望:k4a.dll + k4abt.dll loaded OK / devices: 1 / capture + body tracking OK ... fps

# c. 快照(color + depth + 骨架 overlay,存 qa_screenshots\kinect_snapshot.png):
py -3.10 snap_kinect.py

# d. Viewer 整合:
py -3.10 backend\osc_viewer.py
#    UI: Detection model 選「Azure Kinect」(camera 下拉會自動跳到 Kinect 並反灰)
#    → 勾 Stable ID → Enter → 30fps 骨架;Kinect view 可切 Color/Depth
```

## 8. 從舊機器搬過來的東西(git 以外)

| 檔案 | 位置 | 說明 |
|---|---|---|
| 校準 presets | `backend\calibration_presets.json` | 使用者錄的 preset(內建 preset 在 repo 裡不用搬)。**注意:MediaPipe 錄的 preset 在 Kinect backend 下範圍會失真(expansion/height/sway/jerk/torque),Kinect 要重新校準一組** |
| SDK 安裝檔備份 | `C:\Users\tommy\Downloads\kinect-sdk\` | 省下載(見 §5) |

## 9. 常見問題

- **fps 只有 ~6** → `FIELD_KINECT_GPU` 指到內顯了,見 §6。
- **Backend 下拉沒有「Azure Kinect」** → 缺其中之一:pykinect_azure 未裝 / 兩個 SDK 沒在預設路徑 / 不是 Windows。
- **選了 Azure Kinect 卻跑成 MediaPipe** → tracker 建立失敗自動 fallback(看 console 的 `[PoseEngine] Azure Kinect unavailable`),最常見原因還是 GPU 編號錯。
- **Kinect 開不起來(device open failed)** → RGB 鏡頭被別的程式以 webcam(UVC)佔住(OBS、瀏覽器、camera 下拉停在「Azure Kinect 4K Camera」的 preview)。關掉佔用者再 Enter。
- **CUDA / TensorRT mode** → 不要用。SDK 內附的 ONNX Runtime 1.10 CUDA provider 在新世代 GPU(Ada)上初始化失敗,DirectML 已足夠(30fps)。
- **深度範圍** → NFOV unbinned 約 0.5–3.9 m,超出範圍的人追不到;進劇場先量舞台深度。
- **Mac(Nick/Mark)** → Body Tracking SDK 無 macOS 版,Mac 永遠用 MediaPipe backend;OSC 契約相同,收端無感。

## 10. 已知實測數據(RTX 4080 Laptop,2026-07-30)

- 30 fps(相機上限),tracker 推論 ~11 ms/frame(full 與 lite 皆同)
- 單人追蹤 + per-person OSC(`/field/<id>/<metric>`)實測通過
- 首次建 tracker(DirectML 編譯)約 3 秒,首次推論再多幾秒暖機

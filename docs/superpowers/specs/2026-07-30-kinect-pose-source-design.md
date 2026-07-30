# Azure Kinect Pose Source — 設計規格

- 日期:2026-07-30
- 分支:`feat/kinect-pose-source`(開在 `research/stable-id-tracking` 上,需要 per-person 管線)
- 來源:MAINTENANCE.md §7.5 路線圖 + Tommy 2026-07-30 拍板
- 環境前提:Sensor SDK v1.4.2 + Body Tracking SDK 1.1.2 + pykinect_azure 0.0.4(Python 3.10)已裝好並實機驗證(30 fps @ RTX 4080,`gpu_device_id=1`)

## 1. 目標

把 Azure Kinect 接進 viewer 當作新的 pose backend,取得三個 RGB 鏡頭給不了的能力:

1. **暗場/投影光下可用**(ToF 深度不吃可見光)
2. **真 3D 公厘座標**(dance_metrics 的原生單位,比 MediaPipe 推估準)
3. **原生多人 body tracking**(取代 YOLO 偵測層,深度分人比 RGB bbox 可靠)

OSC wire contract v2(`/field/<id>/<metric>`、`/field/<id>/sk/<joint>`)**完全不變**,Nick/Mark 無感。

## 2. 使用者決策(2026-07-30)

| 決策 | 選擇 |
|---|---|
| 預覽畫面 | **color / depth 可切換**(彩色深度圖;暗場排練用 depth) |
| 鏡像 | **完整支援,與 MediaPipe 路徑行為一致**(畫面翻 + 資料 x 取反 + 左右對調) |
| master §3a 修復 | 先不處理(獨立於本功能) |

## 3. 方案選擇

- **A(採用)**:`KinectRuntime` 同時擔任 frame source 與骨架資料源。`read()` = capture → body tracking → 快取骨架 → 回傳顯示畫面;`AzureKinectPoseSource.estimate()` 只消費快取。PoseSource 協議 / PoseEngine / registry / OSC 均不動。
- B(否決):Kinect RGB 走現有 cv2 UVC 路 + pose source 另開 k4a → 同裝置雙取像路徑,USB 頻寬衝突、色深不同步、做不了 depth 視圖。
- C(否決,留待日後):把 camera 全域鎖機制整套搬進 frame_sources → 踩 §5 地雷區,回歸風險高。

## 4. 架構

```
stream_live()
  └─ frame_source = make_frame_source(backend)     # 新:依 backend 選擇
       ├─ OpenCVFrameSource  → 委派現有 open_camera/read_camera_frame/release_camera(行為零改變)
       └─ KinectFrameSource(KinectRuntime)
            read():
              device capture → k4abt enqueue/pop
              → bodies 正規化成純 numpy:[(body_id, joints(32,4) mm+conf), ...] 快取
              → 回傳 color 或彩色深度圖(BGR, 依 kinect_view)
  └─ engine.process_frame(frame, ts)               # 不變
       └─ AzureKinectPoseSource.estimate(frame, ts, draw)
            ├─ 取 runtime 快取 bodies(與 frame 同一次 capture)
            ├─ mirror 開啟 → mirror_h36m17() + bbox 翻轉
            ├─ k4abt32_to_h36m17() per body
            ├─ PersonTrack(raw_id=body_id, bbox=關節投影框, conf) → MultiPersonTrackRegistry.update()
            ├─ choose_active() → last_tracking / last_h36m_by_id / quality
            └─ 畫 overlay(bbox + stable id + 骨架,沿用現有風格)
  └─ metrics / OSC / MJPEG                          # 不變
```

## 5. 新檔案與修改點

### 5a. `backend/frame_sources.py`(新)

```python
class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def release(self) -> None: ...
    def describe(self) -> str: ...          # 錯誤訊息用
```

- `OpenCVFrameSource(index, owner)`:純委派 osc_viewer 現有函式,不搬邏輯、不改鎖語意。
- `KinectFrameSource(runtime)`:薄包裝 `KinectRuntime.read()`。
- 選擇函式 `make_frame_source(backend, ...)` 由 `stream_live()` 呼叫;讀取失敗的 reopen 重試路徑對 Kinect 對應到 runtime 的 close→reopen。
- `/preview_stream` 不動(cv2 專用;Kinect 模式下 UI 不提供 preview)。

### 5b. `backend/pose_backends/azure_kinect.py`(新;用現成空目錄)

**`KinectRuntime`** — 唯一碰 pykinect_azure 的地方:

- 生命週期:`initialize_libraries` 每進程一次(lazy);device+tracker 每 stream session 開關。
- 組態:720P color、NFOV unbinned、30 fps;DirectML `gpu_device_id` 讀 env `FIELD_KINECT_GPU`(**預設 1** — 本機 adapter 0 是內顯,只有 6 fps;此值必須在 `initialize_libraries()` 之後設,init 會重設 processing_mode);model 讀 env `FIELD_KINECT_MODEL`(`full`|`lite`,預設 full,實測兩者都 30fps/11ms)。
- **pykinect 呼叫全部包 guard**:其 `VERIFY()` 失敗會 `sys.exit(1)` → 以 `except (Exception, SystemExit)` 攔下轉成 `KinectError`。拔線 = stream 錯誤 + 重試,絕不殺進程。
- `read()`:capture → enqueue → pop → 每個 body 轉 `(body_id, joints(32,4))`(xyz mm + confidence 0–3,純 numpy)存 `last_bodies` → 依 `view`(`color`|`depth`)回傳 BGR。深度圖用 SDK colorized depth。
- 關節 2D 投影(overlay/bbox 用):k4a calibration `3d→2d` 到當前 view 的相機;投影也在 runtime 做,pose source 拿到的是 view 座標系的 2D 點。

**`AzureKinectPoseSource`** — 實作 PoseSource duck-type 協議:

- `estimate(frame, timestamp_ms, draw)` → `(annotated_frame, h36m | None)`
- `configure_tracking(...)` / `reset_tracking()`:Stable ID 開 → registry 全人輸出;關 → 只出 active(largest)一人,`last_h36m_by_id=None`(契約:單人固定 id=1,由 PoseEngine 現有 fallback 處理)。
- 屬性:`last_pose_quality`、`last_pose_valid`、`last_tracking`(格式與 MediaPipe 版相同:enabled/state/count/active_id/tracks/bbox/...)、`last_h36m_by_id`。
- quality gate(修 RTMPose3D 沒做的課題):K4ABT confidence {0:0.0, 1:0.4, 2:0.8, 3:1.0} 之 13 個實體關節均值 → `last_pose_quality`;核心四關節(雙髖雙肩)任何一個 =NONE → `last_pose_valid=False`。
- 不用 K4ABT 的 `holding` 概念 — registry 的 hold/re-id 語意原樣沿用(body_id 斷號時由 registry 以 bbox 幾何接回 stable id)。

### 5c. `backend/keypoint_mapping.py`(修改)

新增 `k4abt32_to_h36m17(joints: np.ndarray) -> np.ndarray`,走共用 `_assemble_h36m17`:

| H36M 目標 | K4ABT 來源(index) |
|---|---|
| pelvis | (HIP_LEFT 18 + HIP_RIGHT 22)/2 — 與另兩個 mapping 同慣例,**不用** K4ABT PELVIS(0) |
| r_hip / r_knee / r_ank | 22 / 23 / 24 |
| l_hip / l_knee / l_ank | 18 / 19 / 20 |
| l_sh / l_el / l_wr | 5 / 6 / 7 |
| r_sh / r_el / r_wr | 12 / 13 / 14 |
| head(nose) | NOSE 27 |
| spine / thorax / neck | 衍生(同 `_assemble_h36m17`:mid-shoulders 等),**不用** K4ABT SPINE_NAVEL/SPINE_CHEST/NECK |

- 座標:K4A depth 相機系(x 右、y 下、z 前)、單位 mm、輸出前 pelvis 置根。`dance_metrics` 已軸向無關(height 用 foot→upper-body 軸投影),直接可用。
- 新增 `mirror_h36m17(j17)`:x 取反 + 左右關節 index 對調(1–3↔4–6、11–13↔14–16),純函式。

### 5d. `backend/osc_viewer.py`(修改,最小侵入)

- `available_pose_backends()`:probe 通過(Windows + pykinect_azure 可 import + 兩個 SDK 目錄存在)才加 `{"id": "azure_kinect", "label": "Azure Kinect (3D)"}`。macOS 自然不出現。
- `stream_live()`:取像段換成 frame source 物件;Kinect 模式下 `camera_index` 忽略。
- `kinect_view`(`color`|`depth`)進 `source_state` + `/api/apply` 表單參數;UI 下拉只在 backend=azure_kinect 時顯示(照地雷 #4 加 15Hz ws sync 的 dirty/focus guard;新常數走 `%PLACEHOLDER%` 注入,不手抄第二份)。
- 錯誤顯示:裝置不存在/開啟失敗 → `processing_state["error"]` 明確訊息(如「Azure Kinect not detected」)。

## 6. 行為細節

- **30Hz 恆常餵 tracker**:每次 `read()` 都 enqueue(K4ABT 時序追蹤需要連續幀;實測 11ms/frame,4080 餘裕大),`estimate()` 消費最新快取,分析節奏(analysis_fps)照舊。
- **鏡像**:mirror 開 → 顯示幀翻轉(現有 `apply_live_mirror` 之後的幀就是翻轉的)→ 骨架 `mirror_h36m17()`、bbox/投影點做同幅翻轉,overlay 與資料一致。
- **多人**:registry 參數(hold_seconds、reidentify_*)沿用預設;body_id 為 raw_id。舞者離場→該 stable id 靜默;回場 registry 幾何接回(契約 §8 不變)。
- **效能護欄**:tracker pop timeout 用有限值(非 INFINITE),超時當掉幀處理,不卡 stream loop。

## 7. 測試計畫(unittest,沿用現有 fake 模式)

1. `test_keypoint_mapping.py` 擴充:k4abt32 → h36m17 黃金樣本(手算小骨架)、pelvis 置根、衍生關節、`mirror_h36m17` 對合(mirror∘mirror = identity、左右互換)。
2. `test_pose_backends_kinect.py`(新):fake runtime(純 numpy bodies,不 import pykinect)→ 驗 estimate 流程:registry 餵入、stable id 輸出、quality gate(NONE 核心關節 → invalid)、Stable ID 開/關語意、`last_tracking` 結構。
3. `test_frame_sources.py`(新):協議 + OpenCV 委派(mock osc_viewer 函式)+ Kinect 讀取失敗路徑。
4. 現有 branch 測試全數保持綠(實作前先跑一次建基準;MAINTENANCE.md 內 89/105 兩數字不一致,以實跑為準);pykinect 不可 import 的環境(CI/macOS)所有新測試照樣可跑。
5. 實機 smoke(手動):單人/雙人、遮擋交錯、離場回場、mirror、view 切換、拔線重插、暗房(關燈驗 IR)。

## 8. 非目標

- master §3a 修復(獨立處理)
- 深度式 re-ID 強化(先沿用 registry 幾何;3D mm 位置關聯留待實測後評估)
- skeleton OSC 軸向轉換(K4A y-down/z-forward 原樣輸出;先在 HANDOFF 註記,Mark 視覺端覺得反了再加 flip)
- Orbbec Femto Bolt wrapper、mkv 錄放、Kinect IMU/麥克風

## 9. 風險與備註

- pykinect_azure 的 `sys.exit` 錯誤模型是最大地雷 — guard 必須覆蓋 create/enqueue/pop/close 全路徑,漏一個就是拔線殺 server。
- K4ABT body_id 短期穩定(遮擋/離場會換號)→ stable id 一律由 registry 決定,任何地方都不得直接把 body_id 當 stable id 輸出。
- 深度範圍 NFOV 0.5–3.9m(unbinned)— 進劇場前量舞台深度;不夠再評估 binned(~5.5m,解析度降)。
- 換機部署(單 GPU 桌機)記得 `FIELD_KINECT_GPU=0`。

## 10. 驗收標準

1. UI backend 下拉出現「Azure Kinect (3D)」,選了之後 stream 跑起來 ≥25 fps(單人)
2. 兩人同框 → `/field/1/*`、`/field/2/*` 分流輸出;遮擋回復後 id 不換號
3. mirror、color/depth 切換行為正確,overlay 與人對位
4. 拔線 → UI 顯示錯誤、重插自動恢復,server 不死
5. 全部 unittest 綠(新增 + 既有),macOS 上跑測試不需要 pykinect
6. 關燈(暗場)下骨架照樣輸出

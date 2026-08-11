# Session notes — 2026-08-10(Group Box:群體邊界框 → OSC)

> 接續 `SESSION_NOTES_20260805.md`。分支仍是 `fix/mirror-default-off`,**本次的改動全部未 commit**(見 §6)。
> **§7 是還沒做完的事**,下次接手從那裡開始。
> ⚠️ 本次所有東西**都只在合成資料上驗過,沒有人真的站進去測過**。

## 0. TL;DR

- 新功能 **Group Box**:一個框把場上所有人框起來,送 OSC。**不需要 Stable ID**。
- 主力路徑是 **MediaPipe 抓骨架 + YOLO 抓群體**,兩者獨立。
- OSC 預設 target 改成 **10.0.0.102 + 10.0.0.103**(開機就指向兩台)。
- 裝了 `ultralytics` + `lap`,torch 換成 **CUDA 版**(YOLO 57 ms → **11.9 ms**)。
- Output values 面板多一個 **Group** 分頁可以看即時值。

## 1. 環境變更(**§1 of 20260805 已過時,以這裡為準**)

| 套件 | 版本 | 備註 |
|---|---|---|
| torch | **2.13.0+cu130** | CUDA 13.0,`cuda available: True`,RTX 3080 Laptop |
| torchvision | 0.28.0+cu130 | |
| ultralytics | 8.4.117 | YOLO 人物偵測 |
| lap | 0.5.13 | ByteTrack 需要 |
| matplotlib | **3.9.4(沒動)** | 仍是 SAC 放行的釘住版本 |
| mediapipe | 0.10.35 | 裝完仍正常 import |

### ⚠️ 裝 ultralytics 時一定要釘住 matplotlib

`ultralytics` 依賴 `matplotlib>=3.3.0`,直接 `pip install ultralytics` 會把 matplotlib 升到 3.11.x
→ Smart App Control 擋掉 `_c_internal_utils.pyd` → **`import mediapipe` 直接死**(就是 20260805 §1 那個坑)。

```powershell
.\.venv\Scripts\python.exe -m pip install ultralytics lap "matplotlib==3.9.4"
```

### CUDA torch 的裝法

PyPI 預設索引給的是 **CPU 版** torch。要 GPU 得指定 index:

```powershell
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

- **`cu130` 才有 torch 2.13.0**(cu128 最高只到 2.11.0)。挑同版本號才不會動到相依樹。
- `--no-deps` 是刻意的:Windows 的 CUDA wheel 自帶 DLL,不需要額外的 `nvidia-*` 套件,這樣相依樹一個都不動。
- SAC **沒有**擋 CUDA 的 DLL。
- 裝完 pip 會留下 `~orch` / `~orchvision` 殘資料夾,要手動刪(留著偶爾造成詭異 import 問題)。

**實測(1920×1080,yolov8n)**:CPU median 57 ms → **GPU median 11.9 ms**(min 10.8 / p90 14.0)。

> 註:`FIELD_KINECT_GPU=1` 走的是 DirectML,跟 torch CUDA 是兩條路,不互相影響。但 Kinect backend + YOLO 同時開會搶同一張卡。

## 2. Group Box 是什麼 / 不是什麼

**是**:所有偵測到的人的 2D 框的**聯集**,畫在畫面上,並以 OSC 送出。
**不是**:地板實際範圍。透視會讓同樣的實體距離在近處看起來比遠處寬,舉手也會把框撐大。

Kinect backend 另外能給**真實地板尺寸**(K4ABT 是絕對 mm 座標);MediaPipe 不行(見 §4)。

### 🚨 H36M 陣列**不能**拿來算群體範圍

`keypoint_mapping.py` 的 `k4abt32_to_h36m17()` 最後一行是 `return h36m - h36m[0]`,
**每個舞者的骨盆都被移到 (0,0,0)**。拿它算群體 bbox 會**永遠得到 0,而且不報錯**。

正確來源是 `runtime.last_bodies` 的**原始 K4ABT joints**。
已由 `test_group_extent.py::TestRootCenteringTrap` 釘住。

## 3. OSC 位址(⚠️ 對外新增,要通知 Nick / Mark)

### 兩種 backend 都送

```
/field/group/box_x1  box_y1  box_x2  box_y2      畫面 0–1 比例,左上為 (0,0)
/field/group/box_w   box_h   box_cx  box_cy
/field/group/count                               這一幀實際偵測到幾個人
/field/group/held                                1 = 撐住的舊值,不是實測
```

用 0–1 而不是 pixel:切 performance preset 換解析度時收端不用重對應。

### 只有 Kinect backend 送(公尺)

```
/field/group/width  depth  area  diagonal  aspect  cx  cz
/field/group/units                               套用的倍率(m=1,cm=100)
```

**MediaPipe 刻意不送這幾個**——沒有深度,送出去等於拿畫面比例冒充公尺,收端不會發現。

### 給收端的兩條規則

1. **先讀 `count`**;`count=0` 時 box 是殘值。
2. **`held=1` 當 gate**。舞者被漏掉時框會縮,看起來像「大家突然靠在一起」,數字完全合理但是錯的。

### 單位

UI 的 **Group units**(Metres / Centimetres,只在 Kinect 顯示)。切 cm 時:
長度 ×100、**`area` ×10000**(平方)、`aspect` 不變(比值)。
`/field/group/units` 一律送出實際倍率,收端不必靠數字大小猜單位。

## 4. 為什麼 MediaPipe 沒有公尺

`pose_sources.py` 走 `result.pose_world_landmarks[0]`,那是 **以各人髖部為原點的相對座標**。
兩個人的世界座標在 MediaPipe 下**不存在**,不是調參數能解的。

要在 MediaPipe 下拿到真實地板尺寸,唯一實際的路是**地板 homography**:相機固定 + 地板平坦,
量 4 個已知地板點算一次 3×3 轉換,再把每個人的腳點(bbox 底部中心)映射回地板公尺。**尚未實作**。

## 5. Group Box 與 Stable ID 是獨立的

| Stable ID | Group Box | 結果 |
|---|---|---|
| 關 | 開 | 一個框 + count,**沒有任何身分** |
| 開 | 關 | 原本的每人 ID + 各自 metric |
| 開 | 開 | 兩者都有 |

Stable ID 關閉時走 `pose_sources.py::_detect_group_only()`:直接拿 YOLO 原始偵測框做聯集,
**不碰 registry、不配 slot、不做 re-identification**。也比較省——不會對每個人各跑一次 MediaPipe 裁切。

### 掉幀保護 `GroupExtentTracker`

人數比剛才少時**撐住上一個值並標 `held=1`**;持續超過 `hold_seconds`(預設 1.0 s ≈ 30 分析幀)才承認有人真的下場。
**刻意不用 12 秒**(Stable ID 的 re-identify 值):群體框凍結十二秒比噪音更糟,數字看起來還在動但人已經走了。

## 6. 本次改動(**全部未 commit**)

新檔:

| 檔案 | 內容 |
|---|---|
| `backend/group_extent.py` | 地板位置 / bbox 量測 / 單位縮放 / 掉幀保護 |
| `backend/group_overlay.py` | 畫框 + 畫面座標正規化 |
| `backend/tests/test_group_extent.py` | 含 root-centering 陷阱、area 塌陷、單位縮放 |
| `backend/tests/test_group_overlay.py` | |
| `backend/tests/test_group_without_stable_id.py` | |
| `backend/tests/test_pose_source_construction.py` | **見 §6.1** |
| `backend/tests/test_osc_defaults.py` | |

改動:`osc_viewer.py`、`pose_sources.py`、`pose_engine.py`、`pose_backends/azure_kinect.py`、`tests/test_pose_sources_tracking.py`

**243 tests OK。**

### 6.1 本次踩到的兩個 bug(都是「接線接錯地方」)

兩個症狀一模一樣:**畫面正常、沒有任何錯誤訊息,但 analysis 一次都沒跑**(`frame_count` 一直加、`analysis_count` 卡在 0)。

1. **初始化順序**:`MediaPipePoseSource.__init__` 的 YOLO preload 判斷讀 `self.group_extent_enabled`,但那個屬性在 15 行之後才指派 → `__init__` 丟 `AttributeError` → engine 建不起來。
2. **放進了跑不到的分支**:`_detect_group_only()` 一開始加在 `_select_tracking_crop()` 裡,但 `estimate()` 在 Stable ID 關閉時**直接走單人路 return**,根本到不了 `_select_tracking_crop`。

`test_pose_source_construction.py` 專門守這兩件事:它跑**真正的 `__init__` 和 `estimate()`**(只 mock 掉模型載入)。
其他 group 測試用 `__new__` 建假物件,**結構上抓不到這類接線錯誤**。兩個 bug 都驗證過:把錯誤放回去,測試會紅。

> 相關:`tests/test_pose_sources_tracking.py` 的 `_bare_source()` 補上了 group 屬性。
> **刻意不在 `_detect_group_only` 用 `getattr` 兜底**——兜底會把上面第 1 種 bug 永遠藏起來。

## 7. 未完成 / 下次接手要做的

### 7a. 最高優先:**上機實測**

本次**沒有任何一項在真人身上驗過**。要確認:

- [ ] 框有沒有正確跟著人(MediaPipe + YOLO,Stable ID 關)
- [ ] `count` 準不準、`held` 是否在掉幀時如預期亮起
- [ ] Group 分頁的值有沒有在跳
- [ ] Kinect backend 下的公尺數字對不對(拿捲尺量一次)

### 7b. 深度覆蓋(**沿用 20260805 §4a,仍未解**)

單人在排練位置 4.85 m 追蹤率只有 **27/90**。群體範圍**對這個問題特別敏感**——
它要求所有人同時被追到,少一個框就會縮。`nfov_binned` 已設但**仍未在有人站位時驗證**。

### 7c. 其他

- [ ] **Kinect 校準 preset 要重錄**(仍是 20260805 就記著的問題,`Rebecca_Clibrate_001_0630` 是 MediaPipe 錄的)。
- [ ] `bytetrack.yaml` 仍是 stock 預設。舞蹈場景可拉長 `track_buffer`(90)對短暫遮擋較好;不用 Stable ID 時影響不大。
- [ ] `fix/mirror-default-off` **仍未 push**,本次改動**也未 commit**。
- [ ] MediaPipe 地板 homography(§4)——想在 MediaPipe 下拿真實公尺就得做。
- [ ] `list_cameras()` 在 Windows 不顯示相機名稱(20260805 §4b)。

## 8. ⚠️ 操作陷阱

**`POST /api/apply` 是整份表單覆蓋。** 沒帶到的欄位一律回 `Form(...)` 預設值。
本次用 curl 帶半套去測,結果把 `osc_mode` 打回 `raw`,連帶觸發 `clear_metric_ranges()`,
**把套用中的 `Rebecca_Clibrate_001_0630` 校正清掉了**——而且全程沒有任何錯誤訊息。

要看狀態請讀 **`GET /api/state`**(注意:`/api/status` 不存在,回 404)。要改設定請走 UI。

**`source_state` 只活在記憶體**,每次重啟都回預設——OSC target 就是因為這樣才做成程式預設(§3)。

## 9. 重啟方式(不變)

```powershell
$env:FIELD_KINECT_DEPTH_MODE = "nfov_binned"
$env:K4A_ENABLE_LOG_TO_STDOUT = "0"
$env:K4ABT_ENABLE_LOG_TO_STDOUT = "0"
.\.venv\Scripts\python.exe backend\osc_viewer.py
```

停止一律 `POST /api/camera/release` → `POST /api/shutdown`,**不要硬砍**(Kinect 開著時會把 k4a 裝置卡死)。

# Session notes — 2026-08-05(anarc 的 Windows 機器,首次安裝 + Kinect 上機)

> ⚠️ **後續進度見 `SESSION_NOTES_20260810.md`。**
> 其中 **§1 的環境表已過時**(之後裝了 ultralytics + CUDA torch),
> §5 的 OSC 契約**新增了 `/field/group/*` 一組位址**。
> §4a 的深度覆蓋問題**仍未解**。

> 這台是**新機器**,從零安裝。分支 `fix/mirror-default-off`(3 個 commit,**尚未 push**)。
> 環境細節見 §1;**§4 是還沒做完的事**,下次接手從那裡開始。

## 0. TL;DR

- 全新 Windows 機器裝好了 viewer + Azure Kinect backend,`check_kinect_env.py` 通過(26 fps,`devices connected: 1`)。
- 修掉一個**會凍住整台 server 的 Kinect 死鎖**(§3),已 commit。
- **最大的未解問題:排練站位 4.85 m 超出 NFOV unbinned 的 3.9 m,追蹤率只有 27/90。** `nfov_binned` 是候選解但**尚未實測驗證**。

## 1. 這台機器的環境(與 INSTALL_WINDOWS.md 的差異)

| 項目 | 這台的實際狀況 |
|---|---|
| Python | **3.11.9**(INSTALL_WINDOWS.md 建議 3.10;3.11 實測全綠) |
| venv | `.venv\`(repo 內,已 gitignore)。所有指令用 `.\.venv\Scripts\python.exe` |
| GPU | AMD 內顯 + **RTX 3080 Laptop**(雙顯卡)→ `FIELD_KINECT_GPU` 用**預設 1 即正確**,實測 26 fps(指到內顯會是 ~6 fps) |
| SDK | Sensor 1.4.2 + Body Tracking 1.1.2 已裝,安裝檔留在 `~\Downloads\kinect-sdk\` |

### ⚠️ Smart App Control 會擋 matplotlib(這台特有,最容易重踩)

`VerifiedAndReputablePolicyState = 1`(強制模式)。**matplotlib 3.11.1 的 `_c_internal_utils.pyd` 被擋**,而 mediapipe 在 import 時就會載入 matplotlib → `import mediapipe` 直接失敗。

- 解法:**`matplotlib==3.9.4`**(舊版有信譽紀錄,SAC 放行)+ **`mediapipe==0.10.35`**。
- `requirements.txt` **沒有 pin 版本**,所以任何人重跑 `pip install -r requirements.txt` 都會再抓到 mediapipe 1.0.0 + matplotlib 3.11.x **再壞一次**。
- **不要用關閉 Smart App Control 來解**——它一旦關掉,除非重灌 Windows 否則無法再開啟。

### 啟動指令(env 目前只在啟動的 shell 內,**尚未設成永久**)

```powershell
$env:FIELD_KINECT_DEPTH_MODE = "nfov_binned"
$env:K4A_ENABLE_LOG_TO_STDOUT = "0"    # 不設的話 k4a 每秒噴 ~30 行,43 分鐘灌了 500 KB
$env:K4ABT_ENABLE_LOG_TO_STDOUT = "0"
.\.venv\Scripts\python.exe backend\osc_viewer.py
```

## 2. 相機索引(這台)

DirectShow 列舉順序 = OpenCV 索引(實測相符):

```
[0..3] NDI Webcam Video 1-4   全黑 (mean 0.0)
[4]    Azure Kinect 4K Camera 唯一有畫面的
[5]    OBS Virtual Camera     開不起來
```

MediaPipe backend 要選 **Camera 4**。UI 只顯示 `Camera N` 不顯示名稱——`osc_viewer.py` 的 `list_cameras()` 只查 `mac_names`,Windows 的 `get_directshow_camera_names()` 結果**從未被使用**(pygrabber 等於白裝)。未修,見 §4。

## 3. 本次修掉的 Kinect 死鎖(commit `a791c4f`)

**症狀**:Enter 跑 Kinect 後,stream 結束時整台 server 凍住,連 `/` 都逾時 60s;log 被 `capturesync_drop ... type:Color` 洗版(3071 筆)。**`/api/shutdown` 也回不來,只能硬砍。**

**根因**(py-spy stack dump 實證,非推測):

```
MainThread : release (azure_kinect.py:487)   <- 等 _lock
             stream_live (osc_viewer.py:938) <- finally: frame_source.release()
worker     : k4a_device_get_capture          <- 持有 _lock,無限等待
```

1. `read()` 全程持有 `self._lock`,內部 `self._device.update()` 用 pykinect 預設 **`K4A_WAIT_INFINITE`** → 相機一停,鎖永不釋放。
2. `osc_viewer.py:938` 的 `frame_source.release()` 是**裸同步呼叫,跑在 event loop 上**(同迴圈的 read/reopen/encode 全都有 `await asyncio.to_thread(...)`,只有這行漏)。
3. → event loop 死鎖,全站癱瘓。

**修法**:`CAPTURE_TIMEOUT_MS = 1000` 傳給 `device.update()`。逾時走既有的 `missed_frames >= 5 → reopen` 復原路徑。

**刻意沒動 `osc_viewer.py:938`**:改成 `await asyncio.to_thread(...)` 會踩到 async generator 的 `finally` 在 `aclose()` 期間 await 導致 `RuntimeError: async generator ignored GeneratorExit`,反而製造新的斷線 bug。加了 timeout 後 release 最多等 ~1.35s(capture 1000 + pop 350),是有界停頓。**要根治那行需另外處理 generator 生命週期,是獨立的一題。**

> 硬砍後 `check_kinect_env.py` 仍回 `devices connected: 1`,這次沒卡住——但那是運氣。正常關法仍是 `POST /api/camera/release` → `POST /api/shutdown`。

## 4. 未完成 / 下次接手要做的

### 4a. 最高優先:深度覆蓋範圍(**會決定排練能不能用**)

實測(pelvis,NFOV unbinned,舞者在排練位置):

| 量測 | 值 | 判定 |
|---|---|---|
| 徑向距離 | **4.85 m** | 上限 3.9 m → **超出 0.98 m** |
| 水平角 | 29.5° / 37.5° | 只剩 6.9°,**TIGHT** |
| **追蹤成功率** | **27 / 90 frames** | **掉 70%** |

- **距離是徑向的,轉鏡頭不會改善**;只能換 `nfov_binned`(0.5–5.5 m)或把 Kinect 往前移約 1 m。
- 切 `nfov_binned` 後深度有效像素從 36.1% → **53.8%**,但**當時沒有人站在位置上,追蹤率尚未驗證**。
- 水平偏軸 2.36 m,重新對準可把餘裕從 6.9° 拉到接近 37°(多人散開時特別重要)。
- 量測腳本:`(scratchpad)/kinect_coverage.py`,吃 `FIELD_KINECT_DEPTH_MODE`。**下次請人站到排練位置重跑。**

### 4b. 其他待辦

- [ ] `FIELD_KINECT_DEPTH_MODE` / `K4A_ENABLE_LOG_TO_STDOUT` 要不要設成永久使用者環境變數(目前每次啟動要手動設)。
- [ ] **Kinect 要重錄校準 preset**。開機自動套用的 `Rebecca_Clibrate_001_0630` 是 **MediaPipe 錄的**,在 Kinect 下 expansion/height/sway/jerk/torque 值域會失真。
- [ ] `fix/mirror-default-off` **尚未 push**。帳號 `anarchydancetheatre-ui` 對此 repo 有 `push=True`(非 admin)。死鎖修正值得單獨開 PR 回報上游(py-spy dump 是很強的證據)。
- [ ] mirror 預設關閉**尚未在 Kinect backend 實機驗證**(`azure_kinect.py:230` 讀 `runtime.mirrored` 決定是否跑 `mirror_h36m17()`;邏輯上應無事,但沒跑過)。
- [ ] 面板順序調整後**尚未實機看過**。
- [ ] `list_cameras()` 在 Windows 不顯示相機名稱(§2)。修的話要注意 `osc_viewer.py:477` 的註解:DirectShow 名稱順序**不保證**對得上 OpenCV 索引(這台實測相符,別台未必)。

## 5. ⚠️ OSC 契約已是 v2(對外承諾,務必先通知收端)

master 現在送 **`/field/<id>/<metric>`**(`osc_sender.py:458`),例 `/field/1/energy`。**舊的 `/field/energy` 已不存在。**

- **上場前必須通知 Nick(Max/MSP)與 Mark(TouchDesigner)改 patch**,否則收端會**靜悄悄收不到東西**(不報錯)。
- id 是**槽位不是人**:取最小空號,兩個舞者永遠是 1 和 2;離場超過 `reidentify_seconds`(12s)號碼會釋出,新進場的人可能拿到。
- Stable ID 關閉時單人固定 `id=1`。

## 6. 本次 commit(分支 `fix/mirror-default-off`,未 push)

| commit | 內容 |
|---|---|
| `f3101bd` | Mirror camera 預設改為不勾(`source_state` + HTML `checked` 兩處必須同時改)+ README |
| `a791c4f` | Kinect capture 加 1000ms timeout,修死鎖(§3) |
| `44f3b93` | Calibration 面板移到指標卡下方;`.calibration-panel` 改 `border-top` |

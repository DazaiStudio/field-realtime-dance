# MAINTENANCE.md — FIELD Realtime Dance Viewer 維護交接

> 寫給接手的工程師 / AI 助手(Codex、Claude、其他皆可)。上一手:Claude(Fable 5),2026-07-07。
> 功能與安裝說明在 `HANDOFF.md`(其中部分敘述已過時,見 §4)。本文件是**維護視角**:目前狀態、未修問題、刻意決策、地雷區。

## 0. 接手第一步(TL;DR)

1. 讀完本文件,再讀 `HANDOFF.md`。
2. 建立測試基準:`cd backend && python -m unittest discover -s tests`
   - `master`:69 tests green(1 skip);`research/stable-id-tracking`:89 tests green(1 skip)。
   - 若不是全綠,先查環境再查 code(見 §2 注意事項)。
3. 改任何東西前先看 §4「刻意的設計決策」— 那些不是 bug,不要順手修掉。
4. 要動 `backend/osc_viewer.py` 先讀 §5「地雷區」。

## 1. Repo / 分支狀態(2026-07-07)

| Branch | 狀態 | 說明 |
|---|---|---|
| `master` | **production(排練用)** | 單人版。feat/rtmpose-backend 已合入 + 6/27–7/1 排練期 tuning。本次已修 metrics reset race(`ca894f5`) |
| `research/stable-id-tracking` | 開發中,**待實機驗證** | 多人 stable-ID(YOLO yolov8n + `person_tracker.py` registry)。2026-07-08:4 個 blocker 已修 + 改為 **per-person OSC 輸出 `/field/<id>/<metric>`(新 wire contract,見 §8)**。剩餘工作見 §3b;實機驗證後可合併 |
| `feat/rtmpose-backend` | 已合入 master | 封存 |
| `feat/multi-person` | **已淘汰** | 2026-06 舊多人架構(SlotBinder / per-slot OSC / CentroidTracker),被 research/stable-id-tracking 取代。留作參考,勿繼續開發,勿把兩套 tracker 混用 |

## 2. 環境與跑法

- Windows 主機(RTX 4080):全域 Python 3.10(`C:\Users\tommy\AppData\Local\Programs\Python\Python310\python.exe`)。
  啟動:`python backend/osc_viewer.py` → http://127.0.0.1:9100。
- macOS(Nick 聲音 / Mark 視覺):MediaPipe only,安裝流程見 `HANDOFF.md`。
- 測試:`cd backend && python -m unittest discover -s tests`。注意:
  - calibration flow 測試會對 **127.0.0.1:9000 發真的 UDP OSC**(`FIELD_OSC_ENABLED` 預設 1)— 該 port 有聲音軟體在聽時會收到假資料。
  - 測試殘留 UDP socket 的 ResourceWarning:已知、無害。
  - mediapipe 需 ≥0.10.35(protobuf 6 相容);0.10.20 會在建 PoseLandmarker 時 crash。
- 外部消費者:Nick(Max/MSP)、Mark(TouchDesigner)接 `/field/*` OSC。**wire contract(address、值域)是對外承諾**。

## 3. 未修問題清單(2026-07-07 全面 code review 殘留)

背景:當日做了 9 角度 × 2 diff 的 max-effort review,45 項去重候選、37 項 confirmed。已修:master reset race、branch 換人 reset + 遲滯、pythonosc 測試清理。其餘按優先序:

### 3a. master(影響現役系統)

| 優先 | 問題 | 位置 |
|---|---|---|
| 高 | 第二個 `/stream` 連線 bump session → 第一個(投影)靜默凍在最後一幀,無任何提示 | `osc_viewer.py` `stream()`(~1288);至少加 log/UI 警示 |
| 高 | profile 名稱打 `presets` 或 `ranges` → 下次啟動使用者 preset 檔被誤判格式**整檔清空**(內建 preset 倖存) | `calibration.py` `normalize_presets`(~26);加名稱驗證 |
| 中 | JS 把 `{status:'error'}` 回應整包餵 `update()` → UI 重設為預設(OSC target 顯示 localhost);ws 斷線無 reconnect,假狀態可被下一次 Enter 真的送出 | `osc_viewer.py` JS stopCalibration / applyNormalizeProfile(~3088);update() 各區塊加 payload guard |
| 中 | 每個 OSC 欄位 keystroke(debounce 300ms)都把 stale metrics 重推一次輸出 EMA + adaptive decay → 值跳動 | `refresh_prepared_metrics`(~309):`send_metrics(send_keys=set())` 仍會跑 `_prepare_value` |
| 中 | `get_pose_engine` 無鎖:快速 off/on 可雙建 engine(native leak / close-in-use) | `osc_viewer.py` ~143、~674 |
| 中 | 相機 reopen 搶奪 race:舊 loop 斷線重連可搶走新 stream 的相機(1–2 秒黑畫面,自癒) | `open_camera` / `reopen_live_camera`(~220) |
| 低 | `osc_namespace` 表單預設 `""` 且空值不再 fallback `/field` → 非 UI client 沒帶欄位就丟前綴 | `apply_input` / `apply_osc_config` |
| 低 | 校準錄製中套 profile 沒有 active guard → 0..1 樣本混進 collector,存出壞 preset | `calibration_apply`(~1160) |
| 低 | Stop 按鈕被 15Hz sync 重新 enable → 可雙擊重入 | JS `syncCalibration` |
| 低 | 2026-07-01 前存的使用者 preset,height 是舊語意 → fixed mode 的 `/field/height` 釘在 1.0(內建 Rebecca preset 是新語意,沒事);重新校準即解 | height 語意變更 `69b2616` |

### 3b. research/stable-id-tracking(剩餘工作)

**4 個 blocker 已於 2026-07-08 修復**(commits `c6d5024`、`f061fab`,105 測試綠):YOLO 背景預載+失敗 latch(不再於 frame loop 下載/重試)、holding track 不再跑 pose(不會鎖錯人)、單人 overlay guard 恢復(不閃爍)、tracker 錯誤清 ghost 框、`send_skeleton` 先全驗證再送(不再送半具骨架)、dead state 清理;並改為 **per-person metrics/OSC**(每個 id 一台 DanceMetricsEngine,「換人尖峰」問題從根本解除)。

| 優先 | 問題 | 位置 |
|---|---|---|
| 高 | **實機驗證**:多人、交錯遮擋、M1 效能。per-person 輸出後每個 tracking 中的人都要各跑一次 MediaPipe(這是功能成本,不再是浪費)——M1 上 3–4 人的 fps 要實測,必要時降 analysis_fps 或限制人數上限 | — |
| 高 | tracked 路徑用 IMAGE mode(stateless),校準 preset 是 VIDEO mode 錄的 → jitter 可能讓 jerk/torque 貼 1.0;實機比對,必要時 per-mode 校準 | `_detect_frame_pose(stateless=True)` |
| 中 | `suppress_duplicate_person_tracks`:被刪的 box 仍繼續壓別人 → 真舞者可能整幀沒 track;加 `if not keep[i]: break` | `person_tracker.py` ~58 |
| 中 | registry dicts 跨執行緒無鎖,且 `registry.update()` 在 try/except 外 → 罕見 KeyError 會殺 stream | `pose_sources.py` |
| 低 | `syncTrackingTargets` 15Hz 蓋寫 UI:面板被重新藏起、下拉重建(選單收合)、選擇被改回 auto_largest → 照 `normalizeProfileSelect` 加 activeElement/dirty guard | `osc_viewer.py` JS ~3604 |
| 低 | re-ID 對 stale bbox 仍可能誤鎖(IoU 0.12 門檻寬);pose 已不會跑在 stale crop 上,只剩 re-id 指派這一條路徑 | `person_tracker.py` `_best_reidentify_match` |
| 清理 | `StableTrackSelector` + `expand_bbox` 是死碼(~130 行,只有測試在用,re-id 閾值已 drift 1.35 vs 0.85)→ 刪 | `person_tracker.py` |

### 3c. 兩邊皆有(品質 / 重複)

- 17 關節表 `skeletonJoints`(JS)手抄 `SKELETON_ADDRESS_NAMES`(Python)→ 改用既有 `%PLACEHOLDER%` 注入;metric label 表也有兩份(viewer vs `/charts`)且已 drift(Height vs CoM height)。
- reset 序列在 7 個觸發點各自手抄不同子集(calibration start/stop/apply/clear/import、`/api/apply`、`/api/osc/config`)→ 集中成一個函式,新增狀態才不會漏。
- `osc_sender._normalize` 有 7–8 個 metric-name 分支(BOUNDED/ADAPTIVE/UNBOUNDED/LOG/GAMMA/sync_corr),且 log/gamma **只在 fixed mode 生效**(normalize mode 曲線不同)→ 再加 response 調整前先重整成 per-metric spec 表。
- 15Hz DOM churn 未修完:`renderOscTargets`、addresses 列表、隱藏中的 skeleton 面板每 tick 重建;calibration ranges 塞進每個 ws push 但 JS 從不讀。
- 預設解析度 1080p(`DEFAULT_PERFORMANCE="quality"`)且 Quality 下拉已移除 → 無任何降級路徑(M1 上吃 CPU,注意)。

## 4. 刻意的設計決策(不要當 bug「順手修掉」)

- **RTMPose3D 藏起來**是排練決策(`rtmpose3d_selectable()` 硬回 `False`);且 `state_payload()` 會把 env 選的 backend 改寫回 mediapipe → **實際完全不可達**(`HANDOFF.md` 說 "code remains available" 已不準)。要復活需一併處理:quality gate(RTMPose3D 硬編 quality=1.0,校準 gate 會失效)、`configure_tracking` 對它是 no-op。
- **One-Euro 骨架平滑永久關閉**(舊版預設開)→ 改用 per-metric 輸出 EMA。`/api/smoothing` 是相容性殭屍端點。
- **`sync_correlation` 在 normalize/fixed 是雙極 -1..1**(舊契約 0..1、0.5 中性)。改回去前**先跟 Nick/Mark 對齊**。
- **試音流程:stop 只存 preset、不自動套用**(舊版自動套 fixed)→ 校準完必須手動選 profile,否則 OSC 停在 raw。7/1 改的,是否刻意待跟 Tommy 確認;至少 UI 應警示 raw 狀態。
- **開機自動套 `Rebecca_Clibrate_001_0630` + fixed mode**(env `FIELD_DEFAULT_CALIBRATION_PRESET` 可改名/設空字串停用)。會蓋掉 `--osc-mode`;crash 重啟回到這個 preset 而非當日 profile(最後套用的 preset 沒有落地)。
- **`/stream` 單消費者**(每個連線 bump session):防止兩個 generator 搶同一台相機;副作用見 §3a 第 1 項。

## 5. 地雷區(改 code 前必讀)

1. **`backend/osc_viewer.py` 是 ~4000 行巨檔**:FastAPI app + `VIEWER_HTML`(inline HTML/JS)+ `CHARTS_HTML`。改 Python 端點要同步看對應的 JS fetch;JS 其實是 Python 字串,常數用 `%PLACEHOLDER%` + `.replace()` 注入 — **新常數走這條,不要手抄第二份**。
2. **執行緒模型**:stream loop 用 `asyncio.to_thread(engine.process_frame, ...)`;FastAPI handler 在 event loop。跨兩邊的共享狀態要想並發 — `DanceMetricsEngine` 已有 RLock 模式可參考。stream loop 的 await **沒有 try/except**,worker 丟例外 = stream 死。
3. **換人偵測(branch)**:`PoseEngine._metrics_person_id` 會在 `last_tracking.active_id` 變化時自動 reset metrics + smoother — 新增 selection 模式**不需要**在端點手動 reset。auto_largest 遲滯 = `auto_switch_area_ratio`(預設 1.3)。
4. **15Hz ws `update()` 會蓋 UI**:新增任何輸入控件必須加 dirty/focus/signature guard(參考 `syncCalibration` / `syncPoseBackends`),否則使用者輸入 66ms 內被蓋掉。
5. **metric 改名/新增要動很多處**:`METRIC_NAMES`、`OSC_ADDRESS_NAMES`、`osc_sender` 各 set/dict、`calibration.RANGE_METRICS`、viewer JS `metricLabels`、charts `METRICS`。先全域搜尋再動手。
6. **校準檔**:`calibration_presets.json`(使用者)+ `project_default_calibration_presets.json`(內建,啟動時 merge)。格式沒有版本欄位;height 語意 7/1 改過,舊檔不相容。
7. **演出季內不要動 wire contract**(OSC address、值域、`/api/*` schema)— Nick/Mark 的 patch 直接依賴;要動先通知。目前契約 = §8 的 v2(per-person)。

## 6. 已完成工作紀錄

**2026-07-07**
- review:9 finder 角度 × 2 diff → 45 項驗證(37 CONFIRMED / 3 PLAUSIBLE / 1 REFUTED)+ sweep 6 項;重點已全數抄錄於 §3/§4。
- `master ca894f5`:`DanceMetricsEngine` 加 RLock 修 reset race(校準/apply 端點 vs stream worker),附 race regression test。
- `research/stable-id-tracking 719cb65`:`PoseEngine` 換 active 舞者時 reset metrics/smoother + `choose_active` auto_largest 遲滯,附 tests;`f9767fb`:pythonosc 無 `close()` 的測試清理修正(master 亦同步修)。

**2026-07-08(branch)**
- `c6d5024`:4 個 blocker 修復(YOLO 背景預載+失敗 latch、holding 不跑 pose、單人 overlay guard、ghost 框清理、dead state 移除)。
- `f061fab`:**per-person OSC 輸出**(§8 新契約)— 每個 stable id 一台 DanceMetricsEngine + per-id smoothing/normalize 狀態 + per-id skeleton;send_skeleton 改為先全驗證再送。
- 測試基準:master 69 綠 / branch **105 綠**(各 1 skip)。

## 7. 建議路線圖

1. **master**:修 §3a 高優先 3 項(都是小改動)。「試音 stop 不自動套用」「開機強制 preset」先跟 Tommy 確認是否刻意,再決定修或加警示。
2. **branch**:§3b 剩餘項 → 實機相機驗證(多人、交錯遮擋、M1 效能)→ **通知 Nick/Mark 新 OSC 契約(§8)** → 合併 master。
3. 清理:刪 `StableTrackSelector` 死碼、skeletonJoints 改用 %PLACEHOLDER% 注入。
4. **ai-motion repo**(另一個 repo):culture map 需因 H36M 對應修正重新匯出,morrisness 才會準。
5. **Azure Kinect(演出後升級,Tommy 已表態有興趣)**:
   - **狀態:已實作於 `feat/kinect-pose-source`(2026-07-30),實機 smoke 通過(30fps、多視圖、mirror、OSC);用法/env 見 `HANDOFF.md` "Azure Kinect backend"。設計 spec 與實作計畫在 `docs/superpowers/`。**
   - 為什麼:ToF/IR 深度**不怕暗場與投影光**(劇場最大痛點);真 3D 公厘座標(dance_metrics 的原生單位,比 MediaPipe 推估深度準);原生多人 body tracking(32 關節)可取代 YOLO 偵測層,深度分離比 RGB bbox 可靠。
   - 接法:`PoseEngine` 的 pose source 介面就是現成接縫 — 寫一個 `AzureKinectPoseSource`(K4A body tracking → K4ABT-32 → H36M-17 對應表),沿用 `MultiPersonTrackRegistry` 在 body id 上做 re-ID,per-person 管線(§8)原樣可用。
   - 主要工程:**取像路徑** — 目前 camera loop 綁死 `cv2.VideoCapture`,需加一層 frame-source 抽象讓 Kinect 自帶取像(這是最大塊的改動,pose source 本身反而簡單)。
   - 限制:Body Tracking SDK 只有 Windows/Linux + GPU,**Mac 不能跑**(Nick/Mark 只收 OSC,不受影響);Azure Kinect DK 已停產 → 新購買 **Orbbec Femto Bolt**(官方合作、K4A 相容 SDK);NFOV 深度範圍約 0.5–3.9m(binned ~5.5m),動工前先量舞台深度。
   - Python 端:`pyk4a` 或 `pykinect_azure`(sensor + body tracking 包裝)。

## 8. OSC wire contract v2(2026-07-08 起,branch;合併後即為正式契約)

**所有輸出一律帶舞者 id,無 active/非 active 之分**(Tommy 2026-07-08 拍板;舊單人通道已移除):

- `/field/<id>/<metric>` — 例 `/field/1/energy 0.42`、`/field/2/jerk 0.13`;metric 縮寫同舊制(`sync_vel`、`sync_corr`)。
- `/field/<id>/sk/<joint>` — 每關節 `[x, y, z]`(Skeleton 輸出開啟時,每個追蹤中的舞者都送)。
- `/field/morrisness` — 維持全域(顯示中舞者的 CultureScore)。
- id = stable id(1、2、3…,依入場順序;短暫遮擋後 re-id 沿用同號)。**Stable ID 關閉時單人固定 id=1**,接收端只需一套 schema。
- **id 會回收(2026-08-04 改)**:新 track 取**最小空號**,而不是一路遞增。舊行為會讓一整場排練的 id 爬到幾十,接收端寫死 `/field/1/*` 就靜悄悄收不到東西。現在**線上的 id 集合由「同時在場人數」決定**,兩個舞者永遠是 1 和 2。
  - 代價:**id 代表槽位,不代表人**。舞者離場超過 `reidentify_seconds`(12s)後 track 被丟棄、號碼釋出,新進場的人可能拿到他原本的號。12 秒內的遮擋仍沿用同號(re-id 照舊)。
  - 要「同一個人恆等於同一個 id」是另一個題目;拉長 `reidentify_seconds` 可延長身分保持,但拉太長會把不同人誤認成同一個。
- 舞者被遮擋/離場 → 該 id **靜默**(不送 0);回場後同 id 繼續。
- 校準 fixed ranges 全域套用(所有 id 同一組 range;per-dancer preset 需手動切換)。
- **⚠️ 上場前必須通知 Nick(Max/MSP)與 Mark(TouchDesigner)改 patch**:舊的 `/field/energy` 等位址已不存在。

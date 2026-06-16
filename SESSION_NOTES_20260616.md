# FIELD viewer — 工作記錄 / 重啟指南（2026-06-16）

> 個人本地筆記（local commit、不 push、不上 Notion）。重開機後照「重啟 viewer」那段做即可。

---

## 目前狀態

- **程式碼已安全**：今天的功能都已 commit + push 到 `field-realtime-dance` master，重開機不會丟。
  - `8ba0ebf` — live feed 右下角 YouTube 式**全螢幕按鈕**
  - `2283465` — **9-metric overlay**（只在全螢幕時、畫面右上角顯示）
  - `9b8814e` — overlay 字級放大（12→15px）
  - `578221e` — 右上角 **Quit 鈕**（兩段確認 → 關 server、釋放相機 → 顯示 Viewer stopped）
  - `c5c21c7` — 修 Quit 鈕被 CSS `button{width:100%}` 撐滿全寬的問題
- **環境已修好**：把全域 Python 3.10 的 `mediapipe` 從 0.10.20 升到 **0.10.35**（配合已安裝的 protobuf 6.33.5）。原本按 Detect 會 crash 的問題已解決，已驗證 pose 模型可正常建立。
- **viewer 目前是開著的**（detached，重開機前最後 PID 56640）。重開機後會關掉 → 用下面指令重開。

## ⏭️ 下次繼續：Restart / 狀態燈 / 啟動（討論到一半，未決定）

**關鍵結論（已釐清）**：web 端**無法**啟動「已經 Quit 的 server」——網頁是 server 發出來的，server 行程一死、網頁也沒了，沒有任何端點能回應「啟動」。所以「網頁上放 Restart 把 server 殺掉再復活」這條路不通。

**狀態燈**：綠/紅燈值得加，但在 viewer 頁面內，紅燈≈「頁面也連不上 server」；用處是提示「server 死了、去桌面重開」，不是能在紅燈下從頁面復活。

**三個真正可行的方向（待 Tommy 選）**：
1. **桌面捷徑 + 狀態燈（我推薦）**：.bat/捷徑雙擊就開、再雙擊＝殺舊+重開（這就是 Restart）；Quit 關閉；網頁加綠/紅連線燈。最穩、非技術人員最直覺。
2. **獨立啟動器控制台**：另做一個常駐、不會被 Quit 殺掉的小控制頁（另一個 port），畫面上 Start/Stop/Restart + 狀態燈全有。最完整但要多維護一個程式。
3. **只加軟重置 Reset**：不殺 server，只重開相機/pipeline，救卡頓/換鏡頭；真正關閉仍用 Quit。可單獨或搭配 1。

**未決關鍵問題**：這個 viewer 平常**是誰、在哪台機器上開關**（Tommy 自己 / Mark / Nick / 現場非技術人員）？答案會決定選哪個方向。回來先回答這個。

## 待我自己確認的一件事（重開機前後都可）

瀏覽器 `Ctrl+Shift+R` 重新整理 → 按 **Detect**：
- 預期：顯示骨架、狀態變 `detecting`，**不再卡 waiting、畫面不消失**。
- 按右下角全螢幕鈕 → 右上角應出現 9 個 metric 數值。

---

## 重啟 viewer（重開機後這樣做）

**方法 A — 可見視窗（推薦自己手動跑，看得到 log，Ctrl+C 可停）**

開 PowerShell：

```powershell
cd D:\Github\field-realtime-dance
& "C:\Users\tommy\AppData\Local\Programs\Python\Python310\python.exe" backend\osc_viewer.py
```

看到 `Uvicorn running on http://127.0.0.1:9100` 就是好了，開瀏覽器到 **http://127.0.0.1:9100**。
（不要關那個 PowerShell 視窗，關了 viewer 就停。）

**方法 B — 背景 detached（關掉終端機也繼續跑）**

```powershell
$py = "C:\Users\tommy\AppData\Local\Programs\Python\Python310\python.exe"
$wd = "D:\Github\field-realtime-dance"
Start-Process -FilePath $py -ArgumentList "backend\osc_viewer.py" -WorkingDirectory $wd -WindowStyle Hidden `
  -RedirectStandardOutput (Join-Path $wd "osc_viewer.stdout.log") `
  -RedirectStandardError (Join-Path $wd "osc_viewer.stderr.log")
```

或者：重開機後直接跟我（Claude）說「開 webviewer」，我幫你跑。

## 停止 viewer

- **最簡單**：網頁右上角 **Quit** 鈕 → 再按一次「Confirm quit」確認 → server 自動關閉、釋放相機，畫面顯示「Viewer stopped」。（關網頁分頁本身不會停 server，要用這顆鈕。）
- 方法 A 的可見視窗按 `Ctrl+C`；或
- 用 port 找來關：
```powershell
Get-NetTCPConnection -LocalPort 9100 -State Listen | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
```

---

## 環境重點（之後若再出問題對照用）

| 項目 | 值 |
|---|---|
| 跑 viewer 的 Python | `C:\Users\tommy\AppData\Local\Programs\Python\Python310\python.exe`（全域 3.10） |
| 入口 | `backend/osc_viewer.py` |
| 網頁 | http://127.0.0.1:9100 |
| OSC 輸出 | `udp://127.0.0.1:9000`，prefix `/field`，mode raw |
| 關鍵套件 | **mediapipe ≥ 0.10.35**（protobuf 是 6.33.5，0.10.20 不相容會 crash） |

**已知無害警告**（log 會出現，可忽略，不影響偵測）：
- `AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'` — TensorFlow × protobuf 6 既有衝突
- `tensorflow-intel ... requires protobuf <6.0.0` — 同上，viewer 沒用到 TF

**為何不要用 Bash 背景啟動**：用工具的 run_in_background 跑 server 會被背景任務樹回收（exit 127），連帶把 server 關掉；所以一律用 `Start-Process`（方法 B）或可見視窗（方法 A）。

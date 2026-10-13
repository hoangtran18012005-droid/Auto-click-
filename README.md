# Flappy Bird Auto Player — Windows 11

## Cấu hình mặc định đã kiểm thử

Khi chạy `run.bat` không tham số, ứng dụng nạp `profiles/live-20261013-latency.yaml`. Đây là profile đã dùng trong một lượt live, màn hình game hiển thị đỉnh **1.196 điểm**. Bot được dừng bằng F8 khi đang chơi; vì vậy đây là điểm quan sát được, không phải kết quả OCR sau màn hình kết thúc.

Trong log của lượt đó: thời gian chạy khoảng 1.702 giây, FPS capture trung vị 58,5 và độ trễ capture-to-input trung vị 28,7 ms. Các số này phụ thuộc máy. Profile bao gồm ROI màn hình `left=401, top=289, width=816, height=544`; nếu game ở vị trí khác, chọn lại vùng game và lưu cấu hình.

```powershell
# Chạy giao diện với profile mặc định
.\run.bat

# Khôi phục profile đối chiếu cũ
.\run.bat --config profiles/windows-click.yaml
```

Không chạy hai instance bot cùng lúc. Trong cửa sổ bot, F6 bắt đầu/tạm dừng và F8 dừng. Giữ tab game ở foreground khi bot hoạt động. Profile cũ được giữ nguyên để rollback.

Đây là bot thị giác dựa trên MSS, OpenCV, bộ lập kế hoạch reachability và click chuột. Mã Double DQN trong `rl/` là thành phần nghiên cứu offline, chưa được dùng trong lượt 1.196 điểm.

## 1. Cài đặt

Dùng Windows 11 và Python 3.12 64-bit có Python Launcher (`py`). Giải nén ZIP, mở PowerShell tại thư mục `flappy-bird-bot` chứa `main.py`, rồi chạy:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\install.ps1
.\run.bat
```

Installer tạo `.venv` ngay trong project và cài thư viện. Không cần quyền admin, VS Code hoặc môi trường của Codex. Lệnh tương đương:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 main.py
```

`requirements-tested.txt` ghi các bản thư viện của môi trường kiểm thử code. Lần khởi động đầu Numba cần biên dịch; chờ thông báo chuyển về tab game rồi thực hiện bước 4.

## 2. Mở game và chọn ROI

1. Bạn tự mở https://minigametuctactea.pages.dev/ trên Edge hoặc Chrome, đăng nhập và chọn **Hổ Không Gầm**.
2. Giữ browser zoom 100%, chọn kích thước cửa sổ ổn định; để màn hình **SẴN SÀNG** hiện trên canvas.
3. Trong app, bấm **Chọn vùng game**. App ẩn cửa sổ, hiển thị ảnh màn hình để bạn kéo khung quanh canvas có viền nâu. Không lấy bảng xếp hạng bên phải hoặc tiêu đề trang. Enter xác nhận; Esc hủy.
4. Bấm **Lưu cấu hình**. ROI lưu trong hồ sơ đang dùng. Nếu đổi màn hình, zoom, vị trí hoặc kích thước cửa sổ, chọn ROI lại.

Mặc định bot dùng `profiles/live-20261013-latency.yaml`, gồm ROI và thông số của lượt live đạt 1.196 điểm. Giữ cùng bố cục/kích thước cửa sổ khi chạy lại; nếu đổi vị trí hoặc màn hình, bấm **Chọn vùng game** rồi **Lưu cấu hình**. Tên tab phải chứa `CUỘC ĐUA`; bot chặn click khi game mất foreground. Hồ sơ cũ `profiles/windows-click.yaml` vẫn được giữ để rollback.

## 3. Chọn và cấu hình planner

Mặc định app nay nạp `profiles/live-20261013-latency.yaml`, cấu hình đã dùng trong lượt live vượt 1.000 điểm. Để thử lại hồ sơ đối chiếu cũ, mở riêng:

```powershell
.\run.bat --config profiles/windows-reachability.yaml
```

ROI của hai hồ sơ lưu riêng; chọn lại khi dùng hồ sơ lần đầu. Có thể bấm **Thông số** sau F8 để thay đổi và lưu:

| Thông số | Ý nghĩa / giá trị bắt đầu |
|---|---|
| FPS | Giới hạn vòng lặp; 45 cho đối chiếu, 60 cho hồ sơ ngược; FPS đo được có thể thấp hơn |
| Trọng lực | 710,38 px/s² |
| Vận tốc nhảy | −248,58 px/s; âm là đi lên |
| Tốc độ cột | 139,6 px/s |
| Trễ input | 32 ms; không phải thời gian chạy riêng hàm click |
| Khoảng cách click | 180 ms |
| Biên dưới | 356 trên ảnh chuẩn hóa rộng 600 px |
| Nửa cao hitbox | Trống = bbox; 20 px trong hồ sơ thử, cần hiệu chỉnh bằng log |
| Planner | `maxmin` hoặc `reachability` |
| Tự click CHƠI LẠI | Mặc định tắt; bật khi muốn thử nhiều lượt liên tục |

Các giá trị là điểm khởi đầu cho nhân vật/khung game này, không đảm bảo 1.000 điểm. Đừng đồng thời thay nhiều thông số khi so sánh. `reachability` giữ bảng hình học cố định cho cache; khi không đặt hitbox riêng, dùng nửa rộng 27/nửa cao 25 px. Hồ sơ thử đặt 26/20 px. Không suy kích thước hitbox engine chính xác từ bbox đồ họa.

## 4. Bắt đầu, tạm dừng và dừng

1. Để game ở **SẴN SÀNG**, nhấn **F6** trong app hoặc hotkey toàn hệ thống.
2. Khi app thông báo chờ 3 giây, chuyển về tab game. Bot nhận đúng mẫu SẴN SÀNG, tìm nút vàng và tự click **VÀO ĐUA**, sau đó tự click trong canvas để nhảy.
3. Giữ game foreground. Đổi sang ứng dụng khác hoặc di chuyển cửa sổ khiến bot tạm dừng.
4. **F6** tạm dừng/tiếp tục click. Tạm dừng bot không tự tạm dừng website; game có thể tiếp tục rơi. Nếu website đang tạm dừng, bạn tự bấm nút tiếp tục của game rồi F6.
5. **F8** dừng gửi click; chờ worker ghi xong log. Đưa chuột tới góc màn hình cũng kích hoạt failsafe.
6. Sau khi chết, app đọc điểm OCR. Nếu bật tự chơi lại, app click CHƠI LẠI rồi VÀO ĐUA. Nếu tắt, bạn tự bấm CHƠI LẠI và F6.
7. **F7** bật/tắt debug. Cửa sổ debug có thể chiếm foreground; nên tắt debug trong lượt đo điểm.

Nếu không tự nhận SẴN SÀNG/kết thúc: F8, để đúng màn hình tương ứng, bấm **Sẵn sàng** hoặc **Kết thúc**, kéo khung quanh chữ tiêu đề trong ảnh, Enter. Các mẫu được lưu trong `templates/`. Với Hổ Không Gầm nên giữ chế độ màu cam; template nhân vật có thể kém khi sprite xoay.

## 5. FPS, điểm và log

- App hiện FPS, thời gian quan sát lượt, vận tốc và số khe nhận diện. Thời gian là clock ứng dụng; lượt có tạm dừng không dùng số này thay cho thời gian nội bộ của game.
- Điểm ảnh trực tiếp có thể là `?` nếu chưa có đủ mẫu chữ số. Sau khi chết, **Lượt trước (OCR)** và **Cao nhất đọc được** được cập nhật bằng Windows OCR. Nếu OCR không đọc được, giữ `?`, không nhận là 0 hay suy điểm từ thời gian.
- `logs/run-*.jsonl`: vị trí, khe, vận tốc, hành động, pending input và thời gian từng bước.
- `logs/run-*.json`: cấu hình/hash, sự kiện đầu/cuối lượt, FPS và độ trễ median/p95; tạo đầy đủ sau F8.
- `logs/run-*-result-*.json`: điểm OCR, ảnh nguồn và lỗi OCR nếu có.
- `screenshots/run-*-end-*.png` và ảnh `*-pre-*.jpg`: kết thúc và các frame trước đó, do ứng dụng tự lưu.

Windows OCR cần có ngôn ngữ OCR khả dụng trong Windows. Luôn đối chiếu điểm OCR với số hiện trên website trước khi báo đạt mục tiêu.

## 6. Gửi dữ liệu khi bot chơi sai

F8, chờ bot ghi log xong, bấm **Xuất 3 phiên log gần nhất**. Gửi ZIP ở `exports/` cùng:

- Điểm thực tế trên website, nhân vật và hồ sơ đang dùng.
- FPS quan sát và lỗi: nhảy sớm/muộn, đụng trên/dưới, không tìm được khe, không click, mất focus.
- Ảnh game do bạn chụp nếu ROI hoặc kết quả OCR sai.

Có thể xuất bằng lệnh, không chụp ảnh mới:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m tools.export_logs --last 3
```

Tôi sẽ phân tích dữ liệu bạn gửi và hướng dẫn chạy lại; không tự mở hoặc theo dõi trình duyệt bằng Computer Use.

## 7. Kiểm thử code và benchmark

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp .test-tmp
.\.venv\Scripts\python.exe -m compileall -q main.py bot ui sim tools
.\.venv\Scripts\python.exe -X utf8 -m tools.benchmark_standalone --runs 10 --gates 12 --observed-physics
```

Mô phỏng không mở browser, không MSS, không click thật. FPS MSS chỉ đo khi bạn tự chạy trên máy mình. Có runner đo capture/detection không gửi input:

```powershell
.\.venv\Scripts\python.exe -X utf8 local_runner.py --config profiles/windows-click.yaml --benchmark --seconds 10 --prepare-delay 3 --name my-capture-benchmark
```

Đưa game ra foreground trong thời gian chờ; không cần vào lượt chơi. Runner ghi JSON timing. Mỗi lần chọn tên mới để không ghi đè log. Bản ZIP chạy app không kèm các adapter browser cũ.

Chi tiết thuật toán và giới hạn kiểm thử ở REPORT.md.

## 8. Hồ sơ vật lý gốc mới (desktop)

Hồ sơ `profiles/windows-source.yaml` giữ riêng, không thay thế baseline. Source xác nhận Canvas **900×600**, groundY **535**, fixed update **1/120 s**. Sau khi ROI đầy đủ được resize về **600×400**, các giá trị là g=720, jump=−256,6667, tốc độ ống=136,6667 px/s. Kích thước hitbox ống là 33,3333×36,6667 px; trần/đất dùng sprite 48×48 px. Biên floor nội bộ của planner không bằng groundY trên ảnh vì nó dùng tâm hitbox và phải tái hiện điều kiện sprite chạm đất.

Trong thư mục project có `.venv`, chạy:

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --config profiles/windows-source.yaml
```

1. Tự mở game trên Edge/Chrome, dùng giao diện desktop (viewport CSS rộng hơn 640 px), chọn Hổ Không Gầm.
2. Trong app chọn vùng **toàn bộ canvas**, từ góc trên trái đến dưới phải, gồm nền trời, cột, mặt đất và phần đáy; không lấy bảng xếp hạng, không cắt riêng vùng nhân vật/khe. Ưu tiên sát bên trong viền nâu. ROI phải có tỉ lệ gần 3:2; app chặn ROI lệch quá 5%. Tỉ lệ đúng chưa chứng minh gốc ROI đúng, vì vậy phải chọn đủ canvas.
3. Lưu cấu hình. Giữ nguyên zoom, vị trí và kích thước browser. DPI/displayWidth chỉ đổi kích thước pixel capture; không nhân vật lý thêm với devicePixelRatio. MSS dùng ROI pixel màn hình, OpenCV resize theo chiều rộng và giữ tỉ lệ.
4. F6 một lần. App tự thu nhỏ, biên dịch trước lượt và chờ đúng trình duyệt ở foreground. Lần đầu Numba có thể lâu; không bấm F6 lặp. F8 vẫn dừng được.
5. Giữ game foreground. F6 tạm dừng/tiếp tục; F8 dừng. Bot tự click VÀO ĐUA khi mẫu SẴN SÀNG hợp lệ.
6. Sau lượt, F8, chờ ghi log rồi xuất ba phiên. Gửi ZIP, điểm trên website và FPS app. Log mới có `prediction_residual_px`, `requested_flap`, `input_blocked_reason` để phân biệt sai dự đoán, không muốn nhảy và nhảy bị chặn.

Hồ sơ live đã kiểm thử dùng planner reachability, bù độ trễ 32 ms cùng pipeline click, hình học va chạm theo source và ROI của phiên đạt 1.196 điểm. Lượt đó ghi FPS capture trung vị 58,5 và độ trễ capture-to-input trung vị 28,7 ms. Các số này phụ thuộc máy và bố cục cửa sổ; giữ nguyên ROI nếu muốn tái lập phép thử. `profiles/windows-click.yaml` còn nguyên làm cấu hình rollback.

Bước vật lý 120 Hz khác FPS capture 30. Các ảnh quan sát không cung cấp chính xác phase accumulator hoặc thời điểm browser nhận click. Công thức bù Euler khớp ở các tick; kiểm tra liên tục giữa tick là bảo thủ. Dự đoán đoạn lẻ/timing input vẫn có xấp xỉ, cần log để kiểm chứng. Mobile 900×900 có offset100 bị chặn ở hồ sơ này vì chưa tái hiện render offset.

Mẫu END cũ tối bị loại. Để đọc điểm/tự chơi lại, sau lượt thật: F8, để hộp thoại Va chạm hiện, bấm **Kết thúc** và chọn đúng vùng **chữ “Va chạm rồi!” trên nền sáng**, Enter. Không chọn nền đen, avatar hay nút. Điểm OCR chưa đọc được vẫn là `?`.

Benchmark code, không mở browser hay chụp màn hình:

```powershell
.\.venv\Scripts\python.exe -m pytest tests -q --basetemp .test-source
.\.venv\Scripts\python.exe -X utf8 -m tools.benchmark_source --runs 6 --gates 20 --beam 600 --pipeline-delay-ms 50
```

Rollback cấu hình bằng hồ sơ cũ:

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --config profiles/windows-reachability.yaml
```

Nếu muốn rollback cả code: đóng bot, giải nén snapshot `backups/before-source-physics-*.zip` ra thư mục riêng rồi chép lại bot/ui/sim/tools cùng hồ sơ cần khôi phục. Sao lưu cấu hình và log mới trước khi chép. Snapshot giữ code trước thay đổi, không khẳng định đó là bản đã đạt 1.000 điểm.

Bản source giảm nhiễu vận tốc bằng prior từ lần click gần nhất với g/J đã xác nhận, trọng số 0,7. Nếu vận tốc từ ảnh lệch quá 80 px/s (ảnh chuẩn hóa), bỏ prior và ghi `source_velocity_mismatch`; không giấu dữ liệu trái với xung nhảy. Chu kỳ hành động tương lai còn tăng theo `reaction_interval` đo được nếu capture chậm hơn 30 FPS, thay vì giả định luôn có quyết định mới sau 33 ms.

Bản sửa sau log57điểm: source giữ tốc độ tracker theo source, hiệu chỉnh clipping cạnh ảnh; mẫu END đã được khôi phục từ ảnh người dùng cung cấp. Không cần chọn END lại nếu màn hình giữ bố cục hiện tại. Source dùng source_high_resolution_wait=true để yêu cầu timer1ms; có thể tắt bằng false trong YAML. FPS tăng hay không cần kiểm tra log previous_wait_requested_ms/actual_ms ở lượt do bạn chạy, chưa được xác minh.

## Thử nghiệm sau lượt 312 điểm

Lượt do người dùng chạy với `windows-reachability-top-guard.yaml` đạt 312 điểm, 445,37 giây; đó là bản tốt nhất được người dùng báo **trong đợt thử reachability này**. Kỷ lục toàn bộ project đã xác nhận trước đó là **512 điểm/731,65 giây**, ghi trong `REPORT-history.md` và `evaluation/goal1000-timer-summary.json`. Hồ sơ 512 lịch sử nằm ở `evaluation/live-grid-timer.yaml`; mã nguồn khi đó đã thay đổi qua nhiều đợt và ROI màn hình cũng khác, nên không coi việc dùng YAML ấy với code hiện tại là khôi phục chính xác phiên 512. Hồ sơ 312 được giữ nguyên; bản ZIP trước thay đổi tiếp theo lưu tại `backups/top-guard-312.zip`.

Log cuối lượt cho thấy nhân vật rơi gần mép dưới của khe trong lúc planner còn chọn COAST, rồi click muộn. Bản thử cứu mép dưới sau đó **cho kết quả kém hơn** trong thử nghiệm của người dùng và đã được rút khỏi hồ sơ chạy. Cấu hình thất bại được lưu ở `backups/windows-reachability-two-sided-failed.yaml` để phân tích, không nên chạy lại.

Đóng bot đang chạy, rồi trong thư mục `flappy-bird-bot` chạy lại bản đạt 312 điểm:

```powershell
.\run.bat --config profiles/windows-reachability-top-guard.yaml
```

Chọn vùng game như cũ nếu app yêu cầu, nhấn F6 một lần, đưa trình duyệt game lên trước và để bot chơi. F8 để dừng, bấm “Xuất 3 phiên log gần nhất” và gửi ZIP cùng điểm thực tế. Không cần giải nén snapshot: code top guard đã được khôi phục trong thư mục đang dùng.

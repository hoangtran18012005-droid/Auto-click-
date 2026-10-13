# Học tăng cường Double DQN ngoại tuyến

## Học hành động từ planner ở tình huống khó (đợt mới nhất)

`rl.imitation` hỏi bộ lập kế hoạch hiện có trong simulator rồi lưu cặp trạng thái–hành động. Có thể chỉ lấy các quyết định ở sát ống, khi khe kế tiếp thay đổi độ cao, hoặc khi model cũ có nguy cơ chọn sai. Chế độ `--expert-rollout` để planner điều khiển trọn lượt, tạo chuỗi trạng thái nhất quán thay vì trộn các quyết định của hai policy. Huấn luyện dùng loss ưu tiên hành động planner, đồng thời neo Q của model cũ ở trạng thái ổn định. `best.npz` được chọn trên tập validation và có thể giữ nguyên model ban đầu nếu các epoch mới kém hơn.

Kết quả kiểm thử ngoại tuyến hiện tại: planner đạt giới hạn 30/30 cột trong cả 8 lượt thu thập, so với model cũ đạt 30/30 trong 5/8 lượt trên cùng seed. Sau khi học, model mới đạt trung bình 123,42 cột trên 60 seed mới, model cũ 125,23; cả hai đều 0/60 lượt đạt 1.000. Vì vậy **model khuyến nghị vẫn là `models/dqn-continued/best.npz`**. Model imitation nằm riêng trong `models/dqn-imitation-expert/best.npz` để nghiên cứu, không được ứng dụng Windows tải tự động. Đây là điểm simulator dùng trạng thái vật lý thật, chưa phải điểm website.

Để tự thu thập và huấn luyện lại trong PowerShell ở thư mục `flappy-bird-bot`:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.imitation --resume models/dqn-continued/best.npz --episodes 8 --seed-start 1250000 --max-score 30 --query-every 1 --expert-rollout --anchor-weight 10 --margin 0.01 --epochs 30 --output models/dqn-imitation-my-run
```

Có thể dùng lại dữ liệu đã lưu, không cần hỏi planner lần nữa:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.imitation --resume models/dqn-continued/best.npz --dataset models/dqn-imitation-expert/dataset.npz --anchor-weight 10 --margin 0.01 --epochs 30 --output models/dqn-imitation-retrain
```

Để so sánh trên seed mới, dùng `rl.evaluate` với cùng `--episodes`, `--seed-start`, `--max-score` và `--max-steps` cho hai checkpoint. `dataset-summary.json` ghi số lần hỏi planner, nhãn được chấp nhận, hành động khác model cũ và độ trễ planner. `imitation.jsonl` ghi validation theo epoch. Các file này không đo FPS MSS trên Windows. Snapshot trước đợt học nằm tại `backups/before-planner-imitation.zip`.

## Học tiếp có tập trung vào tình huống khó

**Kết quả đợt thử:** đã chạy2.000.000 bước curriculum và600.000 bước tinh chỉnh nhẹ hơn. Cả hai model cuối kém hơn model cũ trên cùng seed người dùng và30 seed mới; vì vậy model khuyến nghị vẫn là `models/dqn-continued/best.npz`. Các thư mục curriculum/finetune chứa model thử nghiệm và log; `best.npz` của chúng giữ nguyên trọng số model cũ do không vượt tiêu chí validation. Không xem số bước học thêm là bằng chứng cải thiện. Số liệu đầy đủ trong REPORT.md.

Trainer hỗ trợ `--epsilon-final`, `--curriculum-probability`, `--hard-replay-fraction` và `--validation-max-score`. Các mặc định cũ giữ nguyên. Đợt học sau log người dùng sử dụng:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.train --resume models/dqn-continued/best.npz --steps 2000000 --seed 9 --learning-rate 0.0001 --epsilon-final 0.005 --curriculum-probability 0.2 --hard-replay-fraction 0.25 --validation-max-score 100 --eval-every 200000 --eval-episodes 12 --output models/dqn-curriculum-my-run
```

Đây là lệnh huấn luyện, có thể chạy vài phút hoặc lâu hơn tùy CPU. Với cấu hình này:

- 20% episode dùng khe cực trị cao/thấp luân phiên, 80% còn lại dùng khe ngẫu nhiên.
- Xác suất chọn hành động ngẫu nhiên sau giai đoạn giảm epsilon là 0,5%, thay cho5% của cấu hình cũ.
- 25% minibatch lấy từ replay riêng lưu tối đa20.000 transition thuộc75 bước cuối trước va chạm đầu ống; phần còn lại lấy từ replay thông thường. Không đổi reward hoặc công thức Double DQN target. Đây là thay đổi phân bố lấy mẫu, chưa phải prioritized replay có importance weights.
- Validation tăng giới hạn lên100 cột. Khi bật curriculum, tiêu chí chọn best là80% điểm trung bình của12 seed ngẫu nhiên và20% điểm của một chuỗi cực trị cố định. Chuỗi cực trị không được tính là nhiều tình huống độc lập.

Học thêm có thể làm giảm kết quả. `best.npz` giữ cả model trước khi học tiếp trong lựa chọn; `last.npz` có thể kém hơn. Model cũ tại `models/dqn-continued/best.npz` luôn được giữ riêng. Chỉ kết luận cải thiện khi đánh giá trên cùng seed cũ và thêm seed mới, với cùng giới hạn cột/bước. Để thử checkpoint mới, đổi đường dẫn `--checkpoint` trong các lệnh đánh giá phía dưới. Không chạy hai trainer cùng một thư mục output.

Trainer hiện lưu thêm `checkpoint-<tổng số bước>.npz` sau mỗi lần validation để giữ các ứng viên trung gian. Cải tiến này được bổ sung sau hai lượt huấn luyện trên, nên không thể khôi phục các checkpoint trung gian đã bị ghi đè trong hai lượt đó. Log mới có dòng `settings` ghi tham số và `hard_replay_size` trong tiến độ.

Đây là phần mở rộng của project hiện tại. Bộ điều khiển Windows, MSS/OpenCV, click chuột và các hồ sơ YAML được giữ nguyên. `main.py` chưa tải model DQN. Chạy các lệnh dưới đây chỉ chạy mô phỏng: không mở trình duyệt, không chụp màn hình và không gửi click. Model chưa được xác nhận đạt 1.000 điểm hoặc chơi tốt hơn bộ lập kế hoạch hiện tại.

## Chuẩn bị

Mở PowerShell trong thư mục `flappy-bird-bot` chứa `main.py`, `rl` và `profiles`. Nếu vừa giải nén gói `FlappyBirdAutoPlayer`, vào thư mục con:

```powershell
cd .\flappy-bird-bot
powershell -NoProfile -ExecutionPolicy Bypass -File install.ps1
```

Đợi cài đặt hoàn tất, không hủy pip giữa chừng. Nếu đã có `.venv` hoạt động thì không cần cài lại. DQN dùng NumPy có sẵn trong requirements, không cần PyTorch, GPU hoặc Gymnasium.

```powershell
.\.venv\Scripts\python.exe -c "import numpy, cv2; print('Thư viện đã sẵn sàng')"
.\.venv\Scripts\python.exe -X utf8 -m rl.train --help
.\.venv\Scripts\python.exe -X utf8 -m rl.evaluate --help
```

## Huấn luyện

Khởi tạo model riêng để không ghi đè model mẫu:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.train --steps 400000 --seed 7 --epsilon-decay 120000 --eval-every 40000 --eval-episodes 10 --output models/dqn-my-run
```

Hai hành động là COAST và FLAP. Mỗi bước ra quyết định mặc định tương ứng 4 tick vật lý, tức 30 Hz của thời gian game ảo; vật lý vẫn cập nhật ở 120 Hz. Tốc độ in `steps_per_second` là tốc độ chạy mô phỏng/huấn luyện trên CPU, không phải FPS MSS hay FPS website. Có thể chạy hàng nghìn bước ảo mỗi giây.

Các tham số: `--steps` số bước bổ sung, `--seed` seed huấn luyện (>=1), `--hidden` số neuron mỗi lớp ẩn, `--batch` cỡ minibatch, `--learning-rate` tốc độ học, `--warmup` số bước thu replay trước khi học, `--epsilon-decay` tổng số bước để giảm xác suất khám phá từ 1 xuống 0,05. Mặc định MLP 22→64→64→2, batch 64, Adam 0,0003, Huber loss và clip gradient. Cập nhật một minibatch mỗi 4 bước môi trường; target network được đồng bộ sau 1.000 lần cập nhật gradient. Replay lưu tối đa 100.000 transition.

Sau mỗi đợt đánh giá, thư mục output có:

- `best.npz`: checkpoint có điểm trung bình validation tốt nhất trong lượt chạy, kể cả model trước khi học tiếp.
- `last.npz`: checkpoint mới nhất; có thể kém hơn best.
- `train-*.jsonl`: tiến độ, loss, epsilon và kết quả từng episode validation.
- `interrupted.npz`: lưu khi nhấn Ctrl+C để kết thúc sớm.

Không có F6/F8 trong chương trình huấn luyện. Ctrl+C dừng và lưu checkpoint; chương trình không có chức năng tạm dừng bằng phím. Học tiếp vào **thư mục output mới**:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.train --resume models/dqn-offline/best.npz --steps 1200000 --seed 8 --eval-every 100000 --eval-episodes 10 --output models/dqn-my-continued
```

Resume khôi phục trọng số, Adam và số bước đã học để không đặt epsilon về 1. Replay, target network, RNG và episode đang chạy được khởi tạo lại; đây không phải khôi phục bit-for-bit toàn bộ tiến trình. Đổi seed giúp học trên chuỗi cột khác. Trọng số chỉ thay đổi khi chạy `rl.train`; `rl.evaluate` và ứng dụng Windows không tự học từ các lượt thua.

## Đánh giá trên seed mới

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.evaluate --checkpoint models/dqn-offline/best.npz --episodes 30 --seed-start 20000 --max-score 100 --output evaluation/dqn-my-heldout.json
```

Chọn `--checkpoint` đúng model muốn đánh giá, chẳng hạn `models/dqn-continued/best.npz`. Dùng cùng seed và cùng giới hạn điểm khi so sánh các model. Đánh giá greedy, epsilon=0. Đừng chọn lại model theo tập held-out rồi coi tập đó vẫn là dữ liệu kiểm tra độc lập; hãy dành seed mới cho lần xác nhận tiếp theo.

Model tốt nhất của đợt này nằm tại `models/dqn-continued/best.npz`; model 400.000 bước ban đầu được giữ riêng trong `models/dqn-offline/best.npz`. Để thử giới hạn 1.000 cột trong simulator:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.evaluate --checkpoint models/dqn-continued/best.npz --episodes 10 --seed-start 70000 --max-score 1000 --max-steps 50000 --output evaluation/dqn-my-1000.json
```

Đối chiếu với bộ lập kế hoạch source hiện tại:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.evaluate --checkpoint models/dqn-offline/best.npz --episodes 10 --seed-start 20000 --max-score 20 --compare-mpc --output evaluation/dqn-my-vs-mpc.json
```

So sánh dùng cùng chuỗi cột, trạng thái y/v của simulator, nhiễu nếu có, độ trễ 32 ms và cooldown 120 ms. Không chạy nhận diện ảnh thật; vận tốc oracle cũng cấp cho MPC để tránh ưu thế dữ liệu của DQN. Thời gian tính kế hoạch MPC được đo nhưng không cộng vào đồng hồ game ảo; cấu hình ngân sách dự phòng planner đặt 0 chỉ trong evaluator. Vì vậy đây là so sánh policy trong mô phỏng, không đo toàn bộ pipeline Windows. Đối chiếu MPC có thể chạy lâu.

Các bài thử bổ sung:

```powershell
.\.venv\Scripts\python.exe -X utf8 -m rl.evaluate --checkpoint models/dqn-offline/best.npz --episodes 30 --seed-start 30000 --noise-y 1.5 --latency-jitter 0.008 --output evaluation/dqn-my-noisy.json
.\.venv\Scripts\python.exe -X utf8 -m rl.evaluate --checkpoint models/dqn-offline/best.npz --episodes 10 --alternating --output evaluation/dqn-my-alternating.json
.\.venv\Scripts\python.exe -X utf8 -m pytest tests -q --basetemp .test-rl
```

Noise y tính theo pixel canvas native; jitter tính theo giây. `--alternating` tạo khe cao/thấp cực trị luân phiên. JSON ghi seed, số cột vượt, thời gian sống ảo, terminated/truncated, reward tổng, điểm trung bình/trung vị/min/max và thời gian quyết định median/p95/max. Truncated là chạm giới hạn bài thử, không được báo là chết. Giới hạn mặc định 10.000 bước cũng có thể kết thúc lượt trước giới hạn điểm rất cao.

## Mô hình và giới hạn

Môi trường tái sử dụng oracle `sim/source_engine.py`: canvas 900×600, groundY 535, gravity 1080, flap -385, pipe speed 205; cập nhật vy trước y, sprite/hitbox và bốn rectangle ống theo dữ liệu JavaScript người dùng cung cấp. Việc sinh chuỗi cột, phase input, thời điểm cột đầu tiên và việc đếm điểm vẫn là giả định mô phỏng. Số cột vượt trong JSON không phải điểm website đã xác nhận.

State có 22 giá trị chuẩn hóa: vị trí/vận tốc, cooldown, input đang chờ, thời lượng frame, độ trễ cấu hình, khoảng cách biên, tối đa ba cột đang nhìn thấy với mask, sai lệch tới tâm khe. Cột tương lai ngoài canvas bị che; không cho policy biết toàn bộ chuỗi. Vị trí và vận tốc hiện lấy trực tiếp từ simulator, có tùy chọn thêm nhiễu; chưa lấy từ tracker OpenCV. Khi chuyển sang máy thật phải tái hiện đúng schema, sai số quan sát và thời điểm input.

Reward gồm +1 khi vượt cột, -2 khi chết và potential shaping hướng về khe với hệ số 0,2. Bellman target dùng online network chọn hành động và target network đánh giá hành động đó, có mask cooldown/input. Truncated giữ bootstrap; chết thì không. Discount đổi theo thời lượng bước khi mô phỏng mất frame. Mask/cooldown là giới hạn điều khiển của bot, không phải giới hạn click trong JavaScript.

Loss giảm không chứng minh policy chơi tốt. Chỉ cân nhắc tích hợp khi điểm và tỷ lệ sống trên nhiều seed mới tốt hơn baseline, bền với khe đổi độ cao/nhiễu/độ trễ, rồi người dùng xác nhận trên Windows. Hiện chưa thay thế bộ điều khiển đang dùng.

## Chạy bot Windows và gửi dữ liệu

Lệnh bot Windows giữ nguyên:

```powershell
.\.venv\Scripts\python.exe -X utf8 main.py --config profiles/windows-source.yaml
```

Xem README mục hồ sơ source để chọn đầy đủ canvas tỉ lệ 3:2, lưu cấu hình, F6 bắt đầu/tạm dừng và F8 dừng. Đây vẫn là planner source, chưa phải DQN. Sau lượt thật dùng “Xuất 3 phiên log gần nhất”, gửi ZIP cùng điểm website và FPS. Với lỗi học/đánh giá offline, gửi `train-*.jsonl`, JSON đánh giá, câu lệnh đầy đủ và traceback. Không cần mở game khi chạy các lệnh RL.

Snapshot `backups/before-offline-dqn.zip` lưu gói trước đợt mở rộng RL. Cấu hình baseline không bị ghi đè; model nằm riêng trong `models/`.

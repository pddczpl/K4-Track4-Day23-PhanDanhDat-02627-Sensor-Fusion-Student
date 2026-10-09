# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Phan Danh Đạt
- MSSV: 02627
- Email: phandanhdattt@gmail.com
- Link repo (fork): https://github.com/pddczpl/K4-Track4-Day23-PhanDanhDat-02627-Sensor-Fusion-Student
- Commit hash nộp (`git rev-parse HEAD`): HEAD

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`: compare, [0, 198], training_segment-1005081002024129653_5313_150_5333_150_with_camera_labels.tfrecord, 0
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`: 0.9701 (97.01%), 0.7004 (70.04%), tp: 519 / fp: 16 / fn: 222
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse: 0.1503 m, matches: 502, sum_sq_err: 11.3436, ghost_track_frames: 0, missed_gt_frames: 239, mean_confirmed_tracks: 2.5226
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`: rmse: 0.1359 m, matches: 502, sum_sq_err: 9.2668, ghost_track_frames: 0, missed_gt_frames: 239, mean_confirmed_tracks: 2.5226
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:
  Ở cả hai chế độ (lidar-only và fused), số lượng confirmed tracks được ghép cặp thành công (matches = 502) và số lượng ground-truth bị bỏ lỡ (missed_gt_frames = 239) là hoàn toàn giống nhau, đồng thời số lượng track ảo (ghost_track_frames = 0) đều bằng 0 và trung bình số confirmed track/frame đều là 2.5226. Điều này phản ánh đúng thiết kế kiến trúc track-then-fuse của lab: việc khởi tạo (init), tính điểm tồn tại (score) và xóa track (delete) hoàn toàn do modality LiDAR chi phối, do đó camera không sinh thêm track mới hay gây ra ghost tracks.
  Tuy nhiên, về độ chính xác vị trí (RMSE), chế độ fused đạt RMSE = 0.1359 m (tổng bình phương sai số 9.2668 m²), cải thiện đáng kể so với chế độ lidar-only có RMSE = 0.1503 m (tổng bình phương sai số 11.3436 m²). Lý do là khi track nằm trong FOV của camera trước (FRONT), các phép đo 2D pinhole (tâm hộp xe) kết hợp với ma trận Jacobian H đã bổ sung thông tin góc nhìn và tinh chỉnh thêm trạng thái của bộ lọc EKF (giảm hiệp phương sai sai số vị trí theo phương ngang và dọc), giúp quỹ đạo bám sát ground-truth hơn.

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?
   - Vector đo lường $z$: LiDAR đo trực tiếp toạ độ 3D của tâm vật thể trong hệ toạ độ cảm biến $z = [x, y, z]^T \in \mathbb{R}^3$ (đơn vị: mét), hàm đo tuyến tính với ma trận quan sát $H$ kích thước $3 \times 6$. Camera đo toạ độ điểm ảnh 2D trên mặt phẳng ảnh pinhole $z = [u, v]^T \in \mathbb{R}^2$ (đơn vị: pixel), hàm đo $h(x)$ phi tuyến tính ($u = c_i - f_i \frac{y_s}{x_s}, v = c_j - f_j \frac{z_s}{x_s}$), cần tính ma trận Jacobian $H$ kích thước $2 \times 6$ theo đạo hàm quy tắc chuỗi.
   - Hiệp phương sai nhiễu đo $R$: LiDAR có $R = \text{diag}(\sigma_x^2, \sigma_y^2, \sigma_z^2) \in \mathbb{R}^{3 \times 3}$ với $\sigma = 0.1\text{ m}$ (đơn vị $m^2$). Camera có $R = \text{diag}(\sigma_i^2, \sigma_j^2) \in \mathbb{R}^{2 \times 2}$ với $\sigma = 5.0\text{ pixel}$ (đơn vị $\text{pixel}^2$).

2. Vì sao cần gating Mahalanobis trước khi gán?
   - Gating Mahalanobis tính khoảng cách chuẩn hoá theo hiệp phương sai đổi mới $S = H P H^T + R$: $d^2 = \gamma^T S^{-1} \gamma$, và so sánh với ngưỡng phân phối $\chi^2$ tương ứng với số chiều đo lường (3 đối với LiDAR, 2 đối với Camera).
   - Việc này giúp loại bỏ sớm các đo lường ngoại lai (outliers/clutter) và các vật thể không thuộc track trước khi đưa vào thuật toán Greedy, ngăn chặn ghép nhầm giữa các đối tượng ở gần nhau.
   - So với khoảng cách Euclidean thuần tuý, Mahalanobis tính đến độ bất định phân phối xác suất ($P$ và $R$): nếu track có độ bất định lớn theo một hướng (ví dụ hướng vận tốc hoặc chiều sâu camera), sai số Euclidean lớn vẫn có thể được chấp nhận; ngược lại khi độ bất định rất nhỏ, một sai số Euclidean nhỏ cũng có thể bị loại nếu nó không phù hợp thống kê.
   - Ngoài ra, kiểm tra FOV trước Mahalanobis ngăn chặn việc chiếu điểm không hợp lệ (phía sau camera $x_s \le 0$).

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.
   - Pipeline của lab là **track-then-fuse** (state-level / tracking-level fusion).
   - Hệ thống duy trì duy nhất một danh sách track (track list). Trong mỗi frame, pipeline thực hiện dự báo (predict) một lần cho mọi track, sau đó tiến hành gán và cập nhật EKF lần lượt theo từng cảm biến: đầu tiên là gán đo LiDAR (AssocL) và cập nhật EKF, tiếp theo là gán đo Camera (AssocC) và cập nhật EKF trên chính các track đó. Không có bước gộp dữ liệu cảm biến thô (raw data) trước detector.
   - Trên log của `fusion-run-lab` và `grade_run.log`, cấu trúc này thể hiện rõ: các record được ghi theo từng frame và mode; chế độ `lidar` chạy quy trình gồm detection -> predict -> AssocL, còn chế độ `fused` chạy tiếp tục bước AssocC sau AssocL trên cùng một bộ track, không có bước detector đa cảm biến độc lập trước khi tạo track.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?
   - Innovation $\gamma = z - h(x)$ là độ lệch giữa toạ độ điểm ảnh thực tế đo được và toạ độ điểm ảnh dự đoán qua mô hình pinhole $h(x)$.
   - Nếu camera bị lệch calibration (lệch góc xoay/tịnh tiến extrinsic hoặc tiêu cự/tâm ảnh intrinsic), phép chiếu $h(x)$ sẽ bị dịch chuyển có hệ thống (systematic bias).
   - **Triệu chứng:** Residual $\gamma$ sẽ không còn mang tính chất nhiễu trắng Gaussian có kỳ vọng bằng 0 ($\mathbb{E}[\gamma] \neq 0$), mà xuất hiện độ lệch dịch chuyển cố định (nonzero mean/systematic error) nhất quán qua nhiều frame. Khi đó khoảng cách Mahalanobis $\gamma^T S^{-1} \gamma$ tăng cao dẫn đến đo lường thường xuyên bị gate $\chi^2$ từ chối (không ghép được camera), hoặc nếu lọt qua gate thì Kalman update sẽ kéo lệch vị trí 3D của track, làm sai số RMSE tăng vọt.

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.
   - Cần sensor tường minh: Ở những frame không có đo lường nào (`meas_list` rỗng), hàm `associate_and_update` vẫn bắt buộc phải gọi `manager.manage_tracks(unassigned_tracks, unassigned_meas, sensor)` ở cuối lượt. Nhờ đối tượng `sensor` tường minh, manager mới phân biệt được đây là kết thúc lượt LiDAR hay Camera. Nếu là lượt LiDAR, các track nằm trong FOV mà không có đo lường sẽ bị trừ điểm tồn tại (`score -= 1/window`) và kiểm tra điều kiện xoá; nếu là lượt Camera, manager sẽ bỏ qua (không trừ điểm, không xoá track).
   - Vì sao LiDAR quyết định score/init/delete còn Camera chỉ EKF update: LiDAR cung cấp đầy đủ thông tin toạ độ 3D trực tiếp trong không gian và có trường nhìn bao quát, đủ thông tin hình học để khởi tạo track mới (position 3D) và khẳng định sự tồn tại/biến mất của vật thể. Ngược lại, Camera chỉ cho toạ độ ảnh 2D (góc nhìn), bị mất thông tin chiều sâu (độ sâu đơn ảnh là vô số nghiệm dọc theo tia chiếu), nên không thể tự khởi tạo track 3D mới hoặc xác định vật thể đã rời khỏi hiện trường. Do đó, Camera chỉ đóng vai trò cập nhật tinh chỉnh trạng thái EKF cho các track đã có.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.
   - **Khởi tạo và xác nhận (Confirmation):**
     - Track mới sinh từ đo LiDAR chưa gán: điểm ban đầu `score = 1/window`, trạng thái `"initialized"`.
     - Mỗi frame có đo LiDAR: nếu ghép trúng (hit), `score = min(1.0, score + 1/window)`. Track `"initialized"` có hit chuyển thành `"tentative"`.
     - Điều kiện xác nhận: Khi `score > confirmed_threshold` (0.8), track chuyển sang trạng thái `"confirmed"`.
   - **Giữ confirmed sau miss:**
     - Khi track bị miss trong FOV LiDAR, điểm bị trừ `score = score - 1/window`.
     - Tuy nhiên, nếu track đã đạt trạng thái `"confirmed"`, nó **không bị hạ cấp** về `"tentative"` hay `"initialized"` chỉ vì một hay vài lần miss, mà vẫn giữ nguyên trạng thái `"confirmed"`.
   - **Điều kiện xoá track (Deletion):** Track bị xoá nếu thoả mãn **bất kỳ** điều kiện nào sau:
     1. Phương sai vị trí ngang trong ma trận hiệp phương sai $P$ quá lớn: $P[0,0] > max\_P$ hoặc $P[1,1] > max\_P$ (với $max\_P = 9.0\text{ m}^2$).
     2. Track đã `"confirmed"` nhưng điểm tụt xuống dưới ngưỡng xoá: `score < delete_threshold` (0.6).
     3. Track chưa confirm (`state != "confirmed"`) nhưng điểm tụt xuống $\le 0$: `score <= 0.0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Antigravity AI, Claude Sonnet 4.6, Gemini 3.8 Flash
- Dùng cho phần nào (hàm, câu hỏi, debug): Triển khai các hàm trong Part E-H (kalman, camera_fusion, association, track_management), giải thích các câu hỏi kỹ thuật trong SUBMISSION.md, và debug cài đặt môi trường thư viện.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): Chạy toàn bộ bộ test tự kiểm tra (tất cả 122 tests pass), chạy pipeline Waymo 199 frames chế độ compare với seed 0, đối chiếu công thức EKF, Mahalanobis distance và Jacobian với tài liệu lý thuyết HUONG_DAN_KY_THUAT.md.

## Checklist nộp

- [x] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [x] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [x] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [x] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [x] Đã điền đủ file này, gồm khai báo AI
- [x] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [x] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [x] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
